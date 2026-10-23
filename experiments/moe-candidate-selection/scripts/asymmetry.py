"""What the rotation transplant actually exploits: measure the asymmetry directly.

On the frames the rule selects, compare the visual candidate with the baseline on the
two halves separately: position error and rotation error.  If the candidate's rotation
beats the baseline's while its position loses, the gain rests on a conditioning
asymmetry between the two halves of PnP, not on the candidate being better overall.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
MARGIN = 0.10


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    def rot_err(a, b):
        Rd = np.swapaxes(a, -1, -2) @ b
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec())))

    sel = []
    for i in range(len(gt)):
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        if best is not None and base is not None and best[0] > float(base) + MARGIN:
            sel.append((i, best[1]))

    print('selected frames: %d' % len(sel))
    dp, dr = [], []
    cp, cr = [], []
    for i, C in sel:
        dp.append(np.linalg.norm(v7[i][:3, 3] - gt[i][:3, 3]))
        dr.append(rot_err(v7[i][:3, :3], gt[i][:3, :3]))
        cp.append(np.linalg.norm(C[:3, 3] - gt[i][:3, 3]))
        cr.append(rot_err(C[:3, :3], gt[i][:3, :3]))
    dp, dr, cp, cr = map(np.array, (dp, dr, cp, cr))

    print('%-22s %10s %10s' % ('', 'baseline', 'candidate'))
    print('%-22s %10.3f %10.3f' % ('position error mean (m)', dp.mean(), cp.mean()))
    print('%-22s %10.3f %10.3f' % ('position error median (m)', np.median(dp), np.median(cp)))
    print('%-22s %10.3f %10.3f' % ('rotation error mean (deg)', dr.mean(), cr.mean()))
    print('%-22s %10.3f %10.3f' % ('rotation error median (deg)', np.median(dr), np.median(cr)))
    print()
    print('candidate rotation better on %d/%d (%.0f%%)' % (int((cr < dr).sum()), len(sel), 100 * (cr < dr).mean()))
    print('candidate position better on %d/%d (%.0f%%)' % (int((cp < dp).sum()), len(sel), 100 * (cp < dp).mean()))
    print()
    print('correlation between the two halves across frames: r(pos gain, rot gain) = %.3f'
          % np.corrcoef(dp - cp, dr - cr)[0, 1])
    print('ratio of candidate-to-baseline error: position %.2f   rotation %.2f'
          % (cp.mean() / dp.mean(), cr.mean() / dr.mean()))


if __name__ == '__main__':
    main()
