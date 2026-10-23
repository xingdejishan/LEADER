"""MOE only: when V7's rotation is already good, does the visual rotation help?

Bins every frame by the baseline's own rotation error and reports, per bin, what the
visual candidate's rotation would do.  This is the generalisation question for MOE:
ordinary frames sit near 0.9 deg, so the <=1 deg bins are the proxy for them.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


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

    base_r = np.array([rot_err(v7[i][:3, :3], gt[i][:3, :3]) for i in range(len(gt))])
    cand_r = np.full(len(gt), np.nan)
    score_gap = np.full(len(gt), np.nan)
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
        if best is not None:
            cand_r[i] = rot_err(best[1][:3, :3], gt[i][:3, :3])
            if base is not None:
                score_gap[i] = best[0] - float(base)

    print('%-16s %5s %9s %9s %10s %12s %12s' %
          ('V7 rotation', 'n', 'V7 MOE', 'cand MOE', 'cand better', 'swap all', 'swap if gap>0.1'))
    edges = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 1e9)]
    for lo, hi in edges:
        m = (base_r > lo) & (base_r <= hi) & np.isfinite(cand_r)
        if m.sum() < 5:
            continue
        swap_all = np.where(m, cand_r, base_r).mean()
        sel = m & np.isfinite(score_gap) & (score_gap > 0.10)
        swap_sel = np.where(sel, cand_r, base_r).mean()
        print('%-16s %5d %9.3f %9.3f %9.0f%% %12.3f %12.3f'
              % ('%g-%g deg' % (lo, hi), int(m.sum()), base_r[m].mean(), cand_r[m].mean(),
                 100 * (cand_r[m] < base_r[m]).mean(), swap_all, swap_sel))
        if sel.sum():
            print('%-16s %5s %9s %9s %9s  fired on %d frames, their rotation %.2f -> %.2f'
                  % ('', '', '', '', '', int(sel.sum()), base_r[sel].mean(), cand_r[sel].mean()))

    print()
    print('overall V7 MOE %.4f' % base_r.mean())


if __name__ == '__main__':
    main()
