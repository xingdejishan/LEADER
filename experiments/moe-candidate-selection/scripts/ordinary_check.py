"""Does the margin rule ever fire on frames where V7 is already accurate?

The ordinary 1000-frame set has no visual candidate table, so the closest available
test is the sub-population of the hard set where V7 is already good.  If the rule
fires there, it is a false positive and the frame can only get worse.
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

    pos = np.linalg.norm(v7[:, :3, 3] - gt[:, :3, 3], axis=1)
    rot = np.array([rot_err(v7[i][:3, :3], gt[i][:3, :3]) for i in range(len(gt))])

    fired = np.zeros(len(gt), bool)
    cand = {}
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
        cand[i] = best
        if best is not None and base is not None and best[0] > float(base) + MARGIN:
            fired[i] = True

    print('%-14s %6s %6s %8s %10s %10s' % ('V7 position', 'n', 'fired', 'rate', 'V7 pos err', 'cand pos err'))
    for lo, hi, tag in ((0, 1, '<=1 m'), (1, 2, '1-2 m'), (2, 5, '2-5 m'), (5, 20, '5-20 m'), (20, 1e9, '>20 m')):
        m = (pos > lo) & (pos <= hi) if lo else (pos <= hi)
        if m.sum() == 0:
            continue
        c = np.array([np.linalg.norm(cand[i][1][:3, 3] - gt[i][:3, 3]) if cand[i] else np.nan
                      for i in range(len(gt))])[m]
        print('%-14s %6d %6d %7.1f%% %10.3f %10.3f'
              % (tag, int(m.sum()), int(fired[m].sum()), 100.0 * fired[m].mean(),
                 pos[m].mean(), np.nanmean(c)))

    print()
    good = np.where(pos <= 1.0)[0]
    fire_good = [i for i in good if fired[i]]
    print('frames with V7 position <= 1 m: %d, of which the rule fires on %d (%.1f%%)'
          % (len(good), len(fire_good), 100.0 * len(fire_good) / max(1, len(good))))
    if fire_good:
        dp = [np.linalg.norm(cand[i][1][:3, 3] - gt[i][:3, 3]) - pos[i] for i in fire_good]
        dr = [rot_err(cand[i][1][:3, :3], gt[i][:3, :3]) - rot[i] for i in fire_good]
        print('  on those %d frames, switching to the candidate would change position by %+.3f m '
              '(median %+.3f) and rotation by %+.3f deg (median %+.3f)'
              % (len(fire_good), float(np.mean(dp)), float(np.median(dp)),
                 float(np.mean(dr)), float(np.median(dr))))
        print('  frames made worse in position: %d   worse in rotation: %d'
              % (int(np.sum(np.array(dp) > 0)), int(np.sum(np.array(dr) > 0))))


if __name__ == '__main__':
    main()
