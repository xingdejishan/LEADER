"""Where is LEADER's own yaw error coming from: ambiguity, or its estimator?

Two questions, both answerable from the frozen data:
  A  does LEADER's yaw error concentrate where the map is along-track ambiguous?
     (ratio < 0.21 = the gate's definition of "two competing explanations")
  B  on the frames where LEADER is badly wrong, is the true yaw a clear minimum at
     LEADER's own translation?  If yes, LEADER is leaving information on the table and
     the fix is inside its estimator; if no, the information is not there.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F
import pipeline as P

GRID = np.arange(-20.0, 20.001, 2.0)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    keys = [str(k) for k in w['hard_keys']]
    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    yp = np.degrees(np.arctan2(lead[:, 1, 0], lead[:, 0, 0]))
    yg = np.degrees(np.arctan2(gt[:, 1, 0], gt[:, 0, 0]))
    e = (yp - yg + 180.0) % 360.0 - 180.0
    ae = np.abs(e)

    ratio = np.array([json.loads(l).get('ratio', np.nan) for l in
                      open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))])
    g = np.isfinite(ratio) & (ratio < 0.21)
    print('A) ambiguity vs LEADER yaw error')
    print('   gated frames n=%d  mean |yaw| %.2f  median %.2f'
          % (g.sum(), ae[g].mean(), np.median(ae[g])))
    print('   other frames n=%d  mean |yaw| %.2f  median %.2f'
          % ((~g).sum(), ae[~g].mean(), np.median(ae[~g])))
    print('   correlation of |yaw error| with ratio: r = %+.3f'
          % np.corrcoef(ae[np.isfinite(ratio)], ratio[np.isfinite(ratio)])[0, 1])

    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    maps = F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                        for r in refs]), radius=cfg.crop_radius)

    pick = list(np.flatnonzero(ae > 5.0))
    print('\nB) is the true yaw identifiable at LEADER\'s own translation?  (%d frames with |yaw|>5)' % len(pick))
    hits = 0
    for i in pick[:40]:
        t = lead[i][:3, 3]
        mp = maps.local(t, maps.nearby(t)[:cfg.reference_count])
        if len(mp) < 1000:
            continue
        tree = cKDTree(mp)
        scan = P.prepare_scan(frames[i], cfg)
        q = scan if len(scan) <= 2000 else scan[::max(1, len(scan) // 2000)][:2000]
        costs = []
        for d in GRID:
            R = gt[i][:3, :3] @ Rotation.from_euler('z', d, degrees=True).as_matrix()
            costs.append(float(np.minimum(tree.query(q @ R.T + t, k=1, workers=1)[0], 1.0).mean()))
        costs = np.array(costs)
        best = GRID[int(np.argmin(costs))]
        if abs(best) <= 2:
            hits += 1
    print('   truth is a local minimum (within 2 deg) at LEADER position: %d of %d'
          % (hits, min(40, len(pick))))


if __name__ == '__main__':
    main()
