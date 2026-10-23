"""LEADER as the baseline: arbitrate between LEADER and its refinement with the map cost.

The two poses carry the same error (correlation 0.95); the refinement behaves like a
shrinkage toward ~1.8 deg, improving bad frames and hurting good ones.  So instead of
always taking the refined pose, pick per frame whichever of the two has the lower map
cost.  Truth-free, single frame, no other module.

    LEADER   ->  which pose
    V7       ->  which pose
    cost     ->  argmin of the truncated nearest-neighbour cost
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

TRUNC = 1.0
PTS = 2000


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    keys = [str(k) for k in w['hard_keys']]

    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    maps = F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                        for r in refs]), radius=cfg.crop_radius)

    def cost(scan, tree, T):
        q = scan if len(scan) <= PTS else scan[::max(1, len(scan) // PTS)][:PTS]
        W = q @ T[:3, :3].T + T[:3, 3]
        return float(np.minimum(tree.query(W, k=1, workers=1)[0], TRUNC).mean())

    pick = []
    chose_leader = 0
    for i in range(len(keys)):
        scan = P.prepare_scan(frames[i], cfg)
        t = v7[i][:3, 3]
        mp = maps.local(t, maps.nearby(t)[:cfg.reference_count])
        tree = cKDTree(mp)
        c_l = cost(scan, tree, lead[i])
        c_v = cost(scan, tree, v7[i])
        if c_l < c_v:
            pick.append(lead[i]); chose_leader += 1
        else:
            pick.append(v7[i])

    pick = np.stack(pick)
    print('frames %d   chose LEADER %d (%.0f%%)  chose V7 %d'
          % (len(keys), chose_leader, 100.0 * chose_leader / len(keys), len(keys) - chose_leader))
    su.evaluate(lead, None, w, 'LEADER only')
    su.evaluate(v7, None, w, 'V7 only')
    su.evaluate(pick, None, w, 'cost-arbitrated')


if __name__ == '__main__':
    main()
