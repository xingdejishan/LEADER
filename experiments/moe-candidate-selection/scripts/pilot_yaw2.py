"""Pilot 2: does the map's yaw preference predict the true yaw change?

Same 120-frame sample, three variants:
  V1  yaw sweep at the search pose (as pilot 1)
  V2  stricter acceptance margins
  V3  alternating: yaw sweep -> re-run the along-track search -> yaw sweep again

Reports the correlation between the cost improvement and the true yaw improvement,
because no acceptance rule can beat that correlation.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import run_pipeline as RP
import pipeline as P
import fast_lib as F

COARSE = np.arange(-20.0, 20.001, 2.0)
FINE = np.arange(-2.0, 2.001, 0.5)
CP, FP = 1500, 4000


def make_cost(maps, cfg, t):
    mp = maps.local(t, maps.nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    return tree, mp


def sweep(tree, qc, qf, R0, t):
    cc = np.array([float(np.minimum(tree.query(qc @ (R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix()).T + t, k=1, workers=1)[0], 1.0).mean()) for d in COARSE])
    d0 = float(COARSE[int(np.argmin(cc))])
    cf = np.array([float(np.minimum(tree.query(qf @ (R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix()).T + t, k=1, workers=1)[0], 1.0).mean()) for d in (d0 + FINE)])
    k = int(np.argmin(cf))
    c_now = float(np.minimum(tree.query(qf @ R0.T + t, k=1, workers=1)[0], 1.0).mean())
    return float(d0 + FINE[k]), c_now, float(cf[k])


def main():
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    base = np.asarray(w['hard_base'], float)
    RP._OPTS.update(gate2=0.01)
    RP.init()

    Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    full = np.linalg.norm(rv, axis=1)

    rng = np.random.default_rng(11)
    sample = list(rng.choice(len(gt), 120, replace=False))

    rows = []
    for i in sample:
        _, T3, _ = RP.one(i)
        t, R0 = T3[:3, 3], T3[:3, :3]
        cfg = RP._G['cfg']
        scan = P.prepare_scan(RP._G['frames'][i], cfg)
        qc = scan if len(scan) <= CP else scan[::max(1, len(scan) // CP)][:CP]
        qf = scan if len(scan) <= FP else scan[::max(1, len(scan) // FP)][:FP]
        tree, _ = make_cost(RP._G['maps'], cfg, t)
        d1, c0, c1 = sweep(tree, qc, qf, R0, t)

        # V3: apply the yaw, re-run the along-track search from the rotated pose, sweep again
        R1 = R0 @ Rotation.from_euler('z', d1, degrees=True).as_matrix()
        T_rot = np.eye(4)
        T_rot[:3, :3], T_rot[:3, 3] = R1, t
        pose2, _ = P.localize(RP._G['frames'][i], T_rot, RP._G['maps'], cfg)
        T2 = np.asarray(pose2, float)
        tree2, _ = make_cost(RP._G['maps'], cfg, T2[:3, 3])
        d2, _, _ = sweep(tree2, qc, qf, T2[:3, :3], T2[:3, 3])

        def yaw(R):
            return float(np.degrees(Rotation.from_matrix(gt[i][:3, :3].T @ R).as_rotvec())[2])

        rows.append(dict(i=int(i), before=yaw(R0), v1=yaw(R1),
                         v3=yaw(T2[:3, :3] @ Rotation.from_euler('z', d2, degrees=True).as_matrix()),
                         dcost=c0 - c1, angle=float(full[i])))

    b = np.array([abs(r['before']) for r in rows])
    v1 = np.array([abs(r['v1']) for r in rows])
    v3 = np.array([abs(r['v3']) for r in rows])
    dc = np.array([r['dcost'] for r in rows])

    print('sample %d' % len(rows))
    print('V1 mean |yaw| %.3f -> %.3f   improved %d  worsened %d'
          % (b.mean(), v1.mean(), int((v1 < b).sum()), int((v1 > b).sum())))
    print('V3 mean |yaw| %.3f -> %.3f   improved %d  worsened %d'
          % (b.mean(), v3.mean(), int((v3 < b).sum()), int((v3 > b).sum())))
    print()
    print('correlation between cost improvement and |yaw| improvement: r = %.3f'
          % np.corrcoef(dc, b - v1)[0, 1])
    print('correlation between cost improvement and signed yaw change toward truth: r = %.3f'
          % np.corrcoef(dc, b - v1)[0, 1])
    print()
    print('%-8s %8s %8s %10s %10s' % ('margin', 'accept', 'improved', 'mean d|yaw|', 'proj MOE'))
    for m in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        t = dc > m
        delta = np.where(t, b - v1, 0.0)
        proj = full.mean() - delta.mean()
        print('%-8.3f %8d %8d %10.3f %10.4f' % (m, int(t.sum()), int(((b - v1) > 0)[t].sum()),
                                                float((b - v1)[t].mean()) if t.sum() else float('nan'), proj))
    import json
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pilot2.json'), 'w'))


if __name__ == '__main__':
    main()
