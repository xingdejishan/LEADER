"""Pilot: does a yaw search at the pipeline's own position move the pose toward the truth?

For a sample of frames:
  A  run the current pipeline (gate2 -> along-track x lateral search) -> T3
  B  coarse yaw sweep +-20 deg at 2 deg on 1000 points, then fine +-2 deg at 0.5 deg
     on 4000 points, at T3's translation (so translation, hence MPE, cannot change)
  C  accept the fine optimum only if it beats the current cost by a margin

Reports the true yaw error before and after, so we know whether the cost's
preference and the truth agree, before spending a full run.
"""
import importlib.util
import json
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

COARSE = np.arange(-20.0, 20.001, 2.0)
FINE = np.arange(-2.0, 2.001, 0.5)
COARSE_PTS = 1000
FINE_PTS = 4000
MARGIN = 0.002


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
    n = len(gt)
    sample = list(rng.choice(n, 120, replace=False))

    def cost(tree, q, R, t):
        return float(np.minimum(tree.query(q @ R.T + t, k=1, workers=1)[0], 1.0).mean())

    rows = []
    for i in sample:
        _, T3, _ = RP.one(i)
        t = T3[:3, 3]
        R0 = T3[:3, :3]
        mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:RP._G['cfg'].reference_count])
        tree = cKDTree(mp)
        scan = P.prepare_scan(RP._G['frames'][i], RP._G['cfg'])
        qc = scan if len(scan) <= COARSE_PTS else scan[::max(1, len(scan) // COARSE_PTS)][:COARSE_PTS]
        qf = scan if len(scan) <= FINE_PTS else scan[::max(1, len(scan) // FINE_PTS)][:FINE_PTS]

        cc = np.array([cost(tree, qc, R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix(), t)
                       for d in COARSE])
        d0 = float(COARSE[int(np.argmin(cc))])
        cf = np.array([cost(tree, qf, R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix(), t)
                       for d in (d0 + FINE)])
        k = int(np.argmin(cf))
        d_best = float(d0 + FINE[k])
        c_now = cost(tree, qf, R0, t)
        take = bool(cf[k] < c_now - MARGIN)

        def yaw_err(Rp):
            return float(np.degrees(Rotation.from_matrix(gt[i][:3, :3].T @ Rp).as_rotvec())[2])

        rows.append(dict(i=int(i), before=yaw_err(R0),
                         after=yaw_err(R0 @ Rotation.from_euler('z', d_best, degrees=True).as_matrix()),
                         d=d_best, take=take, c_now=c_now, c_best=float(cf[k]),
                         angle_before=float(full[i])))

    b = np.array([abs(r['before']) for r in rows])
    a = np.array([abs(r['after']) for r in rows])
    t = np.array([r['take'] for r in rows])
    print('sample %d frames (random over the hard set)' % len(rows))
    print('mean |yaw error|  before %.3f   after %.3f   (all frames, sweep applied regardless)' % (b.mean(), a.mean()))
    print('frames where the sweep reduced |yaw error|: %d   increased: %d' % (int((a < b - 1e-9).sum()), int((a > b + 1e-9).sum())))
    print('accepted by the cost margin: %d   of those, improved: %d   worsened: %d'
          % (int(t.sum()), int(((a < b - 1e-9) & t).sum()), int(((a > b + 1e-9) & t).sum())))
    if t.sum():
        print('mean |yaw error| on accepted frames: before %.3f  after %.3f' % (b[t].mean(), a[t].mean()))
    print('median chosen dyaw on accepted frames: %.1f deg' % (np.median([r['d'] for r in rows if r['take']]) if t.sum() else float('nan')))
    print()
    print('projected MOE if the sweep were applied everywhere: %.4f deg (baseline %.4f)'
          % ((full - b + a).mean(), full.mean()))
    print('projected MOE if only accepted frames change       : %.4f deg'
          % ((full - np.where(t, b - a, 0.0)).mean()))


if __name__ == '__main__':
    main()
