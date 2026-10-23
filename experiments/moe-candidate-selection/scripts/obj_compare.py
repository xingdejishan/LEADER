"""Which objective's yaw minimum is closest to the truth?

For each frame, sweep yaw over +-15 deg and record where each objective puts its
minimum, then compare with the true yaw.  This decides whether a better cost can
deliver a large MOE gain at all: if the objective's argmin is nowhere near the
truth, no optimiser will fix it.

Objectives compared:
  p2p   the current one: mean truncated nearest-neighbour distance
  p2pl  point-to-plane with Huber weights, normals from the map itself
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

GRID = np.arange(-15.0, 15.001, 1.5)
TRUNC = 1.0
HUBER = 0.25
N_FRAMES = 50


def main():
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)

    RP._OPTS.update(gate2=0.01)
    RP.init()
    rng = np.random.default_rng(3)
    sample = sorted(rng.choice(len(gt), N_FRAMES, replace=False).tolist())

    rows = []
    for i in sample:
        _, T3, _ = RP.one(i)
        t, R0 = T3[:3, 3], T3[:3, :3]
        cfg = RP._G['cfg']
        mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:cfg.reference_count])
        tree = cKDTree(mp)
        scan = P.prepare_scan(RP._G['frames'][i], cfg)
        q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]
        ncache = {}

        c_p2p, c_p2pl, err = [], [], []
        for d in GRID:
            R = R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix()
            W = q @ R.T + t
            dd, idx = tree.query(W, k=1)
            c_p2p.append(float(np.minimum(dd, TRUNC).mean()))
            n = np.empty_like(W)
            for j, mi in enumerate(idx):
                v = ncache.get(int(mi))
                if v is None:
                    _, nb = tree.query(mp[int(mi)], k=10)
                    c = mp[nb] - mp[nb].mean(0)
                    _, vec = np.linalg.eigh(c.T @ c)
                    v = vec[:, 0].copy()
                    ncache[int(mi)] = v
                n[j] = v
            r = np.abs(np.einsum('ij,ij->i', n, W - mp[idx]))
            wt = np.where(r <= HUBER, 1.0, HUBER / np.maximum(r, 1e-9))
            c_p2pl.append(float((wt * np.minimum(r * r, 1.0)).sum() / wt.sum()))
            Rd = gt[i][:3, :3] @ R.T
            err.append(float(np.degrees(Rotation.from_matrix(Rd).as_rotvec())[2]))

        c_p2p = np.array(c_p2p); c_p2pl = np.array(c_p2pl); err = np.array(err)
        d_true = float(GRID[int(np.argmin(np.abs(err)))])
        rows.append(dict(i=int(i),
                         p2p=float(GRID[int(np.argmin(c_p2p))]),
                         p2pl=float(GRID[int(np.argmin(c_p2pl))]),
                         d_true=d_true,
                         err0=float(np.abs(err[GRID == 0][0])),
                         err_best=float(np.abs(err).min())))

    e0 = np.array([r['err0'] for r in rows])
    eb = np.array([r['err_best'] for r in rows])
    print('prior |yaw error| mean %.2f deg; if the sweep picked the true optimum: %.2f deg'
          % (e0.mean(), eb.mean()))
    for key in ('p2p', 'p2pl'):
        e = np.array([abs(r[key] - r['d_true']) for r in rows])
        print('%-5s  |objective argmin - best argmin|: mean %5.2f  median %5.2f  within 1 deg %2d/%d  within 2 deg %2d/%d  exact %2d'
              % (key, e.mean(), np.median(e), int((e <= 1).sum()), len(e),
                 int((e <= 2).sum()), len(e), int((e == 0).sum())))


if __name__ == '__main__':
    main()
