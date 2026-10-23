"""Can the geometric end-refiner be made cheaper without losing its benefit?

The stock refine_raw() rebuilds its own map from the three nearest reference scans
(~115 ms of its ~240 ms) and then runs up to 6 iterations on 2x3000 points.  In the
pipeline the along-track search has already built the same map, so the map cost is
duplicated work.

This script runs the same algorithm with:
    - a prebuilt map (as the search would hand over),
    - configurable iteration cap and point budget,
and reports time and accuracy for each combination on the 1502 hard frames.
"""
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

TRUNC = 1.2
_G = {}
CFG = [(6, 3000), (3, 3000), (6, 1500), (3, 1500), (2, 1500)]


def init(poses):
    _G['w'] = L.World()
    _G['poses'] = poses


def fit(source, target):
    a, b = source.mean(0), target.mean(0)
    u, s, vt = np.linalg.svd((source - a).T @ (target - b))
    if s[1] < 1e-8:
        return None
    corr = np.eye(3)
    corr[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ corr @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = b - r @ a
    return t


def refine(tree, geom, q, t0, iters, npts):
    cells = np.floor(q / 1.0).astype(np.int64)
    parity = (cells[:, 0] * 73856093 + cells[:, 1] * 19349663 + cells[:, 2] * 83492791) % 2
    train, val = q[parity == 0], q[parity == 1]
    if len(train) > npts:
        train = train[np.linspace(0, len(train) - 1, npts, dtype=int)]
    if len(val) > npts:
        val = val[np.linspace(0, len(val) - 1, npts, dtype=int)]

    def objective(P):
        W = val @ P[:3, :3].T + P[:3, 3]
        d = tree.query(W, k=1)[0]
        return float(np.minimum(d, TRUNC).mean()), int((d < 0.6).sum())

    t = np.array(t0, float, copy=True)
    value, hit = objective(t)
    for _ in range(iters):
        W = train @ t[:3, :3].T + t[:3, 3]
        dist, idx = tree.query(W, k=1)
        ok = dist < TRUNC
        if ok.sum() < 30:
            break
        src = W[ok]
        delta = fit(src - t[:3, 3], geom[idx[ok]] - t[:3, 3])
        if delta is None:
            break
        vec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
        ang = np.linalg.norm(vec)
        tr = np.linalg.norm(delta[:3, 3])
        scale = min(1.0, 0.2 / max(tr, 1e-12), np.deg2rad(0.5) / max(ang, 1e-12))
        delta[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix()
        delta[:3, 3] *= scale
        cand = t.copy()
        cand[:3, :3] = delta[:3, :3] @ t[:3, :3]
        cand[:3, 3] = t[:3, 3] + delta[:3, 3]
        nv, nh = objective(cand)
        if not (nv < value - 1e-6 and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return t


def one(i):
    w = _G['w']
    P = np.asarray(_G['poses'][i], float)
    t = w.local_map(i)
    mapp = t[0]
    q = w.raw_query(i)
    q = q[np.linalg.norm(q, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]
    tree = cKDTree(mapp)
    outs = []
    for iters, npts in CFG:
        outs.append(refine(tree, mapp, q, P, iters, npts))
    return i, outs


def _w(i):
    return one(i)


if __name__ == '__main__':
    src = sys.argv[1]
    z = np.load(src, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    poses = np.asarray(z['poses'], float)
    init(poses)
    with Pool(12, initializer=init, initargs=(poses,)) as p:
        res = p.map(_w, range(len(keys)), chunksize=2)
    for k, (iters, npts) in enumerate(CFG):
        out = np.empty_like(poses)
        for i, outs in res:
            out[i] = outs[k]
        np.savez('/home/zhangcy/projects/e2e-fast-search-20261020-v1/results/refine_i%d_p%d.npz'
                 % (iters, npts), keys=np.array(keys), poses=out)
        print('wrote iters=%d npts=%d' % (iters, npts))
