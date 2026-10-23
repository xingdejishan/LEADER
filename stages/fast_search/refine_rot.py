"""Rotation-only refinement: translation held fixed, rotation fitted.

Mirror of the along-track search: that one fixes rotation and searches translation;
this one fixes translation and estimates rotation.  Rotation needs no search — a rigid
fit on the inlier correspondences gives it directly, so the stage is cheap.

    python refine_rot.py ../results/fast.npz ../results/fast_rot.npz
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


def init(poses):
    _G['w'] = L.World()
    _G['poses'] = poses


def fit_rot(source, target):
    a, b = source.mean(0), target.mean(0)
    u, s, vt = np.linalg.svd((source - a).T @ (target - b))
    if s[1] < 1e-8:
        return None
    corr = np.eye(3)
    corr[-1, -1] = np.linalg.det(vt.T @ u.T)
    return vt.T @ corr @ u.T


def one(i):
    w = _G['w']
    t = np.array(_G['poses'][i], float)
    mapp = w.local_map(i)[0]
    q = w.raw_query(i)
    q = q[np.linalg.norm(q, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]
    tree = cKDTree(mapp)
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]
    def obj(P):
        W = val @ P[:3, :3].T + P[:3, 3]
        d = tree.query(W, k=1)[0]
        return float(np.minimum(d, TRUNC).mean()), int((d < 0.6).sum())
    value, hit = obj(t)
    centre = t[:3, 3].copy()
    for _ in range(6):
        W = train @ t[:3, :3].T + centre
        dist, idx = tree.query(W, k=1)
        ok = dist < TRUNC
        if ok.sum() < 30:
            break
        R = fit_rot(W[ok] - centre, mapp[idx[ok]] - centre)
        if R is None:
            break
        vec = Rotation.from_matrix(R).as_rotvec()
        ang = np.linalg.norm(vec)
        scale = min(1.0, np.deg2rad(0.5) / max(ang, 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre                      # translation never moves
        nv, nh = obj(cand)
        if not (nv < value - 1e-6 and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return i, t


def _w(i):
    return one(i)


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    z = np.load(src, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    poses = np.asarray(z['poses'], float)
    init(poses)
    with Pool(12, initializer=init, initargs=(poses,)) as p:
        res = p.map(_w, range(len(keys)), chunksize=2)
    out = np.empty_like(poses)
    moved = 0
    for i, T in res:
        out[i] = T
        moved += int(np.linalg.norm(T[:3, 3] - poses[i, :3, 3]) > 1e-9)
    np.savez(dst, keys=np.array(keys), poses=out)
    print('wrote', dst, 'translation moved on', moved, 'frames (must be 0)')
