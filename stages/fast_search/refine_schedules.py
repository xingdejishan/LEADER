"""Compare update schedules for the local refiner.

    joint   fit R and t together each round            (the stock behaviour)
    rot     fit R only, translation held at the input  (measured earlier)
    trans   fit t only, rotation held at the input
    alt     even rounds update R only, odd rounds update t only

All four use the same map, the same inlier gate (1.2 m), the same per-round clamp
(0.2 m / 0.5 deg) and the same validation gate (must improve by 1e-6, hits must not
drop).  Nothing else changes, so the schedule is the only difference.

    python refine_schedules.py ../results/fast.npz
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

MODES = ('joint', 'rot', 'trans', 'alt')
_G = {}


def init(poses):
    _G['w'] = L.World()
    _G['poses'] = poses


def _fit(A, B):
    a, b = A.mean(0), B.mean(0)
    U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
    if s[1] < 1e-8:
        return None, None
    corr = np.eye(3)
    corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ corr @ U.T
    return R, b - R @ a


def run_mode(tree, geom, q, T, mode):
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        W = val @ M[:3, :3].T + M[:3, 3]
        d = tree.query(W, k=1)[0]
        return float(np.minimum(d, 1.2).mean()), int((d < 0.6).sum())

    t = np.array(T, float)
    value, hit = obj(t)
    for it in range(6):
        if mode == 'joint':
            do_r, do_t = True, True
        elif mode == 'rot':
            do_r, do_t = True, False
        elif mode == 'trans':
            do_r, do_t = False, True
        else:
            do_r, do_t = (it % 2 == 0), (it % 2 == 1)
        W = train @ t[:3, :3].T + t[:3, 3]
        dist, idx = tree.query(W, k=1)
        ok = dist < 1.2
        if ok.sum() < 30:
            break
        origin = t[:3, 3].copy()
        R, tr = _fit(W[ok] - origin, geom[idx[ok]] - origin)
        if R is None:
            break
        vec = Rotation.from_matrix(R).as_rotvec()
        ang = np.linalg.norm(vec)
        trn = np.linalg.norm(tr)
        scale = min(1.0, 0.2 / max(trn, 1e-12), np.deg2rad(0.5) / max(ang, 1e-12))
        cand = t.copy()
        if do_r:
            cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        if do_t:
            cand[:3, 3] = origin + tr * scale
        nv, nh = obj(cand)
        if not (nv < value - 1e-6 and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return t


def one(i):
    w = _G['w']
    P = np.array(_G['poses'][i], float)
    mapp = w.local_map(i)[0]
    q = w.raw_query(i)
    q = q[np.linalg.norm(q, axis=1) <= 50]
    c = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(c, axis=0, return_index=True)
    q = q[np.sort(ids)]
    tree = cKDTree(mapp)
    return i, [run_mode(tree, mapp, q, P, m) for m in MODES]


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
    out = '/home/zhangcy/projects/e2e-fast-search-20261020-v1/results/'
    for k, m in enumerate(MODES):
        arr = np.empty_like(poses)
        for i, lst in res:
            arr[i] = lst[k]
        np.savez(out + 'sched_%s.npz' % m, keys=np.array(keys), poses=arr)
        print('wrote', m)
