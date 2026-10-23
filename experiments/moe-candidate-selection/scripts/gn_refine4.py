"""Point-to-plane Gauss-Newton refinement, 4-DOF (x, y, z, yaw) or 6-DOF.

Literature it follows:
  * Chen & Medioni 1991 / Low 2004        point-to-plane linearised ICP
  * Segal, Haehnel & Thrun, RSS 2009      Generalized-ICP
  * Censi, ICRA 2007                      pose covariance from the Hessian
  * LeGO-LOAM, IROS 2018                  ground-plane / 2-D motion constraint:
                                          solve yaw + translation, let the vehicle's
                                          roll and pitch come from the prior
  * Magnusson, NDT, 2009                  the voxelised-Gaussian alternative

The 4-DOF variant is the point: a planar vehicle on a locally flat road has a
well-defined yaw but poorly identified roll/pitch from a single scan, and the
Hessian of a point-to-plane cost on a voxelised map is exactly the quantity that
says so.  The 6-DOF variant is kept for the ablation.
"""
import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import run_pipeline as RP
import pipeline as P

TRUNC = 1.0
HUBER = 0.25
MAX_ROT = np.deg2rad(1.0)
MAX_TRANS = 0.30
LAMBDA = 1e-4
_G = {}


def init(iters, dof):
    RP._OPTS.update(gate2=0.01)
    RP.init()
    _G['iters'] = iters
    _G['dof'] = dof


def solve(tree, mp, q, T, iters, dof, ncache):
    T = np.array(T, float)
    used = 0
    for _ in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        W = q @ R.T + t
        d, idx = tree.query(W, k=1)
        ok = d < TRUNC
        if ok.sum() < 50:
            break
        Wk, mk, ik = W[ok], mp[idx[ok]], idx[ok]
        n = np.empty_like(Wk)
        for j, mi in enumerate(ik):
            key = int(mi)
            v = ncache.get(key)
            if v is None:
                _, nb = tree.query(mp[key], k=10)
                c = mp[nb] - mp[nb].mean(0)
                _, vec = np.linalg.eigh(c.T @ c)
                v = vec[:, 0].copy()
                ncache[key] = v
            n[j] = v
        r = np.einsum('ij,ij->i', n, Wk - mk)
        a = np.abs(r)
        wgt = np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-9))
        cross = np.cross(Wk, n)
        if dof == 4:
            J = np.stack([cross[:, 2], n[:, 0], n[:, 1], n[:, 2]], axis=1)
        else:
            J = np.hstack([cross, n])
        Jw = J * wgt[:, None]
        H = Jw.T @ J
        b = Jw.T @ r
        H += np.eye(H.shape[0]) * (LAMBDA * np.trace(H) / H.shape[0] + 1e-9)
        try:
            delta = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break
        if dof == 4:
            rot = np.array([0.0, 0.0, delta[0]])
            tr = delta[1:4]
        else:
            rot, tr = delta[:3], delta[3:6]
        nr = np.linalg.norm(rot)
        if nr > MAX_ROT:
            rot *= MAX_ROT / nr
        nt = np.linalg.norm(tr)
        if nt > MAX_TRANS:
            tr *= MAX_TRANS / nt
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
        dT[:3, 3] = tr
        T = dT @ T
        used += 1
        if np.linalg.norm(rot) < 1e-4 and np.linalg.norm(tr) < 1e-3:
            break
    return T, used


def one(i):
    _, T3, diag = RP.one(i)
    cfg = RP._G['cfg']
    t = T3[:3, 3]
    mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    scan = P.prepare_scan(RP._G['frames'][i], cfg)
    q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]
    T, used = solve(tree, mp, q, T3, _G['iters'], _G['dof'], {})
    return i, T3, T, dict(key=diag['key'], steps=used)


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--dof', type=int, default=4, choices=(4, 6))
    a = ap.parse_args()

    init(a.iters, a.dof)
    n = len(RP._G['frames'])
    rng = np.random.default_rng(5)
    idx = sorted(rng.choice(n, min(a.frames, n), replace=False).tolist())
    with Pool(a.workers, initializer=init, initargs=(a.iters, a.dof)) as p:
        res = p.map(_w, idx, chunksize=4)

    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    pos = {k: j for j, k in enumerate(w['hard_keys'])}
    keys = [RP._G['frames'][i]['key'] for i, _, _, _ in res]
    g = np.stack([gt[pos[str(k)]] for k in keys])

    def stats_of(poses, tag):
        t = np.linalg.norm(poses[:, :3, 3] - g[:, :3, 3], axis=1)
        Rd = np.swapaxes(poses[:, :3, :3], -1, -2) @ g[:, :3, :3]
        r = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
        print('%-12s n=%d  MPE %.4f  MOE %.4f  medT %.4f  medR %.4f  ok1/2 %d'
              % (tag, len(t), t.mean(), r.mean(), np.median(t), np.median(r),
                 int(((t <= 1) & (r <= 2)).sum())))

    stats_of(np.stack([r[1] for r in res]), 'search')
    stats_of(np.stack([r[2] for r in res]), 'GN %d-DOF' % a.dof)
    st = np.array([r[3]['steps'] for r in res])
    print('iterations used: median %d max %d (budget %d)' % (int(np.median(st)), int(st.max()), a.iters))


if __name__ == '__main__':
    main()
