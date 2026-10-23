"""Point-to-plane Gauss-Newton refinement (no search).

Literature it follows:
  * Chen & Medioni 1991 / Low 2004   point-to-plane linearised ICP
  * Segal, Haehnel & Thrun, RSS 2009 Generalized-ICP (plane-to-plane)
  * Censi, ICRA 2007                 pose covariance from the Hessian -> which DOF
                                     are actually observable from this scan
  * Zhang et al. / LeGO-LOAM IROS 2018  ground-plane constraint for roll/pitch

Rather than sweeping a grid, this linearises the point-to-plane residual once per
iteration and solves a 6x6 system.  Normals are computed lazily on the map points
that actually get matched, so the cost stays proportional to the query size.

    python gn_refine.py --frames 300 --iters 8
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
MAX_ROT_STEP = np.deg2rad(1.0)
MAX_TRANS_STEP = 0.20
_G = {}


def init(iters):
    RP._OPTS.update(gate2=0.01)
    RP.init()
    _G['iters'] = iters


def point_to_plane(tree, mp, q, T, iters, ncache):
    T = np.array(T, float)
    R0, t0 = T[:3, :3].copy(), T[:3, 3].copy()
    hist = []
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
                w, vec = np.linalg.eigh(c.T @ c)
                v = vec[:, 0].copy()
                ncache[key] = v
            n[j] = v
        r = np.einsum('ij,ij->i', n, Wk - mk)
        # robust weight (Huber)
        a = np.abs(r)
        wgt = np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-9))
        J = np.hstack([np.cross(Wk, n), n])
        Jw = J * wgt[:, None]
        H = Jw.T @ J
        b = Jw.T @ r
        H += np.eye(6) * 1e-6 * np.trace(H) / 6.0
        try:
            delta = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break
        rot = delta[:3]
        nr = np.linalg.norm(rot)
        if nr > MAX_ROT_STEP:
            rot = rot * (MAX_ROT_STEP / nr)
        tr = delta[3:]
        nt = np.linalg.norm(tr)
        if nt > MAX_TRANS_STEP:
            tr = tr * (MAX_TRANS_STEP / nt)
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
        dT[:3, 3] = tr
        T = dT @ T
        hist.append((float(np.linalg.norm(rot)), float(np.linalg.norm(tr)), int(ok.sum())))
        if np.linalg.norm(rot) < 1e-4 and np.linalg.norm(tr) < 1e-3:
            break
    _ = (R0, t0)
    return T, hist


def one(i):
    _, T3, diag = RP.one(i)
    cfg = RP._G['cfg']
    t = T3[:3, 3]
    mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    scan = P.prepare_scan(RP._G['frames'][i], cfg)
    q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]
    T, hist = point_to_plane(tree, mp, q, T3, _G['iters'], {})
    return i, T3, T, dict(key=diag['key'], steps=len(hist),
                          last_rot=hist[-1][0] if hist else 0.0,
                          last_trans=hist[-1][1] if hist else 0.0,
                          inliers=hist[-1][2] if hist else 0)


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--iters', type=int, default=8)
    a = ap.parse_args()

    init(a.iters)
    n = len(RP._G['frames'])
    rng = np.random.default_rng(5)
    idx = sorted(rng.choice(n, min(a.frames, n), replace=False).tolist())
    with Pool(a.workers, initializer=init, initargs=(a.iters,)) as p:
        res = p.map(_w, idx, chunksize=4)

    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    pos = {k: j for j, k in enumerate(w['hard_keys'])}

    def stats_of(poses, tag):
        g = np.stack([gt[pos[str(k)]] for k in T3keys])
        t = np.linalg.norm(poses[:, :3, 3] - g[:, :3, 3], axis=1)
        Rd = np.swapaxes(poses[:, :3, :3], -1, -2) @ g[:, :3, :3]
        r = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
        print('%-10s n=%d   MPE %.4f   MOE %.4f   medT %.4f  medR %.4f   ok1/2 %d'
              % (tag, len(t), t.mean(), r.mean(), np.median(t), np.median(r),
                 int(((t <= 1) & (r <= 2)).sum())))

    T3keys = [RP._G['frames'][i]['key'] for i, _, _, _ in res]
    stats_of(np.stack([r[1] for r in res]), 'search')
    stats_of(np.stack([r[2] for r in res]), 'GN p2plane')
    steps = np.array([r[3]['steps'] for r in res])
    print('GN iterations used: median %d, max %d' % (int(np.median(steps)), int(steps.max())))


if __name__ == '__main__':
    main()
