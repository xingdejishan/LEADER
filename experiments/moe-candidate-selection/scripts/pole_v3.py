"""Column-based pole detector, then the feasibility estimate.

Detector: after removing the ground, each 0.2 m XY cell records its vertical extent and
point count.  A pole is a small connected group of cells (footprint <= 0.8 m) with a
large vertical extent (>= 1.5 m).  Walls and tree crowns have large footprints and are
rejected; the ground is already gone, so nothing merges through the road.

    python pole_v3.py count      # how many poles per frame
    python pole_v3.py estimate   # the yaw experiment
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
CELL = 0.2
RANGE = 35.0
MIN_EXTENT = 1.5
MAX_FOOT = 0.8
MIN_CELLS = 4
GROUND_BAND = 0.25
ASSOC = 2.5
N_REF = 6
ITERS = 6


def ground_z(points):
    key = (np.floor(points[:, 0]).astype(np.int64) * 100000
           + np.floor(points[:, 1]).astype(np.int64))
    order = np.argsort(key)
    ks = key[order]
    zs = points[order, 2]
    uniq, start = np.unique(ks, return_index=True)
    bounds = np.append(start, len(ks))
    g = np.empty(len(uniq))
    for k in range(len(uniq)):
        seg = zs[bounds[k]:bounds[k + 1]]
        g[k] = np.percentile(seg, 10) if len(seg) >= 4 else seg.min()
    return np.array([dict(zip(uniq.tolist(), g.tolist()))[k] for k in key])


def poles_of(points):
    if len(points) < 200:
        return np.zeros((0, 3))
    pts = points[(points[:, 2] - ground_z(points)) > GROUND_BAND]
    if len(pts) < 100:
        return np.zeros((0, 3))
    ij = np.floor(pts[:, :2] / CELL).astype(np.int64)
    ij -= ij.min(0)                      # non-negative indices, so the packing is unambiguous
    key = ij[:, 0] * 100000 + ij[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(uniq), np.inf)
    zmax = np.full(len(uniq), -np.inf)
    cnt = np.zeros(len(uniq), int)
    np.minimum.at(zmin, inv, pts[:, 2])
    np.maximum.at(zmax, inv, pts[:, 2])
    np.add.at(cnt, inv, 1)
    ext = zmax - zmin
    ok = (ext >= MIN_EXTENT) & (cnt >= 5)
    if ok.sum() == 0:
        return np.zeros((0, 3))
    gi = np.zeros((ij[:, 0].max() + 2, ij[:, 1].max() + 2), bool)
    ci, cj = np.divmod(uniq, 100000)
    gi[ci[ok], cj[ok]] = True
    lab, n = ndimage.label(gi, structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        m = lab[ci, cj] == k
        if m.sum() < MIN_CELLS:
            continue
        sub = pts[np.isin(inv, np.flatnonzero(ok & m))]
        if len(sub) < 15:
            continue
        foot = np.linalg.norm(sub[:, :2].max(0) - sub[:, :2].min(0))
        if foot > MAX_FOOT:
            continue
        out.append([sub[:, 0].mean(), sub[:, 1].mean(), float(sub[:, 2].max() - sub[:, 2].min())])
    return np.asarray(out) if out else np.zeros((0, 3))


def umeyama2d(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'count'
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    keys = [str(k) for k in w['hard_keys']]
    rng = np.random.default_rng(59)
    sample = sorted(rng.choice(len(keys), 12, replace=False).tolist())

    counts = []
    for i in sample:
        date, stamp = keys[i].split('/')
        p = os.path.join(DATA, date, 'velodyne_sync', stamp + '.bin')
        if not os.path.isfile(p):
            continue
        q = F.decode(p)
        q = q[np.linalg.norm(q[:, :2], axis=1) <= RANGE]
        pl = poles_of(q)
        counts.append(len(pl))
        if mode == 'count':
            print('frame %5d  poles %d' % (i, len(pl)))
    if mode == 'count':
        c = np.array(counts)
        print('median %.0f   frames with >=3: %d/%d' % (np.median(c), int((c >= 3).sum()), len(c)))
        return

    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    rpos = np.asarray([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs])
    rtree = cKDTree(rpos)
    ref_paths = [r['scan_path'] for r in refs]
    ref_poses = [np.asarray(r['world_from_body'], float) for r in refs]
    wrap = lambda x: np.degrees(np.arctan2(np.sin(x), np.cos(x)))

    rows = []
    for i in sample:
        date, stamp = keys[i].split('/')
        p = os.path.join(DATA, date, 'velodyne_sync', stamp + '.bin')
        if not os.path.isfile(p):
            continue
        q = F.decode(p)
        q = q[np.linalg.norm(q[:, :2], axis=1) <= RANGE]
        pq = poles_of(q)
        if len(pq) < 3:
            rows.append(dict(ok=False))
            continue
        t7, R7 = v7[i][:3, 3], v7[i][:3, :3]
        _, nb = rtree.query(t7, k=N_REF)
        mw = []
        for j in np.atleast_1d(nb):
            pr = poles_of(F.decode(ref_paths[int(j)]))
            if len(pr):
                T = ref_poses[int(j)]
                mw.append(pr[:, :2] @ T[:3, :3][:2, :2].T + T[:2, 3])
        mp = np.vstack(mw) if mw else np.zeros((0, 2))
        mp = mp[np.linalg.norm(mp - t7[:2], axis=1) <= RANGE] if len(mp) else mp
        if len(mp) < 3:
            rows.append(dict(ok=False))
            continue
        mt = cKDTree(mp)
        yaw = np.arctan2(R7[1, 0], R7[0, 0])
        t = t7[:2].copy()
        matched = 0
        for _ in range(ITERS):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            P = pq[:, :2] @ Rz.T + t
            d, j = mt.query(P, k=1)
            keep = d < ASSOC
            if keep.sum() < 3:
                break
            matched = int(keep.sum())
            R, tt = umeyama2d(pq[keep, :2], mp[j[keep]])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            t = tt
        if matched < 3:
            rows.append(dict(ok=False))
            continue
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        P = pq[:, :2] @ Rz.T + t
        d, j = mt.query(P, k=1)
        keep = d < ASSOC
        bear = np.degrees(np.arctan2(P[keep, 1] - t[1], P[keep, 0] - t[0]))
        yg = np.arctan2(gt[i][1, 0], gt[i][0, 0])
        rows.append(dict(ok=True, n_q=len(pq), n_m=int(keep.sum()),
                         e7=abs(float(wrap(np.arctan2(R7[1, 0], R7[0, 0]) - yg))),
                         ee=abs(float(wrap(yaw - yg))),
                         spread=float(np.ptp(bear)) if keep.sum() > 1 else 0.0))
    ok = [r for r in rows if r.get('ok')]
    print('estimable %d / %d' % (len(ok), len(rows)))
    if ok:
        e7 = np.array([r['e7'] for r in ok]); ee = np.array([r['ee'] for r in ok])
        print('poles median %d  matched median %d  spread median %.0f'
              % (int(np.median([r['n_q'] for r in ok])), int(np.median([r['n_m'] for r in ok])),
                 np.median([r['spread'] for r in ok])))
        print('yaw err V7 median %.2f   pole-estimate median %.2f   improved %d/%d'
              % (np.median(e7), np.median(ee), int((ee < e7).sum()), len(ok)))


if __name__ == '__main__':
    main()
