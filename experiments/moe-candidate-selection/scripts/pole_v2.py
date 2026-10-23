"""Pole feasibility, v2: ground segmentation before clustering.

The v1 detector produced one 85k-point cluster that swallowed the road and every pole
touching it, so coverage collapsed to 3%.  This version estimates a local ground height
per 1 m cell, removes everything within 0.25 m of it, and only then clusters.

Estimation is unchanged: 2D rigid (yaw + translation) by iterated closest point started
at the deployed pose; no query truth is used for association.
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

VOX = 0.25
MIN_HEIGHT = 1.2
MAX_WIDTH = 0.9
MIN_POINTS = 20
RANGE = 35.0
ASSOC = 2.5
N_REF = 6
ITERS = 6
GROUND_BAND = 0.25


def strip_ground(points):
    xy = points[:, :2]
    cell = np.floor(xy).astype(np.int32)
    key = cell[:, 0].astype(np.int64) * 100000 + cell[:, 1]
    order = np.argsort(key)
    ks = key[order]
    zs = points[order, 2]
    uniq, start = np.unique(ks, return_index=True)
    bounds = np.append(start, len(ks))
    ground = np.empty(len(uniq))
    for k in range(len(uniq)):
        seg = zs[bounds[k]:bounds[k + 1]]
        ground[k] = np.percentile(seg, 10) if len(seg) >= 4 else seg.min()
    lut = dict(zip(uniq.tolist(), ground.tolist()))
    g = np.array([lut[k] for k in key])
    keep = (points[:, 2] - g) > GROUND_BAND
    return points[keep]


def poles_of(points):
    if len(points) < 100:
        return np.zeros((0, 3))
    origin = points.min(0)
    idx = np.floor((points - origin) / VOX).astype(np.int32)
    grid = np.zeros(idx.max(0) + 1, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return np.zeros((0, 3))
    pt_lab = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    order = np.argsort(pt_lab)
    lab_sorted = pt_lab[order]
    pts_sorted = points[order]
    bounds = np.searchsorted(lab_sorted, np.arange(1, n + 2))
    out = []
    for k in range(n):
        a, b = bounds[k], bounds[k + 1]
        if b - a < MIN_POINTS:
            continue
        seg = pts_sorted[a:b]
        zr = seg[:, 2].max() - seg[:, 2].min()
        xy = seg[:, :2]
        w = np.linalg.norm(xy.max(0) - xy.min(0))
        if zr < MIN_HEIGHT or w > MAX_WIDTH:
            continue
        out.append([xy[:, 0].mean(), xy[:, 1].mean(), zr, b - a])
    return np.asarray(out) if out else np.zeros((0, 4))


def umeyama2d(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


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

    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    rpos = np.asarray([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs])
    rtree = cKDTree(rpos)
    ref_paths = [r['scan_path'] for r in refs]
    ref_poses = [np.asarray(r['world_from_body'], float) for r in refs]

    from scipy.spatial.transform import Rotation
    err = np.degrees([np.linalg.norm(Rotation.from_matrix(
        gt[i][:3, :3].T @ v7[i][:3, :3]).as_rotvec()) for i in range(len(keys))])
    rng = np.random.default_rng(53)
    sample = sorted(list(rng.choice(np.flatnonzero(err < 1.0), 15, replace=False)) +
                    list(rng.choice(np.flatnonzero(err > 3.0), 15, replace=False)))

    cache = {}
    rows = []
    for i in sample:
        date, stamp = keys[i].split('/')
        qpath = os.path.join('/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT',
                             date, 'velodyne_sync', stamp + '.bin')
        if not os.path.isfile(qpath):
            continue
        q = F.decode(qpath)
        q = q[np.linalg.norm(q[:, :2], axis=1) <= RANGE]
        pq = poles_of(strip_ground(q))
        if len(pq) < 3:
            rows.append(dict(ok=False, n_q=len(pq)))
            continue

        t7, R7 = v7[i][:3, 3], v7[i][:3, :3]
        _, nb = rtree.query(t7, k=N_REF)
        mworld = []
        for j in np.atleast_1d(nb):
            j = int(j)
            if j not in cache:
                pr = poles_of(strip_ground(F.decode(ref_paths[j])))
                Tj = ref_poses[j]
                cache[j] = pr[:, :2] @ Tj[:3, :3][:2, :2].T + Tj[:2, 3] if len(pr) else np.zeros((0, 2))
            mworld.append(cache[j])
        mp = np.vstack([m for m in mworld if len(m)]) if any(len(m) for m in mworld) else np.zeros((0, 2))
        if len(mp) == 0:
            rows.append(dict(ok=False, n_q=len(pq)))
            continue
        mp = mp[np.linalg.norm(mp - t7[:2], axis=1) <= RANGE]
        if len(mp) < 3:
            rows.append(dict(ok=False, n_q=len(pq)))
            continue
        mt = cKDTree(mp)

        yaw = np.arctan2(R7[1, 0], R7[0, 0])
        t = t7[:2].copy()
        matched = 0
        for _ in range(ITERS):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            P = (pq[:, :2] @ Rz.T) + t
            d, j = mt.query(P, k=1)
            keep = d < ASSOC
            if keep.sum() < 3:
                break
            matched = int(keep.sum())
            R, tt = umeyama2d(pq[keep, :2], mp[j[keep]])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            t = tt
        if matched < 3:
            rows.append(dict(ok=False, n_q=len(pq), n_m=matched))
            continue
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        P = (pq[:, :2] @ Rz.T) + t
        d, j = mt.query(P, k=1)
        keep = d < ASSOC
        bear = np.degrees(np.arctan2(P[keep, 1] - t[1], P[keep, 0] - t[0]))
        yaw_gt = np.arctan2(gt[i][1, 0], gt[i][0, 0])
        wrap = lambda x: np.degrees(np.arctan2(np.sin(x), np.cos(x)))
        rows.append(dict(ok=True, n_q=len(pq), n_m=int(keep.sum()),
                         err_v7=abs(float(wrap(np.arctan2(R7[1, 0], R7[0, 0]) - yaw_gt))),
                         err_est=abs(float(wrap(yaw - yaw_gt))),
                         spread=float(np.ptp(bear)) if keep.sum() > 1 else 0.0))

    okr = [r for r in rows if r.get('ok')]
    print('frames attempted %d   estimable: %d (%.0f%%)' % (len(rows), len(okr), 100.0 * len(okr) / max(1, len(rows))))
    if okr:
        e7 = np.array([r['err_v7'] for r in okr]); ee = np.array([r['err_est'] for r in okr])
        print('query poles median %d   matched median %d   bearing spread median %.0f deg'
              % (int(np.median([r['n_q'] for r in okr])), int(np.median([r['n_m'] for r in okr])),
                 np.median([r['spread'] for r in okr])))
        print('yaw error  V7  median %.2f  mean %.2f' % (np.median(e7), e7.mean()))
        print('yaw error  est median %.2f  mean %.2f' % (np.median(ee), ee.mean()))
        print('improved on %d/%d   correlation r = %+.3f'
              % (int((ee < e7).sum()), len(okr),
                 np.corrcoef(e7, ee)[0, 1] if len(okr) > 2 else float('nan')))


if __name__ == '__main__':
    main()
