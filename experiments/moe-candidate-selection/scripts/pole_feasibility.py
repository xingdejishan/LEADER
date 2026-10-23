"""Pole-landmark feasibility: can single-frame pole fitting beat LEADER's yaw?

Per frame:
  * pole map  : poles extracted from the nearby REFERENCE scans only, placed in world
                with the reference poses (no query truth anywhere)
  * query     : poles extracted from the single query scan, in body coordinates
  * estimate  : 2D rigid (yaw + translation) by iterated closest-point + Umeyama,
                initialised at the deployed V7 pose; no truth used for association
Reports coverage, landmark angular spread, yaw error versus V7, and the correlation
between the two errors.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import fast_lib as F

VOX = 0.30
MIN_HEIGHT = 1.2
MAX_WIDTH = 1.0
MIN_POINTS = 25
RANGE = 35.0
ASSOC = 2.5
N_REF = 6
ITERS = 6


def poles_of(points):
    """pole-like vertical clusters: tall, thin, enough points"""
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
        out.append([xy[:, 0].mean(), xy[:, 1].mean(), zr])
    return np.asarray(out) if out else np.zeros((0, 3))


def umeyama2d(P, Q):
    """rotation+translation taking P onto Q (2D)"""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return R, t


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

    rng = np.random.default_rng(53)
    # stratified: 15 frames where V7 is already good, 15 where it is bad
    err = np.degrees([np.linalg.norm(Rotation.from_matrix(
        gt[i][:3, :3].T @ v7[i][:3, :3]).as_rotvec()) for i in range(len(keys))])
    good = np.flatnonzero(err < 1.0)
    bad = np.flatnonzero(err > 3.0)
    sample = sorted(list(rng.choice(good, 15, replace=False)) + list(rng.choice(bad, 15, replace=False)))

    rows = []
    for i in sample:
        date, stamp = keys[i].split('/')
        qpath = os.path.join(os.path.dirname(ref_paths[0].rstrip('/')),
                             '..', '..', date, 'velodyne_sync', stamp + '.bin')
        qpath = os.path.normpath(qpath)
        if not os.path.isfile(qpath):
            continue
        q = F.decode(qpath)
        q = q[np.linalg.norm(q[:, :2], axis=1) <= RANGE]
        pq = poles_of(q)
        if len(pq) < 3:
            rows.append(dict(i=int(i), n_q=len(pq), ok=False))
            continue

        t7, R7 = v7[i][:3, 3], v7[i][:3, :3]
        _, nb = rtree.query(t7, k=N_REF)
        mworld = []
        for j in np.atleast_1d(nb):
            Tj = ref_poses[int(j)]
            pr = poles_of(F.decode(ref_paths[int(j)]))
            if len(pr) == 0:
                continue
            pw = pr[:, :2] @ Tj[:3, :3][:2, :2].T + Tj[:2, 3]
            mworld.append(pw)
        if not mworld:
            rows.append(dict(i=int(i), n_q=len(pq), ok=False))
            continue
        mp = np.vstack(mworld)
        mp = mp[np.linalg.norm(mp - t7[:2], axis=1) <= RANGE]
        if len(mp) < 3:
            rows.append(dict(i=int(i), n_q=len(pq), ok=False))
            continue
        mt = cKDTree(mp)

        # iterative closest point in 2D, starting at V7 (no truth)
        yaw0 = np.arctan2(R7[1, 0], R7[0, 0])
        yaw = yaw0
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
            rows.append(dict(i=int(i), n_q=len(pq), n_m=matched, ok=False))
            continue

        # angular spread of the matched landmarks as seen from the estimate
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        P = (pq[:, :2] @ Rz.T) + t
        d, j = mt.query(P, k=1)
        keep = d < ASSOC
        bear = np.degrees(np.arctan2(P[keep, 1] - t[1], P[keep, 0] - t[0]))
        spread = float(np.ptp(bear)) if keep.sum() > 1 else 0.0

        yaw_gt = np.arctan2(gt[i][1, 0], gt[i][0, 0])
        rows.append(dict(i=int(i), n_q=len(pq), n_m=matched, ok=True,
                         yaw=v7[i][:3, :3],
                         err_v7=float(abs(np.degrees(np.arctan2(np.sin(yaw0 - yaw_gt), np.cos(yaw0 - yaw_gt))))),
                         err_est=float(abs(np.degrees(np.arctan2(np.sin(yaw - yaw_gt), np.cos(yaw - yaw_gt))))),
                         spread=spread, dt=float(np.linalg.norm(t - gt[i][:2, 3]))))

    okr = [r for r in rows if r.get('ok')]
    print('frames attempted %d   with >=3 query poles and a match: %d (%.0f%%)'
          % (len(rows), len(okr), 100.0 * len(okr) / max(1, len(rows))))
    if okr:
        e7 = np.array([r['err_v7'] for r in okr])
        ee = np.array([r['err_est'] for r in okr])
        sp = np.array([r['spread'] for r in okr])
        nm = np.array([r['n_m'] for r in okr])
        print('matched poles: median %d' % int(np.median(nm)))
        print('bearings spread: median %.0f deg' % np.median(sp))
        print('yaw error  V7  median %.2f  mean %.2f' % (np.median(e7), e7.mean()))
        print('yaw error  est median %.2f  mean %.2f' % (np.median(ee), ee.mean()))
        print('improved on %d of %d frames; correlation of the two errors r = %+.3f'
              % (int((ee < e7).sum()), len(okr), np.corrcoef(e7, ee)[0, 1]))


if __name__ == '__main__':
    main()
