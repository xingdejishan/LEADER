"""Strict solver comparison on identical (world, pixels, ids).

A: the deterministic fast solver shipped in match_prov.py (farthest-point minimal sets,
   8 fixed trials, one LM refinement).
B: EPnP + RANSAC 2000 + LM (common.solve_pnp).

Same correspondences, same intrinsics, same qualification thresholds, same extrinsics.
Reports qualified counts, correctness, per-call time, and whether A lost the two poses
that B found (no solution at all, or a solution rejected by the qualification).
"""
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
sys.path.insert(0, STORE)
sys.path.insert(0, '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1')
import common as C

K = C.K
T_BC = np.asarray(C.load(C.STORE / 'library.json')['T_BC'], float)


def camera_of(rv, tv):
    M = np.eye(4)
    M[:3, :3] = cv2.Rodrigues(rv)[0]
    M[:3, 3] = np.asarray(tv).ravel()
    return M


def residuals_of(cam, world, pixels):
    c = C.transform(world, cam)
    p = c @ K.T
    return np.linalg.norm(p[:, :2] / np.maximum(p[:, 2:3], 1e-12) - pixels, axis=1), c


def pnp_result(cam, residual, ids, world, pixels, body_from_camera, ix):
    if len(ix) < 8 or not np.isfinite(cam).all():
        return None
    unique = np.unique(ids[ix])
    cells = np.floor(pixels[ix] / [808 / 4, 616 / 4]).astype(int).clip(0, 3)
    cov = len(np.unique(cells[:, 1] * 4 + cells[:, 0]))
    pose = np.linalg.inv(cam) @ np.linalg.inv(body_from_camera)
    return dict(pose=pose, unique_query_inliers=int(len(unique)), coverage_cells=int(cov),
                inlier_ratio=float(len(unique) / max(1, len(np.unique(ids)))),
                qualified=bool(len(unique) >= 15 and cov >= 6
                               and len(unique) / max(1, len(np.unique(ids))) >= .2))


def fast(world, pixels, ids, trials=8):
    world = np.asarray(world, float); pixels = np.asarray(pixels, float)
    ids = np.asarray(ids, np.int32)
    if len(world) < 8:
        return None
    n = len(world)
    centre = world.mean(0)
    cur = int(np.argmax(((world - centre) ** 2).sum(1)))
    dist = np.linalg.norm(world - world[cur], axis=1)
    order = []
    for _ in range(min(n, 4 * trials)):
        i = int(np.argmax(dist)); order.append(i)
        dist = np.minimum(dist, np.linalg.norm(world - world[i], axis=1))
    if len(order) < 4:
        return None
    best = None
    for t in range(trials):
        pick = [order[(t + j * trials) % len(order)] for j in range(4)]
        if len(set(pick)) < 4:
            continue
        ok, rv, tv = cv2.solvePnP(world[pick], pixels[pick], K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        res, cam = residuals_of(camera_of(rv, tv), world, pixels)
        inl = np.flatnonzero((res <= 12) & (cam[:, 2] > .5))
        if best is None or len(inl) > len(best[0]):
            best = (inl, rv, tv)
        if len(inl) >= 0.6 * n:
            break
    if best is None or len(best[0]) < 8:
        return None
    ix = best[0]
    rv, tv = cv2.solvePnPRefineLM(world[ix], pixels[ix], K, None, best[1], best[2])
    res, cam = residuals_of(camera_of(rv, tv), world, pixels)
    ix = np.flatnonzero((res <= 12) & (cam[:, 2] > .5))
    return pnp_result(camera_of(rv, tv), res, ids, world, pixels, T_BC, ix)


def correct(P, g):
    if P is None:
        return False
    R = np.asarray(P[:3, :3], float); t = np.asarray(P[:3, 3], float)
    return (np.linalg.norm(t - g[:3, 3]) <= 1.0
            and np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ g[:3, :3]).as_rotvec())) <= 2.0)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = [str(k) for k in w['hard_keys']]
    gt = {k: np.asarray(p, float) for k, p in zip(keys, w['hard_gt'])}
    dump = np.load('/dev/shm/corr_dump.npy', allow_pickle=True).item()

    rows = []
    for idx, d in sorted(dump.items()):
        world = np.asarray(d['world'], float); pixels = np.asarray(d['pixels'], float)
        ids = np.asarray(d['ids'])
        if len(world) < 8:
            continue
        g = gt[keys[idx]]
        t0 = time.perf_counter(); A = fast(world, pixels, ids); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); B = C.solve_pnp(world, pixels, ids, T_BC); tb = time.perf_counter() - t0
        seeds = []
        for sd in (20, 7, 99):
            s = C.solve_pnp(world, pixels, ids, T_BC, seed=sd)
            seeds.append(correct(None if s is None else s['pose'], g))
        rows.append(dict(idx=idx, A=A, B=B, ta=ta, tb=tb, seeds=seeds,
                         ca=correct(None if A is None else A['pose'], g),
                         cb=correct(None if B is None else B['pose'], g)))

    n = len(rows)
    qa = sum(1 for r in rows if r['A'] and r['A']['qualified'])
    qb = sum(1 for r in rows if r['B'] and r['B']['qualified'])
    ca = sum(1 for r in rows if r['ca']); cb = sum(1 for r in rows if r['cb'])
    print('frames %d' % n)
    print('A fast   : solutions %d  qualified %d  correct %d   median %.1f ms  total %.1f ms'
          % (sum(1 for r in rows if r['A']), qa, ca,
             1000 * np.median([r['ta'] for r in rows]), 1000 * sum(r['ta'] for r in rows)))
    print('B RANSAC : solutions %d  qualified %d  correct %d   median %.1f ms  total %.1f ms'
          % (sum(1 for r in rows if r['B']), qb, cb,
             1000 * np.median([r['tb'] for r in rows]), 1000 * sum(r['tb'] for r in rows)))
    print()
    both = sum(1 for r in rows if r['ca'] and r['cb'])
    onlyB = sum(1 for r in rows if r['cb'] and not r['ca'])
    onlyA = sum(1 for r in rows if r['ca'] and not r['cb'])
    print('correct: both %d   only B %d   only A %d   neither %d' % (both, onlyB, onlyA, n - both - onlyB - onlyA))
    print()
    for r in rows:
        if not r['cb']:
            continue
        A = r['A']
        state = ('no solution' if A is None else
                 'solution but NOT qualified' if not A['qualified'] else 'qualified')
        print('frame %d: B correct; A -> %s%s' %
              (r['idx'], state,
               '' if A is None else ' (inliers %d, cells %d)' % (A['unique_query_inliers'], A['coverage_cells'])))
    stab = [sum(r['seeds']) for r in rows]
    print()
    print('B across seeds 20/7/99: correct count per frame -> %s'
          % {k: stab.count(k) for k in sorted(set(stab))})


if __name__ == '__main__':
    main()
