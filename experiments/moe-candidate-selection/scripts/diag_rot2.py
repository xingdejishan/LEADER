"""Is the residual 0.12 deg just an iteration-schedule difference? Probe one frame."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(P, 'stages/fast_search'))
sys.path.insert(0, os.path.join(P, 'stages'))
import fast_lib as F
import pipeline as Pn

z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
prod = np.asarray(z['poses'], np.float64)
refs = json.load(open(P + '/inputs/refs.json'))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
maps = F.MapSource(refs, pos, radius=60.0)
cfg = Pn.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=1)


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def run(tree, geom, q, T, eps=1e-6, iters=6, tie=True, cap=0.5):
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        d = tree.query(val @ M[:3, :3].T + M[:3, 3], k=1)[0]
        return float(np.minimum(d, 1.2).mean()), int((d < 0.6).sum())

    t = np.array(T, float)
    centre = t[:3, 3].copy()
    value, hit = obj(t)
    for it in range(iters):
        W = train @ t[:3, :3].T + centre
        dist, ix = tree.query(W, k=1)
        sel = dist < 1.2
        if sel.sum() < 30:
            break
        A, B = W[sel] - centre, geom[ix[sel]] - centre
        a, b = A.mean(0), B.mean(0)
        U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
        if s[1] < 1e-8:
            break
        corr = np.eye(3)
        corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
        vec = Rotation.from_matrix(Vt.T @ corr @ U.T).as_rotvec()
        scale = min(1.0, np.deg2rad(cap) / max(np.linalg.norm(vec), 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre
        nv, nh = obj(cand)
        ok = (nv < value - eps) and (nh >= hit if tie else True)
        if not ok:
            break
        t, value, hit = cand, nv, nh
    return t, it + 1


for i in (98779, 26327, 41338):
    k = keys[i]
    date, ts = k.split('/')
    rec = {'key': k, 'scan_path': '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin' % (date, ts)}
    T0 = t0[i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
    tree = cKDTree(mp)
    scan = Pn.prepare_scan(rec, cfg)
    u = Pn.along_track_direction(T0)
    cs = F.coarse_stage(mp, scan, T0, u, workers=1, n_coarse=1000, topk=6)
    pose, _ = Pn.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
    q50 = scan[np.linalg.norm(scan, axis=1) <= 50]
    _, ids = np.unique(np.floor(q50 / 0.3).astype(np.int64), axis=0, return_index=True)
    q = q50[np.sort(ids)]
    print('\nframe', i, k)
    for iters in (6, 8, 10, 12, 16, 24):
        T, used = run(tree, mp, q, pose, iters=iters)
        print('  iters cap %2d (used %2d): %.4f deg' % (iters, used, rot_deg(T[:3, :3], prod[i][:3, :3])))
    T, used = run(tree, mp, q, pose, tie=False, iters=24)
    print('  no tie-break (used %2d): %.4f deg' % (used, rot_deg(T[:3, :3], prod[i][:3, :3])))
    T, used = run(tree, mp, q, pose, cap=0.25, iters=24)
    print('  cap 0.25 (used %2d): %.4f deg' % (used, rot_deg(T[:3, :3], prod[i][:3, :3])))
    T, used = run(tree, mp, q, pose, eps=1e-4, iters=24)
    print('  eps 1e-4 (used %2d): %.4f deg' % (used, rot_deg(T[:3, :3], prod[i][:3, :3])))
    T, used = run(tree, mp, q, T0, iters=24)
    print('  from t0  (used %2d): %.4f deg' % (used, rot_deg(T[:3, :3], prod[i][:3, :3])))
