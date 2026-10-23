"""Why does my rot_only differ from the online run by ~0.02 deg? Try variants."""
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

NEW = '/data2/dataset/NCLT'
z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
prod = np.asarray(z['poses'], np.float64)

SH = '/home/zhangcy/.dsh_scratch/sweep_full'
idx, mine = [], []
for name in sorted(f for f in os.listdir(SH) if f.endswith('.npz')):
    s = np.load(os.path.join(SH, name))
    idx.append(s['idx'])
    mine.append(s['rot'][:, 0])
idx = np.concatenate(idx)
mine = np.concatenate(mine)
ok = np.isfinite(mine[:, 0, 0])
idx, mine = idx[ok], mine[ok]


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


dr = rot_deg(mine[:, :3, :3], prod[idx][:, :3, :3])
print('diff deg: mean %.4f  median %.4f  p90 %.4f  >0.01 %d / %d'
      % (dr.mean(), np.median(dr), np.percentile(dr, 90), int((dr > 0.01).sum()), len(dr)))
worst = idx[np.argsort(-dr)[:4]]
print('worst frames', list(worst), [keys[i] for i in worst])

refs = json.load(open(P + '/inputs/refs.json'))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
maps = F.MapSource(refs, pos, radius=60.0)
cfg = Pn.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=1)


def vary(tree, geom, q, T, trunc, hitfrac, cap, nsig=None):
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        d = tree.query(val @ M[:3, :3].T + M[:3, 3], k=1)[0]
        return float(np.minimum(d, trunc).mean()), int((d < hitfrac).sum())

    t = np.array(T, float)
    centre = t[:3, 3].copy()
    value, hit = obj(t)
    for _ in range(6):
        W = train @ t[:3, :3].T + centre
        dist, ix = tree.query(W, k=1)
        sel = dist < trunc
        if sel.sum() < 30:
            break
        A, B = W[sel] - centre, geom[ix[sel]] - centre
        a, b = A.mean(0), B.mean(0)
        U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
        if s[1] < 1e-8:
            break
        corr = np.eye(3)
        corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ corr @ U.T
        vec = Rotation.from_matrix(R).as_rotvec()
        scale = min(1.0, np.deg2rad(cap) / max(np.linalg.norm(vec), 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre
        nv, nh = obj(cand)
        if not (nv < value - 1e-6 and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return t


for i in worst:
    k = keys[i]
    date, ts = k.split('/')
    rec = {'key': k, 'scan_path': '%s/%s/velodyne_sync/%s.bin' % (NEW, date, ts)}
    T0 = t0[i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
    tree = cKDTree(mp)
    scan = Pn.prepare_scan(rec, cfg)
    u = Pn.along_track_direction(T0)
    cs = F.coarse_stage(mp, scan, T0, u, workers=1, n_coarse=1000, topk=6)
    pose, _ = Pn.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
    pose = np.asarray(pose, np.float64)
    q50 = scan[np.linalg.norm(scan, axis=1) <= 50]
    cells = np.floor(q50 / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    qq = q50[np.sort(ids)]
    print('\nframe', i, k, 'scan', len(scan), 'q', len(qq), 'map', len(mp),
          'prod rot err vs mine already known')
    for name, kw in (('A trunc1.2 cap.5', dict(trunc=1.2, hitfrac=0.6, cap=0.5)),
                     ('B trunc1.0 cap.5', dict(trunc=1.0, hitfrac=0.6, cap=0.5)),
                     ('C trunc1.2 cap1', dict(trunc=1.2, hitfrac=0.6, cap=1.0)),
                     ('D trunc1.0 hit1.0', dict(trunc=1.0, hitfrac=1.0, cap=0.5)),
                     ('E trunc1.0 hit.5', dict(trunc=1.0, hitfrac=0.5, cap=0.5))):
        T = vary(tree, mp, qq, pose, **kw)
        print('   %-18s rot vs prod %.4f deg   pos delta %.3e'
              % (name, rot_deg(T[:3, :3], prod[i][:3, :3]), np.linalg.norm(T[:3, 3] - prod[i][:3, 3])))
    print('   %-18s rot vs prod %.4f deg' % ('A on all q (no 50m)',
          rot_deg(vary(tree, mp, scan[np.sort(np.unique(np.floor(scan / 0.3).astype(np.int64), axis=0, return_index=True)[1])],
                       pose, 1.2, 0.6, 0.5)[:3, :3], prod[i][:3, :3])))
