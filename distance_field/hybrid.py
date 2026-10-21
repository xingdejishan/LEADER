"""Hybrid: enumerate the fine grid, rank with the GPU field, decide with exact KD.

This was the third attempt at using the field to cut exact evaluations. It lost
accuracy too (median 1.41 m at keep=3, 1.92 m at keep=8, against 0.78 m for the
full exact enumeration), because the field error (0.05-0.08 m) exceeds the cost
differences between neighbouring poses inside one fine window.
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/r5b-lossless-20261019-v1')
sys.path.insert(0, '/home/zhangcy/projects/r5c-distancefield-20261019-v1')
import fast_lib as F
import df_core as DF

H = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel = json.load(open(H + 'selection.json'))
refs, qs = sel['references'], sel['queries']
z = np.load(H + 'scored.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
v7 = z['refined_V7_poses']
gt = z['reference']
rep = z['replaced'].astype(bool)
qmap = {str(q['key']): q for q in qs}
rpos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

S_COARSE = np.arange(-8.0, 8.0001, 1.0)
D_HOME = np.arange(-1.0, 1.0001, 0.25)
D3 = np.array([-0.5, 0.0, 0.5])
S_FINE = np.arange(-2, 3) * 0.25
D_FINE = np.arange(-0.4, 0.4001, 0.1)
TOPK_CAND = 6


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def frame(i, maps, dev, keep=3):
    c = v7[i][:3, 3]
    M = maps.local(c)
    scan = F.decode(qmap[keys[i]]['scan_path'])
    scan = scan[(np.linalg.norm(scan, axis=1) > 1) & (np.linalg.norm(scan, axis=1) < 60)]
    scan = F.voxel(scan, 0.3)
    u = v7[i][:3, :3][:, 0].copy(); u /= np.linalg.norm(u)
    lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
    R0 = v7[i][:3, :3]; o = v7[i][:3, 3]
    tree = cKDTree(M)
    qc = scan[:1500]; qf = scan[:4000]
    bc = qc @ R0.T; bf = qf @ R0.T

    fld = DF.Field(M, voxel=0.4, trunc=1.0, dev=dev)

    # coarse: exact, keeps candidate selection identical to the baseline
    t0 = time.perf_counter()
    P = np.empty((len(D_HOME) * len(qc), 3))
    for a, d in enumerate(D_HOME):
        P[a * len(qc):(a + 1) * len(qc)] = bc + (o + d * lat)
    c0 = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(D_HOME), -1).mean(1)
    dh = float(D_HOME[int(np.argmin(c0))])
    DC = np.clip(dh + D3, -2, 2)
    grid = [(s, float(d)) for s in S_COARSE for d in DC]
    P = np.empty((len(grid) * len(qc), 3))
    for a, (s, d) in enumerate(grid):
        P[a * len(qc):(a + 1) * len(qc)] = bc + (o + s * u + d * lat)
    cc = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(grid), -1).mean(1)
    t_coarse = time.perf_counter() - t0
    picks = []
    for t in np.argsort(cc):
        s = grid[t][0]
        if all(abs(s - grid[p][0]) >= 1.5 for p in picks):
            picks.append(t)
        if len(picks) >= TOPK_CAND:
            break

    all_poses = []
    for t in picks:
        s0, d0 = grid[t]
        all_poses.append([(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                          if -8 <= s0 + ds <= 8 and -2 <= d0 + dd <= 2])

    # fine: field ranks the enumerated poses, exact KD decides among the top few
    t0 = time.perf_counter(); sync(dev)
    flat = np.concatenate(all_poses, axis=0)
    P = np.empty((len(flat) * len(qf), 3))
    for a, (s, d) in enumerate(flat):
        P[a * len(qf):(a + 1) * len(qf)] = bf + (o + s * u + d * lat)
    val, _ = fld.sample(torch.from_numpy(P).to(dev).float())
    field_cost = val.view(len(flat), len(qf)).mean(1).cpu().numpy()
    sync(dev); t_gpu = time.perf_counter() - t0

    shortlist = []
    off = 0
    for part in all_poses:
        n = len(part)
        order = np.argsort(field_cost[off:off + n])[:keep]
        shortlist.extend(off + int(x) for x in order)
        off += n
    shortlist = np.asarray(shortlist)

    t0 = time.perf_counter()
    best = (1e9, 0.0, 0.0)
    for a in shortlist:
        s, d = flat[a]
        cst = float(np.minimum(F.query_dist(tree, bf + (o + s * u + d * lat)), 1.0).mean())
        if cst < best[0]:
            best = (cst, s, d)
    t_exact = time.perf_counter() - t0

    s, d = best[1], best[2]
    T = v7[i].copy(); T[:3, 3] = o + s * u + d * lat
    return (float(np.linalg.norm(T[:3, 3] - gt[i][:3, 3])), t_coarse, t_gpu, t_exact,
            len(shortlist), len(flat))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dep = np.linalg.norm(v7[:, :3, 3] - gt[:, :3, 3], axis=1)
    frames = []
    for lo, hi in [(0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8)]:
        idx = np.where((dep >= lo) & (dep < hi) & (~rep))[0]
        if len(idx):
            frames.extend(int(x) for x in idx[np.linspace(0, len(idx) - 1,
                                                          min(max(n // 5, 1), len(idx)), dtype=int)])
    dev = DF.pick_device()
    maps = F.MapSource(refs, rpos)
    E, tc, tg, te = [], [], [], []
    for k, i in enumerate(frames):
        e, a, b, c_, ns, nf = frame(i, maps, dev, keep=keep)
        E.append(e); tc.append(a); tg.append(b); te.append(c_)
        print(' %2d/%d frame %4d dep=%5.2f err=%.2f  coarse %.0f | field %.1f | '
              'exact(%.0f/%.0f) %.0f ms'
              % (k + 1, len(frames), i, dep[i], e, a * 1e3, b * 1e3, ns, nf, c_ * 1e3),
              flush=True)
    E = np.array(E)
    print('\nkeep=%d  frames=%d  med=%.2f  <=1m=%.0f%%'
          % (keep, len(E), np.median(E), 100 * (E <= 1).mean()))
    print('  coarse %.0f ms | field %.1f ms | exact %.0f ms | total %.0f ms'
          % (np.median(tc) * 1e3, np.median(tg) * 1e3, np.median(te) * 1e3,
             (np.median(tc) + np.median(tg) + np.median(te)) * 1e3))
