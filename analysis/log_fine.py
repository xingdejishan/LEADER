"""Record the full fine-grid cost matrix per frame, plus per-pose truth error.

No pruning: every one of the six candidates gets its complete 5x9 window scored
at 4000 points, so the resulting statistics describe the search geometry itself
rather than the pruning heuristic.

Writes fine_stats.npz with arrays
    dep      (N,)     deployed translation error
    coarse   (N,6)    coarse cost of the six selected candidates
    centre   (N,6)    centre cost of those candidates at 4000 points
    cost     (N,6,45) exact cost of every pose in every window
    err      (N,6,45) truth translation error of every pose
    replaced (N,)     suspect-window mask
"""
import json
import sys
import time

import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/r5b-lossless-20261019-v1')
import fast_lib as F

H = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
OUT = '/data2/zhangcy/r5e-adaptive-20261019-v1/fine_stats.npz'

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

_MS = None


def init():
    global _MS
    _MS = F.MapSource(refs, rpos)


def run(i):
    c = v7[i][:3, 3]
    M = _MS.local(c)
    scan = F.decode(qmap[keys[i]]['scan_path'])
    scan = scan[(np.linalg.norm(scan, axis=1) > 1) & (np.linalg.norm(scan, axis=1) < 60)]
    scan = F.voxel(scan, 0.3)
    u = v7[i][:3, :3][:, 0].copy(); u /= np.linalg.norm(u)
    lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
    R0 = v7[i][:3, :3]; o = v7[i][:3, 3]
    tree = cKDTree(M)
    qc = scan[:1500]; qf = scan[:4000]
    bc = qc @ R0.T; bf = qf @ R0.T

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
    picks = []
    for t in np.argsort(cc):
        if all(abs(grid[t][0] - grid[p][0]) >= 1.5 for p in picks):
            picks.append(t)
        if len(picks) >= 6:
            break

    cost_mat = np.full((6, 45), np.nan, np.float32)
    err_mat = np.full((6, 45), np.nan, np.float32)
    cen_mat = np.full(6, np.nan, np.float32)
    for a, t in enumerate(picks):
        s0, d0 = grid[t]
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -8 <= s0 + ds <= 8 and -2 <= d0 + dd <= 2]
        Q = np.empty((len(poses) * len(qf), 3))
        for k, (s, d) in enumerate(poses):
            Q[k * len(qf):(k + 1) * len(qf)] = bf + (o + s * u + d * lat)
        costs = np.minimum(F.query_dist(tree, Q), 1.0).reshape(len(poses), -1).mean(1)
        cost_mat[a, :len(poses)] = costs
        for k, (s, d) in enumerate(poses):
            err_mat[a, k] = np.linalg.norm(o + s * u + d * lat - gt[i][:3, 3])
        for k, (s, d) in enumerate(poses):
            if abs(s - s0) < 1e-9 and abs(d - d0) < 1e-9:
                cen_mat[a] = costs[k]
                break
    return dict(dep=float(np.linalg.norm(v7[i][:3, 3] - gt[i][:3, 3])),
                coarse=cc[picks].astype(np.float32), centre=cen_mat,
                cost=cost_mat, err=err_mat, replaced=bool(rep[i]))


if __name__ == '__main__':
    t0 = time.perf_counter()
    with Pool(12, initializer=init) as p:
        R = p.map(run, range(len(keys)))
    print('done %d frames in %.0f s' % (len(R), time.perf_counter() - t0))
    np.savez_compressed(OUT,
                        dep=np.array([r['dep'] for r in R], np.float32),
                        coarse=np.stack([r['coarse'] for r in R]),
                        centre=np.stack([r['centre'] for r in R]),
                        cost=np.stack([r['cost'] for r in R]),
                        err=np.stack([r['err'] for r in R]),
                        replaced=np.array([r['replaced'] for r in R]))
    print('saved', OUT)
