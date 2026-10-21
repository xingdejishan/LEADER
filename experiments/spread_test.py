"""A/B the point-subset choice: prefix vs even stride vs azimuth-sorted.

Same budget, same grid, same adaptive window rule; only which N points the
stages see changes. Run: python spread_test.py <prefix|stride|azimuth>
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
THRESH = 0.10
MODE = 'stride'
_MS = None


def init():
    global _MS
    _MS = F.MapSource(refs, rpos)


def take(points, n):
    if len(points) <= n:
        return points
    if MODE == 'prefix':
        return points[:n]
    if MODE == 'stride':
        return points[::max(1, len(points) // n)][:n]
    az = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(az)
    pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
    return points[np.sort(pick)]


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
    qc = take(scan, 1500) @ R0.T
    qf = take(scan, 4000) @ R0.T

    P = np.empty((len(D_HOME) * len(qc), 3))
    for a, d in enumerate(D_HOME):
        P[a * len(qc):(a + 1) * len(qc)] = qc + (o + d * lat)
    c0 = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(D_HOME), -1).mean(1)
    dh = float(D_HOME[int(np.argmin(c0))])
    DC = np.clip(dh + D3, -2, 2)
    grid = [(s, float(d)) for s in S_COARSE for d in DC]
    P = np.empty((len(grid) * len(qc), 3))
    for a, (s, d) in enumerate(grid):
        P[a * len(qc):(a + 1) * len(qc)] = qc + (o + s * u + d * lat)
    cc = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(grid), -1).mean(1)
    picks = []
    for t in np.argsort(cc):
        if all(abs(grid[t][0] - grid[p][0]) >= 1.5 for p in picks):
            picks.append(t)
        if len(picks) >= 6:
            break

    cen = np.empty(len(picks))
    for a, t in enumerate(picks):
        s0, d0 = grid[t]
        cen[a] = float(np.minimum(F.query_dist(tree, qf + (o + s0 * u + d0 * lat)), 1.0).mean())
    order = np.argsort(cen); base = cen[order[0]]
    best = (1e9, 0.0, 0.0); expanded = 0
    for a in order:
        if expanded > 0 and cen[a] - base > THRESH:
            continue
        s0, d0 = grid[picks[a]]
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -8 <= s0 + ds <= 8 and -2 <= d0 + dd <= 2]
        Q = np.empty((len(poses) * len(qf), 3))
        for k, (s, d) in enumerate(poses):
            Q[k * len(qf):(k + 1) * len(qf)] = qf + (o + s * u + d * lat)
        costs = np.minimum(F.query_dist(tree, Q), 1.0).reshape(len(poses), -1).mean(1)
        expanded += 1
        j = int(np.argmin(costs))
        if costs[j] < best[0]:
            best = (float(costs[j]), poses[j][0], poses[j][1])
    T = v7[i].copy(); T[:3, 3] = o + best[1] * u + best[2] * lat
    return float(np.linalg.norm(T[:3, 3] - gt[i][:3, 3])), expanded


if __name__ == '__main__':
    globals()['MODE'] = sys.argv[1] if len(sys.argv) > 1 else 'stride'
    dep = np.linalg.norm(v7[:, :3, 3] - gt[:, :3, 3], axis=1)
    t0 = time.perf_counter()
    with Pool(12, initializer=init) as p:
        R = p.map(run, range(len(keys)))
    e = np.array([r[0] for r in R]); ex = np.array([r[1] for r in R])
    m = ~rep
    print('MODE=%-7s %.0f s' % (MODE, time.perf_counter() - t0))
    print('  MPE=%.4f  median=%.4f  <=1m=%.0f%%  windows=%.2f'
          % (e.mean(), np.median(e), 100 * (e <= 1).mean(), ex.mean()))
    for lo, hi in [(1, 3), (3, 5), (5, 8)]:
        s = (dep >= lo) & (dep < hi) & m
        if s.sum():
            print('  dep %d-%d m: median %.2f  <=1m %.0f%%'
                  % (lo, hi, np.median(e[s]), 100 * (e[s] <= 1).mean()))
