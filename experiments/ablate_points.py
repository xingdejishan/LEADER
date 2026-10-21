"""Point-budget and solver ablation for the coarse-to-fine search module.

This is the harness behind docs/SPEED.md.

    python ablate_points.py acc <solver> <subset> <n_fine> [n_coarse] [n_frames]
    python ablate_points.py lat <solver> <subset> <n_fine> [n_coarse] [n_frames]

solver : adaptive | prune | full
         adaptive -> adaptive.solve_adaptive      (about 1.4 windows, exact scoring)
         prune    -> fast_lib.solve(prune=True)   (the gradient bound; unsound, see
                                                   docs/ADAPTIVE.md)
         full     -> fast_lib.solve(prune=False)  (all six windows)
subset : stride | linspace
         stride   -> points[::max(1, len // n)][:n]   (what 96e140d ships)
         linspace -> evenly spaced from first to last point

`acc` runs the whole frame set over a process pool and appends one row per run to
speed_results.jsonl; `lat` runs single-process with warm caches and reports the
per-frame breakdown the docs quote in milliseconds.

The subset rule is swapped by monkeypatching fast_lib.spread_subset, so the module
under test is not modified. The module location defaults to the deployed copy and
can be pointed elsewhere with CFS_DEPLOY.
"""
import json
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.environ.get('CFS_DEPLOY',
                        '/home/zhangcy/projects/coarse-fine-search-20261020-v1')
sys.path.insert(0, os.path.join(DEPLOY, 'src'))
import fast_lib as F          # noqa: E402
import adaptive as A          # noqa: E402

H = os.environ.get('CFS_FRAMES', '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/')
OUT = os.path.join(HERE, 'speed_results.jsonl')

_MODE = 'linspace'
_SOLVER = 'adaptive'
_NF = 4000
_NC = 1500
_MS = None
_V7 = None
_GT = None
_KEYS = None
_QM = None


def subset(points, n):
    if len(points) <= n:
        return points
    if _MODE == 'stride':
        return points[::max(1, len(points) // n)][:n]
    idx = np.linspace(0, len(points) - 1, num=n)
    return points[idx.astype(np.int64)]


def solve(M, scan, pose, u, nf, nc, solver):
    """Return (s, d, windows_expanded_or_nan, n_evals)."""
    if solver == 'adaptive':
        s, d, cost, ev, ex = A.solve_adaptive(
            M, scan, np.asarray(pose), u, n_coarse=nc, n_fine=nf)
        return s, d, ex, ev
    s, d, cost, ev = F.solve(M, scan, np.asarray(pose), u,
                             n_coarse=nc, n_fine=nf, prune=(solver == 'prune'))
    return s, d, float('nan'), ev


def load():
    sel = json.load(open(H + 'selection.json'))
    refs, qs = sel['references'], sel['queries']
    z = np.load(H + 'scored.npz', allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    return refs, qs, keys, z['refined_V7_poses'], z['reference'], z['replaced'].astype(bool)


def prepare(qmap, key):
    scan = F.decode(qmap[key]['scan_path'])
    r = np.linalg.norm(scan, axis=1)
    return F.voxel(scan[(r > 1.0) & (r < 60.0)], 0.3)


def _init(refs, v7, gt, keys, qmap, mode, solver, nf, nc):
    global _MS, _V7, _GT, _KEYS, _QM, _MODE, _SOLVER, _NF, _NC
    _MODE, _SOLVER, _NF, _NC = mode, solver, nf, nc
    F.spread_subset = subset
    rpos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
    _MS = F.MapSource(refs, rpos)
    _V7, _GT, _KEYS, _QM = v7, gt, keys, qmap


def _run(i):
    c = _V7[i][:3, 3]
    M = _MS.local(c)
    scan = prepare(_QM, _KEYS[i])
    u = _V7[i][:3, :3][:, 0].copy()
    u /= np.linalg.norm(u)
    t0 = time.perf_counter()
    s, d, ex, ev = solve(M, scan, _V7[i], u, _NF, _NC, _SOLVER)
    ms = (time.perf_counter() - t0) * 1e3
    lat = np.cross(u, [0, 0, 1.0])
    lat /= np.linalg.norm(lat)
    T = _V7[i].copy()
    T[:3, 3] = c + s * u + d * lat
    return float(np.linalg.norm(T[:3, 3] - _GT[i][:3, 3])), ex, ev, ms, len(scan)


def acc(solver, mode, nf, nc, nframes):
    refs, qs, keys, v7, gt, rep = load()
    qmap = {str(q['key']): q for q in qs}
    idx = list(range(len(keys)))[:nframes]
    with Pool(10, initializer=_init,
              initargs=(refs, v7, gt, keys, qmap, mode, solver, nf, nc)) as pool:
        out = pool.map(_run, idx)
    e = np.array([o[0] for o in out])
    ev = np.array([o[2] for o in out])
    npts = np.array([o[4] for o in out])
    dep = np.linalg.norm(v7[idx][:, :3, 3] - gt[idx][:, :3, 3], axis=1)
    m = ~rep[idx]
    row = {'kind': 'acc', 'solver': solver, 'subset': mode, 'n_fine': nf, 'n_coarse': nc,
           'frames': len(idx), 'mpe': float(e.mean()), 'median': float(np.median(e)),
           'le1': float((e <= 1).mean()),
           'evals': float(ev.mean()),
           'stride_one_frames': int((npts // nf == 1).sum())}
    for lo, hi in ((1, 3), (3, 5), (5, 8)):
        s_ = (dep >= lo) & (dep < hi) & m
        if s_.sum():
            row['dep_%d_%d_le1' % (lo, hi)] = float((e[s_] <= 1).mean())
    print('acc %-8s %-8s nf=%-5d nc=%-5d  MPE=%.4f  med=%.4f  <=1m=%.1f%%  evals=%.0f  '
          '5-8m=%.0f%%' % (solver, mode, nf, nc, row['mpe'], row['median'],
                           100 * row['le1'], row['evals'],
                           100 * row.get('dep_5_8_le1', float('nan'))))
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(row) + '\n')


def lat(solver, mode, nf, nc, nframes):
    global _MODE, _SOLVER
    _MODE, _SOLVER = mode, solver
    F.spread_subset = subset
    refs, qs, keys, v7, gt, rep = load()
    qmap = {str(q['key']): q for q in qs}
    rpos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
    maps = F.MapSource(refs, rpos)

    def one(i):
        c = v7[i][:3, 3]
        t0 = time.perf_counter()
        M = maps.local(c)
        t1 = time.perf_counter()
        scan = prepare(qmap, keys[i])
        t2 = time.perf_counter()
        u = v7[i][:3, :3][:, 0].copy()
        u /= np.linalg.norm(u)
        s, d, ex, ev = solve(M, scan, v7[i], u, nf, nc, solver)
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, ev

    for i in range(min(60, len(keys))):
        one(i)                                   # warm the map and reference caches
    a = np.array([one(i) for i in range(min(nframes, len(keys)))])
    total = a[:, 0] + a[:, 1] + a[:, 2]
    row = {'kind': 'lat', 'solver': solver, 'subset': mode, 'n_fine': nf, 'n_coarse': nc,
           'frames': len(a), 'map_ms': float(a[:, 0].mean()),
           'prep_ms': float(a[:, 1].mean()), 'solve_ms': float(a[:, 2].mean()),
           'total_ms': float(total.mean()), 'total_p50': float(np.median(total)),
           'total_p90': float(np.percentile(total, 90)), 'evals': float(a[:, 3].mean())}
    print('lat %-8s %-8s nf=%-5d nc=%-5d  total=%.1f ms (p50 %.1f, p90 %.1f) = '
          'map %.1f + prep %.1f + solve %.1f'
          % (solver, mode, nf, nc, row['total_ms'], row['total_p50'], row['total_p90'],
             row['map_ms'], row['prep_ms'], row['solve_ms']))
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    kind, solver, mode, nf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nc = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
    nframes = int(sys.argv[6]) if len(sys.argv) > 6 else (200 if kind == 'lat' else 1502)
    (acc if kind == 'acc' else lat)(solver, mode, nf, nc, nframes)
