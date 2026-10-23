"""Extra: the kept `prune=True` path and the adaptive path must equal the old code.

Compares, on real frames, CFS.solve(prune=True/False) against stages/fast_search
fast_lib.solve(prune=True) and adaptive.solve_adaptive.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(ROOT, 'stages'))
from algorithms import CFS, MoE  # noqa: E402
import fast_lib as F  # noqa: E402
import adaptive  # noqa: E402
import pipeline as P  # noqa: E402

refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
poses = np.asarray(z['poses'], np.float64)

old_cfg = P.Config(workers=1)
moe = MoE(refs, positions, dict(workers=1))
maps = F.MapSource(refs, positions, radius=60.0)
cfs = CFS(coarse_points=1000, fine_points=1500)
cfs_prune = CFS(coarse_points=1000, fine_points=1500, prune=True)

print('idx  adaptive(old)            adaptive(new)            prune(old)  prune(new)')
for i in (0, 1, 4, 17, 200, 719):
    T0 = poses[i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
    scan = P.prepare_scan(frames[i], old_cfg)
    u = P.along_track_direction(T0)
    a = adaptive.solve_adaptive(mp, scan, T0, u, n_coarse=1000, n_fine=1500, workers=1)
    b = cfs.solve(mp, scan, T0, u, workers=1)
    c = F.solve(mp, scan, T0, u, workers=1, n_coarse=1000, n_fine=1500,
                topk=6, prune=True)
    d = cfs_prune.solve(mp, scan, T0, u, workers=1)
    print('%4d  s=%.3f d=%.2f c=%.6f  s=%.3f d=%.2f c=%.6f  %s  %s  evals %d/%d -> %s'
          % (i, a[0], a[1], a[2], b['s'], b['d'], b['cost'],
             '=' if (a[0], a[1], a[2]) == (c[0], c[1], c[2]) else 'DIFF',
             '=' if (c[0], c[1], c[2]) == (d['s'], d['d'], d['cost']) else 'DIFF',
             c[3], d['evals'],
             'expanded=%s' % (b['expanded'],)))
