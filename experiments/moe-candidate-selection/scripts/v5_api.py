"""API smoke test: every documented method, called with its documented signature."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, ROOT)
from algorithms import CFS, MoE  # noqa: E402
from algorithms.moe import MapSource  # noqa: E402

refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
T0 = np.asarray(z['poses'], np.float64)[17]

cfs = CFS()
pts = cfs.decode(frames[17]['scan_path'])
scan = cfs.prepare_scan(pts)
print('decode', pts.shape, pts.dtype, '| prepare_scan', scan.shape)
print('spread_subset', cfs.spread_subset(scan, 100).shape)
print('CFS()', cfs.coarse_points, cfs.fine_points, cfs.topk, cfs.thresh, cfs.trunc,
      cfs.voxel_size, cfs.min_range, cfs.max_range)

moe = MoE(refs, positions, dict(workers=1))
u = moe.cfg and None
from algorithms.moe import along_track_direction  # noqa: E402
u = along_track_direction(T0)
mp = moe.maps.local(T0[:3, 3], moe.maps.nearby(T0[:3, 3])[:3])
cs = cfs.coarse(mp, scan, T0, u, workers=1)
print('coarse keys', sorted(cs), '| surface', cs['surface'].shape, '| picks', cs['picks'],
      '| n_evals', cs['n_evals'])
sol = cfs.solve(mp, scan, T0, u, workers=1, coarse=cs)
print('solve', {k: sol[k] for k in ('s', 'd', 'cost', 'evals', 'expanded')},
      '| surface', sol['surface'].shape)

tree = cKDTree(mp)
print('cost', round(moe.cost(tree, scan), 6))
print('ambiguity', [round(v, 6) for v in moe.ambiguity(tree, scan, T0, u)])
pose = np.array(T0)
pose[:3, 3] = T0[:3, 3] + sol['s'] * u + sol['d'] * np.cross(u, [0, 0, 1.0]) / np.linalg.norm(np.cross(u, [0, 0, 1.0]))
pose2 = moe.refine_rotation(tree, mp, scan, pose)
print('refine_rotation rot delta deg',
      round(float(np.degrees(np.arccos(np.clip((np.trace(pose2[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))), 4),
      '| translation frozen:', bool(np.array_equal(pose2[:3, 3], pose[:3, 3])))
p, dg = moe.localize(scan, T0, visual_pose=None)
print('localize diag keys', sorted(dg))
print('  ', {k: dg[k] for k in ('gain', 'ran_fine', 's', 'd', 'expanded', 'rotated',
                              'ratio', 'chosen')})
out = moe.localize_many([('a', scan, T0), ('b', scan, T0, pose2)], workers=1)
print('localize_many', [(k, len(p), d['chosen']) for k, p, d in out])
print('MapSource', moe.maps.hit, 'hits', moe.maps.miss, 'misses', len(moe.maps.nearby(T0[:3, 3])))
