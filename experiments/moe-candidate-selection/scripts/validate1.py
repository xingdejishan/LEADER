"""Fidelity check while the sweep runs: my P_rot[eps=1e-6] vs the online run's poses."""
import json
import os
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
SH = '/home/zhangcy/.dsh_scratch/sweep_full'

z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
prod = np.asarray(z['poses'], np.float64)

# coverage guard population: refs within 30 m
refs = json.load(open(P + '/inputs/refs.json'))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
from scipy.spatial import cKDTree
tree = cKDTree(pos)
cnt = np.array([len(tree.query_ball_point(t0[i][:3, 3], 30.0)) for i in range(len(keys))])
print('frames with <1 ref within 30 m: %d   <2: %d   <3: %d'
      % (int((cnt < 1).sum()), int((cnt < 2).sum()), int((cnt < 3).sum())))

same = np.abs(prod[:, :3, 3] - t0[:, :3, 3]).max(1) < 1e-12
print('frames where the online run kept t0 exactly: %d' % int(same.sum()))

shards = sorted(f for f in os.listdir(SH) if f.endswith('.npz'))
if not shards:
    sys.exit(0)
idx, mine = [], []
for name in shards:
    s = np.load(os.path.join(SH, name))
    idx.append(s['idx'])
    mine.append(s['rot'][:, 0])
idx = np.concatenate(idx)
mine = np.concatenate(mine)
good = np.isfinite(mine[:, 0, 0])
idx, mine = idx[good], mine[good]
d = np.linalg.norm(mine[:, :3, 3] - prod[idx][:, :3, 3], axis=1)
dr = np.degrees(np.arccos(np.clip((np.trace(
    np.swapaxes(mine[:, :3, :3], -1, -2) @ prod[idx][:, :3, :3], axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
print('compared %d frames (%d shards): |dt| mean %.2e max %.2e  >1mm %d | rot mean %.2e deg max %.2e'
      % (len(idx), len(shards), d.mean(), d.max(), int((d > 1e-3).sum()), dr.mean(), dr.max()))
if (d > 1e-6).any():
    k = int(np.argmax(d))
    print('worst frame', idx[k], keys[idx[k]], d[k], dr[k])
    print(' mine', mine[k][:3, 3], '\n prod', prod[idx[k]][:3, 3])
