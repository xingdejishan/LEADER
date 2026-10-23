"""V2: bit-exact single frame against results/three_route_merged.npz (5 keys)."""
import json
import os
import sys

import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, ROOT)
from algorithms import CFS, MoE  # noqa: E402

refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
keys = [str(k) for k in z['keys']]
poses = np.asarray(z['poses'], np.float64)

ref = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
rkeys = [str(k) for k in ref['keys']]
rposes = np.asarray(ref['poses'], np.float64)
look = {k: i for i, k in enumerate(rkeys)}

moe = MoE(refs, positions, dict(workers=1, gate2=-1.0, delta=0.05))
print('MoE cfg', moe.cfg)

worst = 0.0
for i in range(5):
    scan = moe.prepare_scan(frames[i])
    pose, diag = moe.localize(scan, poses[i], visual_pose=None)
    j = look[keys[i]]
    diff = float(np.abs(pose - rposes[j]).max())
    worst = max(worst, diff)
    print('%s  scan=%d  diff=%.3e  s=%.3f d=%.3f expanded=%s rotated=%s gain=%.6f ratio=%.6f'
          % (keys[i], len(scan), diff, diag['s'], diag['d'], diag['expanded'],
             diag['rotated'], diag['gain'], diag['ratio']))
print('V2 max abs diff over 5 keys = %.3e' % worst)
print('V2 ' + ('PASS' if worst == 0.0 else 'FAIL'))

# cross-check the diagnostics of the same five frames against the reference table
rows = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
for i in range(5):
    j = look[keys[i]]
    print('  ref row %d: gain=%.6f ratio=%.6f rotated=%s'
          % (i, rows[j]['gain'], rows[j]['ratio'], rows[j]['rotated']))
