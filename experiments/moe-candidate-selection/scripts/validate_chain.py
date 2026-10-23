"""Validate the shared-prefix / upper-bound rotation chain against the project's
rot_only called independently, on the frames pass 2 has already written."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(P, 'stages/fast_search'))
sys.path.insert(0, os.path.join(P, 'stages'))
sys.path.insert(0, '/home/zhangcy/.dsh_scratch')
import fast_lib as F
import pipeline as Pn
from rot_only_snapshot import rot_only

SH = '/home/zhangcy/.dsh_scratch/sweep_rot_smoke'
z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)

refs = json.load(open(P + '/inputs/refs.json'))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
maps = F.MapSource(refs, pos, radius=60.0)
cfg = Pn.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=1)

EPS = (1e-6, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
files = sorted(f for f in os.listdir(SH) if f.endswith('.npz'))
print('shards', len(files))
worst = 0.0
n = 0
for name in files[:2]:
    s = np.load(os.path.join(SH, name))
    for j, i in enumerate(s['idx']):
        if not np.isfinite(s['rot'][j, 0, 0, 0]):
            continue
        i = int(i)
        k = keys[i]
        date, ts = k.split('/')
        rec = {'key': k, 'scan_path': '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin' % (date, ts)}
        scan = Pn.prepare_scan(rec, cfg)
        T0 = t0[i]
        mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
        tree = cKDTree(mp)
        q = scan[np.linalg.norm(scan, axis=1) <= 50]
        _, ids = np.unique(np.floor(q / 0.3).astype(np.int64), axis=0, return_index=True)
        q = q[np.sort(ids)]
        for m, eps in enumerate(EPS):
            ref = rot_only(tree, mp, q, T0, eps=eps)
            d = np.abs(ref - s['rot'][j, m]).max()
            worst = max(worst, d)
            n += 1
        # also: the unsorted-q variant must differ, or the test proves nothing
print('compared %d (frame, eps) pairs: max |shared-prefix - project rot_only| = %.3e' % (n, worst))
