import json
import os

import numpy as np

R = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
files = ['results/full-online-final.npz', 'results/full-online.npz',
         'results/fullb-shard1.npz', 'results/fullb-shard3.npz',
         'results/hybrid_rotation.npz', 'results/final.npz',
         'inputs/initial.npz', 'results/three_route_unified.npz',
         'results/yaw_sweep.npz', 'results/sweep-summary.npz',
         'results/three_route_merged.npz']
for f in files:
    p = R + f
    if not os.path.exists(p):
        print(f, 'MISSING')
        continue
    z = np.load(p, allow_pickle=True)
    print('==', f, os.path.getsize(p) // 1024, 'KB', len(z.files), 'arrays')
    for k in z.files[:14]:
        a = z[k]
        extra = ''
        if a.ndim == 1 and a.size and a.dtype.kind in 'OUS':
            extra = ' :: ' + str(a[:3])[:120]
        print('    %-24s %-18s %s%s' % (k, str(a.shape), a.dtype, extra))

tp = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth/2012-02-12.json'
d = json.load(open(tp))
print('\ntruth entries:', len(d))
for i, (ts, v) in enumerate(d.items()):
    print('  ts', ts, 'keys', list(v.keys()))
    print('  ', json.dumps(v)[:400])
    if i >= 1:
        break
