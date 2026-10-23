"""What would a perfect gate buy? (read-only)"""
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

z = np.load(P + 'full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items() if False else []:
        pass

import json
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)

e0 = np.empty(len(keys)); e1 = np.empty(len(keys))
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth[(d, ts)]
    e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
    e1[i] = np.linalg.norm(fin[i][:3, 3] - g[:3, 3])

oracle = np.minimum(e0, e1)
print('front end only        MPE=%.4f' % e0.mean())
print('current (all search)  MPE=%.4f' % e1.mean())
print('perfect gate (oracle) MPE=%.4f' % oracle.mean())
print('best scalar gate ~0.02 MPE~0.1825 (computed from the gain buckets)')
d = e1 - e0
print('\nframes where the branch is better by >0.2 m: %d (%.1f%%)'
      % (int((d < -0.2).sum()), 100.0 * (d < -0.2).mean()))
print('frames where it is worse by >0.2 m: %d (%.1f%%)'
      % (int((d > 0.2).sum()), 100.0 * (d > 0.2).mean()))
print('oracle gain over the best scalar gate: %.4f m' % (0.1825 - oracle.mean()))
