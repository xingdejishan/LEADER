"""Where are the long bursts, and is the biggest one the known suspect window? (read-only)"""
import json
import os

import numpy as np

TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], float)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

order = sorted(range(len(keys)), key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1])))
err = np.array([np.linalg.norm(t0[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3]) for i in order])
dates = np.array([keys[i].split('/')[0] for i in order])
ts = np.array([int(keys[i].split('/')[1]) for i in order])

mask = err > 2.0
runs, start = [], None
for i in range(len(mask)):
    if mask[i] and start is None:
        start = i
    if start is not None and (i + 1 >= len(mask) or not mask[i + 1] or dates[i + 1] != dates[i]):
        runs.append((start, i))
        start = None

runs.sort(key=lambda r: r[1] - r[0], reverse=True)
print('longest bursts (>2 m):')
for a, b in runs[:6]:
    print('  %s  %s -> %s  %.1f s  %3d frames  mean error %.1f m  max %.1f m'
          % (dates[a], ts[a], ts[b], (ts[b] - ts[a]) / 1e6, b - a + 1,
             err[a:b + 1].mean(), err[a:b + 1].max()))

suspect = '/home/zhangcy/projects/v7-alternating-e7-full-20261018-v6/nclt-2012-02-12-suspect255-odometry.npz'
if os.path.exists(suspect):
    s = np.load(suspect, allow_pickle=True)
    print('\nsuspect-window file keys:', list(s.keys()))
    a, b = runs[0]
    inner = (dates == dates[a]) & (ts >= ts[a]) & (ts <= ts[b])
    print('longest burst frames: %d; of them inside 2012-02-12 = %d' % (b - a + 1, int(inner.sum())))
