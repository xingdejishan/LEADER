"""Is the per-frame attitude error continuous noise, or quantised by a candidate set?"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
T = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
truth = {}
for f in os.listdir(T):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(T, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(P + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
w = np.zeros((len(t0), 3))
for i in range(len(t0)):
    R = t0[i, :3, :3].T @ gt[i, :3, :3]
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    w[i] = np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)

for ax, name in enumerate('xyz'):
    h, e = np.histogram(w[:, ax], bins=30, range=(-3, 3))
    print('w_%s  mean %+.3f  std %.3f  skew %.2f  kurt %.2f'
          % (name, w[:, ax].mean(), w[:, ax].std(),
             float(((w[:, ax] - w[:, ax].mean()) ** 3).mean() / w[:, ax].std() ** 3),
             float(((w[:, ax] - w[:, ax].mean()) ** 4).mean() / w[:, ax].std() ** 4)))
    print('   ' + ' '.join('%d' % c for c in h))
n = np.linalg.norm(w, axis=1)
h, e = np.histogram(n, bins=24, range=(0, 6))
print('\n|w| histogram (0..6 deg, 0.25 bins):')
print('   ' + ' '.join('%d' % c for c in h))
print('   mean %.3f median %.3f p90 %.3f p99 %.3f'
      % (n.mean(), np.median(n), np.percentile(n, 90), np.percentile(n, 99)))
