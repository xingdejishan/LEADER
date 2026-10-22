"""Correlation of the image-rotation error with the front-end error, on the 400-frame run."""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SRC = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_validate.npz'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def expv(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


rec = list(np.load(SRC, allow_pickle=True)['rec'])
print('records %d' % len(rec))
for tag, key in (('anchored', 'db_an'), ('prior-free', 'db_fr')):
    sel = [r for r in rec if key in r]
    i = np.array([r['i'] for r in sel], int)
    d = np.array([r[key] for r in sel], float)
    ep = np.array([logv(t0[k, :3, :3].T @ gt[k, :3, :3]) for k in i])
    ei = np.array([logv(expv(d[j]) @ t0[i[j], :3, :3].T @ gt[i[j], :3, :3])
                   for j in range(len(i))])
    ep, ei = np.degrees(ep), np.degrees(ei)
    print('\n## %s  n=%d' % (tag, len(i)))
    print('   mean |front-end err| %.3f   mean |image err| %.3f'
          % (np.linalg.norm(ep, axis=1).mean(), np.linalg.norm(ei, axis=1).mean()))
    print('   correlation of the two error vectors (pooled components) %+.3f'
          % np.corrcoef(ep.ravel(), ei.ravel())[0, 1])
    print('   mean |image err - front-end err| %.3f  (independent noise => %.3f)'
          % (np.linalg.norm(ei - ep, axis=1).mean(),
             np.sqrt(np.linalg.norm(ep, axis=1).mean() ** 2
                     + np.linalg.norm(ei, axis=1).mean() ** 2)))
    base = ra(t0[i, :3, :3], gt[i, :3, :3])
    R = np.stack([t0[i[j], :3, :3] @ expv(d[j]) for j in range(len(i))])
    img = ra(R, gt[i, :3, :3])
    print('   MOE: front end %.3f -> image %.3f' % (base.mean(), img.mean()))
