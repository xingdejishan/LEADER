"""Gated blend: does the image increment help where it is large?

If the front end is usually right, a small image increment is noise; a large one may
mean the front end is wrong.  This groups the 400 frames by the size of the image
increment, sweeps the blend factor inside each group, and repeats the check on a
split half so a fluke is visible.
"""
import json
import os

import numpy as np
from scipy import stats

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SRC = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_validate.npz'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(str(k).split('/'))] for k in Z['keys']])


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


rec = [r for r in np.load(SRC, allow_pickle=True)['rec'] if 'db_fr' in r]
i = np.array([r['i'] for r in rec], int)
d = np.array([r['db_fr'] for r in rec], float)
step = np.degrees(np.linalg.norm(d, axis=1))
R0, G = t0[i, :3, :3], gt[i, :3, :3]
base = ra(R0, G)
print('frames %d ; |increment| p25/50/75 = %s'
      % (len(i), np.round(np.percentile(step, [25, 50, 75]), 2)))

half = np.arange(len(i)) % 2 == 0
for lo, hi in ((0, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 99)):
    m = (step >= lo) & (step < hi)
    if m.sum() < 20:
        continue
    line = '  step [%.1f,%.1f) n=%3d  base %.3f |' % (lo, hi if hi < 99 else 9.9,
                                                     int(m.sum()), base[m].mean())
    for lam in (0.0, 0.3, 0.5, 0.8):
        R = np.stack([R0[j] @ expv(lam * d[j]) for j in np.where(m)[0]])
        e = ra(R, G[m])
        gain = base[m].mean() - e.mean()
        line += '  l%.1f %+.3f' % (lam, -gain)
    print(line)

print('\n## split-half consistency of gating on the increment size (>1.2 deg)')
big = step > 1.2
for lam in (0.3, 0.5):
    for name, sel in (('first half', big & half), ('second half', big & ~half)):
        R = np.stack([R0[j] @ expv(lam * d[j]) for j in np.where(sel)[0]])
        e = ra(R, G[sel])
        b = base[sel]
        better = int((e < b).sum())
        p = stats.binomtest(better, int(sel.sum()), 0.5).pvalue if sel.sum() else 1.0
        print('   lam %.1f %-11s n=%3d  MOE %.3f -> %.3f  improved %d/%d  p=%.3f'
              % (lam, name, int(sel.sum()), b.mean(), e.mean(), better, int(sel.sum()), p))
