"""Per-frame paired test of the image/front-end rotation fusion (reads expQ_corr.npz)."""
import json
import os

import numpy as np
from scipy import stats

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SRC = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expQ_corr.npz'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])


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


rows = list(np.load(SRC, allow_pickle=True)['rows'])
print('rows %d' % len(rows))
for tag, key in (('anchored', 'db_an'), ('prior-free', 'db_fr')):
    sel = [r for r in rows if key in r]
    print('\n## %s  (n=%d)' % (tag, len(sel)))
    i = np.array([r['i'] for r in sel], int)
    d = np.array([r[key] for r in sel], float)
    R0 = t0[i, :3, :3]
    G = gt[i, :3, :3]
    base = ra(R0, G)
    print('   %-6s %8s %8s %8s %9s %9s' % ('lam', 'mean', 'median', 'p90', 'improved', 'sign p'))
    for lam in (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        R = np.stack([R0[j] @ expv(lam * d[j]) for j in range(len(i))])
        e = ra(R, G)
        better = int((e < base).sum())
        p = stats.binomtest(better, len(i), 0.5).pvalue
        print('   %-6.2f %8.3f %8.3f %8.3f %5d/%-4d %9.4f'
              % (lam, e.mean(), np.median(e), np.percentile(e, 90), better, len(i), p))
    # per-frame gain distribution at the best lambda
    lam = 0.3
    R = np.stack([R0[j] @ expv(lam * d[j]) for j in range(len(i))])
    e = ra(R, G)
    gain = base - e
    print('   at lam %.2f: mean gain %+.3f deg, median %+.3f, frac improved %.1f%%'
          % (lam, gain.mean(), np.median(gain), 100 * (gain > 0).mean()))
    # does the gain correlate with the size of the image correction?
    step = np.degrees(np.linalg.norm(d, axis=1))
    print('   corr(image correction size, gain) = %+.3f ; median correction %.2f deg'
          % (np.corrcoef(step, gain)[0, 1], np.median(step)))
