"""Is the attitude error a property of the *place* (the reference map) or of the frame?

The map is built from a different recording day, so the same spot is seen by the
query on four other days.  If the front end's attitude error comes from fitting a
map that is itself rotated, the error at one spot must repeat across days; if it
is estimation noise, errors at the same spot are independent.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

Z = np.load(PROJECT + 'results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
dates = np.array([k.split('/')[0] for k in keys])
pos = t0[:, :3, 3]

w = np.zeros((len(t0), 3))
for i in range(len(t0)):
    R = t0[i, :3, :3].T @ gt[i, :3, :3]
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    w[i] = np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)

level = np.linalg.norm(w, axis=1).mean()
print('frames %d ; mean |rotation error| %.3f deg' % (len(w), level))

xy = pos[:, :2]
tree = cKDTree(xy)
pairs = tree.query_pairs(3.0, output_type='ndarray')
print('pairs within 3 m: %d' % len(pairs))

diff = dates[pairs[:, 0]] != dates[pairs[:, 1]]
cross = pairs[diff]
same = pairs[~diff]
print('  cross-date %d ; same-date %d' % (len(cross), len(same)))

dw = np.linalg.norm(w[cross[:, 0]] - w[cross[:, 1]], axis=1)
dd = np.linalg.norm(pos[cross[:, 0]] - pos[cross[:, 1]], axis=1)
close = dd < 1.5
print('\n## same place, different day (separation < 1.5 m: %d pairs)' % int(close.sum()))
print('   mean angle between the two error vectors : %.3f deg' % dw[close].mean())
print('   independent noise would give             : %.3f deg' % (np.sqrt(2) * level))
print('   identical (map-induced) would give       : 0.000 deg')
cc = np.corrcoef(w[cross[close, 0]].ravel(), w[cross[close, 1]].ravel())[0, 1]
print('   component correlation across days        : %+.3f  (independent = 0)' % cc)

ws = np.linalg.norm(w[same[:, 0]] - w[same[:, 1]], axis=1)
print('\n## same place, same day (temporal neighbours): mean angle %.3f deg' % ws.mean())

# split into two halves by date and see whether a per-place correction generalises
dA = dates == '2012-02-12'
dB = dates == '2012-02-19'
for a, b in ((dA, dB), (dB, dA)):
    tr = cKDTree(pos[a][:, :2])
    dd2, ii = tr.query(pos[b][:, :2], k=1, distance_upper_bound=1.5)
    ok = np.isfinite(dd2)
    if ok.sum() < 10:
        continue
    src = np.where(a)[0][ii[ok]]
    dst = np.where(b)[0][ok]
    print('   train %s -> test %s (%d frames within 1.5 m): MOE %.3f -> %.3f after '
          'subtracting the neighbour day\'s error'
          % (dates[src][0], dates[dst][0], int(ok.sum()),
             np.linalg.norm(w[dst], axis=1).mean(),
             np.linalg.norm(w[dst] - w[src], axis=1).mean()))
