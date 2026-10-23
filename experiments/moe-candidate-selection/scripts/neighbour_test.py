"""Do the branch's failures have good neighbours? (read-only)"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], float)
gain = np.asarray(z['gain'], float)
P_search = np.asarray(z['P_search'], float)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

n = len(keys)
e0 = np.empty(n)
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth[(d, ts)]
    e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])

order = sorted(range(n), key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1])))
rank = {i: r for r, i in enumerate(order)}
e0_sorted = e0[order]

bad = [i for i in range(n) if 5.0 <= e0[i] < 8.0]
print('frames with front-end error 5-8 m: %d' % len(bad))
for lag in (1, 2, 5):
    vals = []
    for i in bad:
        r = rank[i]
        for off in (-lag, lag):
            j = r + off
            if 0 <= j < n and keys[order[j]].split('/')[0] == keys[i].split('/')[0]:
                vals.append(e0_sorted[j])
    v = np.array(vals)
    if len(v):
        print('  neighbours at +-%d frame: n=%d  median error %.2f m   within 0.5 m: %.0f%%   >2 m: %.0f%%'
              % (lag, len(v), np.median(v), 100 * (v <= 0.5).mean(), 100 * (v > 2).mean()))

# how isolated are the failures overall?
print('\nisolation of all frames with error > 2 m:')
bad2 = [i for i in range(n) if e0[i] > 2.0]
print('  count %d' % len(bad2))
for lag in (1, 2):
    alone = 0
    for i in bad2:
        r = rank[i]
        nb = [e0_sorted[r + off] for off in (-lag, lag)
              if 0 <= r + off < n and keys[order[r + off]].split('/')[0] == keys[i].split('/')[0]]
        if nb and all(x > 2.0 for x in nb):
            alone += 1
    print('  with both +-%d neighbours also >2 m: %d (%.0f%%)' % (lag, alone, 100.0 * alone / max(len(bad2), 1)))

# how much of the branch correction is temporally incoherent?
corr = np.linalg.norm(P_search[:, :3, 3] - t0[:, :3, 3], axis=1)
run = gain > 0.01
print('\ncorrection magnitude on searched frames: n=%d median %.2f m p90 %.2f m'
      % (int(run.sum()), np.median(corr[run]), np.percentile(corr[run], 90)))
big = run & (corr > 2.0)
print('  corrections larger than 2 m: %d' % int(big.sum()))
