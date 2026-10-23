"""Do the two gates separate what they are supposed to? (read-only)

gate 2  (descent depth)  : should predict "the search will actually help"
gate 1  (ambiguity ratio): should pick out the frames that are severely wrong
"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']

z = np.load(P + 'full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)

gain, ratio = {}, {}
for name in FILES:
    with open(P + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'key' in r and r.get('seconds'):
                gain[r['key']] = r.get('gain')
                ratio[r['key']] = r.get('ratio')
print('rows with gain: %d, with ratio: %d (of %d)'
      % (sum(1 for v in gain.values() if v is not None),
         sum(1 for v in ratio.values() if v is not None), len(keys)))

truth = {}
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
change = e1 - e0
helped = change < -0.1
damaged = change > 0.1

G = np.array([gain.get(k) if gain.get(k) is not None else np.nan for k in keys], float)
R = np.array([ratio.get(k) if ratio.get(k) is not None else np.nan for k in keys], float)


def auc(score, label):
    m = np.isfinite(score)
    s, y = score[m], label[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return float('nan'), int(m.sum())
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y.sum()
    neg = (~y).sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg), int(m.sum())


print('\n--- gate 2 (gain) ---')
a, n = auc(G, helped)
print('AUC(gain -> "search helps") = %.3f  (n=%d, positives=%d)'
      % (a, n, int(helped.sum())))
a2, n2 = auc(G, e0 > 0.5)
print('AUC(gain -> "t0 error > 0.5 m") = %.3f  (n=%d, positives=%d)'
      % (a2, n2, int((e0 > 0.5).sum())))
print('%-14s %8s %8s %8s %8s %8s' % ('gain bin', 'n', 'mean t0', 'delta', 'helped', 'damaged'))
for lo, hi in ((-1, 0), (0, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 1e9)):
    m = np.isfinite(G) & (G > lo) & (G <= hi)
    if m.sum() == 0:
        continue
    print('%-14s %8d %8.3f %8.3f %8d %8d'
          % ('%.3f-%.3f' % (lo, hi), int(m.sum()), e0[m].mean(), change[m].mean(),
             int((helped & m).sum()), int((damaged & m).sum())))

print('\n--- gate 1 (ambiguity ratio) ---')
if not np.isfinite(R).any():
    print('ratio not recorded in this run (the sweep is skipped when the visual branch is off)')
else:
    a3, n3 = auc(-R, e0 > 1.0)
    print('AUC(-ratio -> "t0 error > 1 m") = %.3f (n=%d)' % (a3, n3))
    for thr in (0.21, 0.30):
        m = np.isfinite(R) & (R < thr)
        print('ratio<%.2f: %d frames, t0 mean %.3f, >1m %d of %d, >5m %d of %d'
              % (thr, int(m.sum()), e0[m].mean(), int((e0[m] > 1).sum()), int((e0 > 1).sum()),
                 int((e0[m] > 5).sum()), int((e0 > 5).sum())))
