"""Does the geometric branch help exactly the frames it is designed for?

Buckets the produced frames by the front-end (t0) error and reports what the
geometric branch did inside each bucket. Read-only.
"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']

rows = []
for name in FILES:
    try:
        with open(P + name) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('skipped') or not r.get('seconds'):
                    continue
                rows.append(r)
    except IOError:
        pass

truth = {}
for f in os.listdir(TRUTH):
    date = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)

e0, e1 = [], []
for r in rows:
    date, ts = r['key'].split('/')
    g = truth.get((date, ts))
    if g is None:
        continue
    T0 = np.asarray(r['t0'], np.float64).reshape(4, 4)
    T1 = np.asarray(r['pose'], np.float64).reshape(4, 4)
    e0.append(np.linalg.norm(T0[:3, 3] - g[:3, 3]))
    e1.append(np.linalg.norm(T1[:3, 3] - g[:3, 3]))
e0 = np.array(e0); e1 = np.array(e1)
d = e1 - e0
print('frames=%d  overall MPE: front end %.4f -> MoE %.4f' % (len(e0), e0.mean(), e1.mean()))

edges = [(0, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 8), (8, 1e9)]
print('\n%-12s %7s %9s %9s %8s %8s %8s' %
      ('t0 error bin', 'n', 'mean t0', 'mean MoE', 'delta', 'helped', 'damaged'))
for lo, hi in edges:
    m = (e0 >= lo) & (e0 < hi)
    if m.sum() == 0:
        continue
    print('%-12s %7d %9.3f %9.3f %8.3f %8d %8d'
          % ('%.1f-%.1f m' % (lo, hi) if hi < 1e8 else '>8 m', int(m.sum()),
             e0[m].mean(), e1[m].mean(), d[m].mean(),
             int((d[m] < -0.1).sum()), int((d[m] > 0.1).sum())))

print('\nfocus 2-8 m: n=%d  mean %.3f -> %.3f (delta %.3f)  helped %d  damaged %d'
      % (int(((e0 >= 2) & (e0 < 8)).sum()),
         e0[(e0 >= 2) & (e0 < 8)].mean(), e1[(e0 >= 2) & (e0 < 8)].mean(),
         d[(e0 >= 2) & (e0 < 8)].mean(),
         int((d[(e0 >= 2) & (e0 < 8)] < -0.1).sum()),
         int((d[(e0 >= 2) & (e0 < 8)] > 0.1).sum())))
