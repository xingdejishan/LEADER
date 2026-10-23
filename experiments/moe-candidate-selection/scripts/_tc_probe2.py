import json, os
import numpy as np
P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']
rows = {}
for name in FILES:
    try:
        with open(P + name) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('key') and r.get('seconds'):
                    rows[r['key']] = r
    except IOError as e:
        print('missing', name, e)
print('rows', len(rows))
r0 = next(iter(rows.values()))
print('sample row keys:', sorted(r0.keys()))
print(json.dumps({k: (v if not isinstance(v, (list, dict)) else type(v).__name__)
                  for k, v in r0.items()}, default=str)[:600])
for k in ('solver', 'coarse_points', 'fine_points', 'topk', 's', 'd', 'cost', 'evals', 'expanded'):
    if k in r0:
        print(' ', k, '=', r0[k])

z = np.load(P + 'full-online.npz', allow_pickle=True)
keys = [str(x) for x in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
truth = {}
for f in os.listdir(TRUTH):
    dd = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(dd, str(ts))] = np.asarray(v['T_WB'], np.float64)
e0 = np.array([np.linalg.norm(t0[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3])
               for i in range(len(keys))])
band = np.where((e0 >= 5) & (e0 <= 8))[0]
print('\nband n =', len(band))
S = [rows[keys[i]] for i in band if keys[i] in rows]
print('rows found for band:', len(S))
sv = np.array([r['s'] for r in S]); dv = np.array([r['d'] for r in S])
cv = np.array([r['cost'] for r in S])
print('s: zero-count %d, min %.2f max %.2f' % (int((np.abs(sv) < 1e-9).sum()), sv.min(), sv.max()))
print('d: min %.2f max %.2f' % (dv.min(), dv.max()))
print('cost mean %.3f' % cv.mean())
ev = [r.get('expanded') for r in S]
print('expanded', ev[:10])
print('evals', [r.get('evals') for r in S][:5])
