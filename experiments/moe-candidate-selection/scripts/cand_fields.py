import json
V = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
     'verify-frozen_nopruneoptREF/%06d.json')
d = json.load(open(V % 0))
print('pose entry names:', list(d['poses'].keys()))
c = d['poses']['candidate_00']
print('candidate fields:', list(c.keys()))
for k, v in c.items():
    s = str(v)
    print('  %-18s %s' % (k, s[:90]))
print()
print('best_candidate field:', str(d.get('best_candidate'))[:120])
# count candidates per frame and qualification-like flags
import collections
cnt = collections.Counter()
for i in range(0, 200):
    try:
        dd = json.load(open(V % i))
    except Exception:
        continue
    cands = [n for n in dd['poses'] if n.startswith('candidate_')]
    cnt[len(cands)] += 1
print('candidates per frame (first 200):', dict(cnt))
