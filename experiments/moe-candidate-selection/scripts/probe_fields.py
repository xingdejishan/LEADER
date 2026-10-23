import json, os
V = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
     'verify-frozen_nopruneoptREF/%06d.json')
d = json.load(open(V % 0))
for name in ('candidate_00', 'candidate_03'):
    c = d['poses'][name]
    print('==', name, '==')
    for k, v in c.items():
        print('  %-18s %s' % (k, str(v)[:160]))
print()
print('== match-level files available? ==')
store = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov'
for sub in sorted(os.listdir(store))[:12]:
    p = os.path.join(store, sub)
    if os.path.isdir(p):
        n = len(os.listdir(p))
        print('  %-32s %d files' % (sub, n))
