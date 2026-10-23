import json
p = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/match-frozen_nopruneopt/000000.json'
d = json.load(open(p))
print('top keys:', list(d.keys())[:12])
h = d.get('hypotheses') or []
print('hypotheses:', len(h))
if h:
    print('hypothesis keys:', list(h[0].keys()))
    for k, v in h[0].items():
        print('  %-22s %s' % (k, str(v)[:110]))
