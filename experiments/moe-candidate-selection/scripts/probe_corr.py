import json
p = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/match-frozen_nopruneopt/000000.json'
d = json.load(open(p))
for k in ('per_pair', 'full_candidates', 'capped_candidates', 'selected', 'accepted'):
    v = d.get(k)
    print('%-18s type=%s' % (k, type(v).__name__), end=' ')
    if isinstance(v, list):
        print('len=%d' % len(v))
        if v:
            e = v[0]
            print('   elem type', type(e).__name__, end=' ')
            if isinstance(e, dict):
                print('keys', list(e)[:10])
            elif isinstance(e, (list, tuple)):
                print('len', len(e), 'first', str(e[0])[:80])
            else:
                print(str(e)[:100])
    elif isinstance(v, dict):
        print('keys', list(v)[:10])
        k0 = list(v)[0]
        print('   first value:', str(v[k0])[:120])
    else:
        print(str(v)[:120])
