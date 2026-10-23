import json, os, sys, numpy as np
RES = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'
V = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
     'verify-frozen_nopruneoptREF/%06d.json')
d = [json.loads(l) for l in open(os.path.join(RES, 'three_route_merged_diag.jsonl'))]
acc = [k for k, x in enumerate(d) if x.get('visual')]
print('accepted frames:', acc)
print('their ratios:', ['%.3f' % d[i]['ratio'] for i in acc])

tot = dict(vT=0, vF=0, vMiss=0, noFair=0)
for i in range(1502):
    rec = json.load(open(V % i))['poses']
    for k, v in rec.items():
        if not k.startswith('candidate_'):
            continue
        f = v.get('fair')
        if f is None:
            tot['noFair'] += 1; continue
        g = f.get('geometry_verified')
        if g is True: tot['vT'] += 1
        elif g is False: tot['vF'] += 1
        else: tot['vMiss'] += 1
print('candidate entries:', tot)

for i in acc:
    rec = json.load(open(V % i))['poses']
    c = [(k, v) for k, v in rec.items() if k.startswith('candidate_')]
    rows = []
    for k, v in c:
        f = v.get('fair') or {}
        rows.append((k, f.get('geometry_score'), f.get('geometry_verified'), f.get('valid'),
                     f.get('residual_median_m')))
    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -9))
    print('\nframe %d  n_cand=%d  top3 (name, score, verified, valid, res_med):' % (i, len(c)))
    for r in rows[:3]:
        print('   ', r)
    nv = sum(1 for r in rows if r[2] is True)
    print('    verified candidates in this frame: %d/%d' % (nv, len(rows)))
