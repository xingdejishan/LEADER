import json, os, numpy as np
RES='/home/zhangcy/projects/e2e-moe-20261020-v1/results'
d=[json.loads(l) for l in open(os.path.join(RES,'three_route_merged_diag.jsonl'))]
print('merged: visual=%d of %d'%(sum(1 for x in d if x.get('visual')),len(d)))
print('ratio<0.21: %d'%sum(1 for x in d if x['ratio']==x['ratio'] and x['ratio']<0.21))
print('nan ratio: %d'%sum(1 for x in d if x['ratio']!=x['ratio']))

V='/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF/%06d.json'
ncand=[]
for i in range(1502):
    r=json.load(open(V%i))['poses']
    ncand.append(len([k for k in r if k.startswith('candidate_')]))
print('frames with >=1 candidate:', sum(1 for x in ncand if x>0))
print('hist:', {k:ncand.count(k) for k in sorted(set(ncand))})
sample=None
for i in range(1502):
    r=json.load(open(V%i))['poses']
    if any(k.startswith('candidate_') for k in r):
        sample=(i,r); break
i,r=sample
print('sample', i, 'entries', list(r.keys()))
k=[x for x in r if x.startswith('candidate_')][0]
print(json.dumps(r[k],indent=1)[:1800])
print('baseline_V7 keys:', list(r['baseline_V7'].keys()))
print(json.dumps(r['baseline_V7'],indent=1)[:800])
