import json, numpy as np
B='/home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1'
TR='/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth/%s.json'
SCORED='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz'
d=np.load(B+'/results/full-online-final.npz',allow_pickle=True)
keys=[str(k) for k in d['keys']]; t0=np.asarray(d['t0'],float)
truth={dt: json.load(open(TR%dt)) for dt in sorted({k.split('/')[0] for k in keys})}
rot=np.full(len(keys),np.nan)
for i,k in enumerate(keys):
    dt,s=k.split('/'); r=truth[dt].get(s)
    if r is None: continue
    T=np.asarray(r['T_WB'],float)
    rot[i]=np.degrees(np.arccos(np.clip((np.trace(t0[i][:3,:3].T@T[:3,:3])-1)/2,-1,1)))
sc=np.load(SCORED,allow_pickle=True)
skey=[str(k) for k in sc['keys']]; rep=np.asarray(sc['replaced'],bool)
rep_keys=[skey[i] for i in np.flatnonzero(rep)]
idx={k:i for i,k in enumerate(keys)}
ws=np.array([int(k.split('/')[1]) for k in rep_keys],dtype=np.int64); lo,hi=ws.min(),ws.max()
off=np.asarray([np.asarray(sc['reference'][i],float)[:3,3]-np.asarray(sc['raw'][i],float)[:3,3] for i in np.flatnonzero(rep)])
med=np.median(off,axis=0)
clean=np.ones(len(keys),bool)
for k in rep_keys:
    if k in idx: clean[idx[k]]=False
for i,k in enumerate(keys):
    dt,s=k.split('/')
    if dt!='2012-02-12' or not clean[i]: continue
    ts=int(s)
    if lo<=ts<=hi:
        v=t0[i][:3,3]-np.asarray(truth[dt][s]['T_WB'],float)[:3,3]
        n=np.linalg.norm(v)
        if n>5.0 and float(v@med/max(1e-12,n*np.linalg.norm(med)))>0.99: clean[i]=False
ok=np.isfinite(rot)
high=np.flatnonzero(clean&ok&(rot>3.0)); ctrl=np.flatnonzero(clean&ok&(rot<=3.0))
print('clean',int(clean.sum()),'rot>3',len(high),'ctrl',len(ctrl))
def alloc(items,m):
    by={}
    for i in items: by.setdefault(keys[i].split('/')[0],[]).append(i)
    tot=len(items); ds=sorted(by); base={d:len(by[d])/tot*m for d in ds}
    take={d:int(np.floor(base[d])) for d in ds}
    rem=m-sum(take.values())
    for d in sorted(ds,key=lambda x:-(base[x]-take[x]))[:rem]: take[d]+=1
    return by,take
rng=np.random.RandomState(20261024)
by,take=alloc(high,30)
sel=[]
for d in sorted(by): sel+= [keys[x] for x in rng.choice(by[d],take[d],replace=False)]
by2,take2=alloc(ctrl,10)
sel2=[]
for d in sorted(by2): sel2+= [keys[x] for x in rng.choice(by2[d],take2[d],replace=False)]
allk=sel+sel2
print('high per date',{k.split('/')[0]:take[k.split('/')[0]] for k in []} or take)
print('ctrl per date',take2)
idx2={k:i for i,k in enumerate(keys)}
json.dump(allk,open('/data2/zhangcy/dsh-angle-20261024/pilot_keys.json','w'))
json.dump([dict(key=k,role='high' if k in set(sel) else 'ctrl',t0_rot=float(rot[idx2[k]])) for k in allk],
          open('/data2/zhangcy/dsh-angle-20261024/pilot_roles.json','w'),indent=1)
print('selected',len(allk))
print('high t0 rot range', round(min(rot[idx2[k]] for k in sel),2), round(max(rot[idx2[k]] for k in sel),2))
print('ctrl t0 rot range', round(min(rot[idx2[k]] for k in sel2),2), round(max(rot[idx2[k]] for k in sel2),2))
