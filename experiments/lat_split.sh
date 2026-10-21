#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
cd /home/zhangcy/projects/r5b-lossless-20261019-v1
$PY - <<'EOF'
import json, sys, time, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/home/zhangcy/projects/r5b-lossless-20261019-v1')
import fast_lib as F
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
per={}
for i,k in enumerate(keys): per.setdefault(k.split('/')[0],[]).append((int(k.split('/')[1]),i))
d=sorted(per)[0]; seq=[i for _,i in sorted(per[d])][:200]
MS=F.MapSource(refs,rpos)

tc=[];tf=[];tb=[];nel=[]
for n,i in enumerate(seq):
    c=v7[i][:3,3]; M=MS.local(c)
    q=F.decode(qmap[keys[i]]['scan_path'])
    q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]
    q=F.voxel(q,0.3)
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    tree=cKDTree(M); R3=v7[i][:3,:3]; o=v7[i][:3,3]
    lat=np.cross(u,[0,0,1.0]); lat/=np.linalg.norm(lat)
    qc=q[:1500]; qf=q[:4000]
    bc=qc@R3.T; bf=qf@R3.T
    t0=time.perf_counter()
    P=np.empty((len(F.D_ORIGIN)*len(qc),3))
    for a,dd in enumerate(F.D_ORIGIN): P[a*len(qc):(a+1)*len(qc)]=bc+(o+dd*lat)
    c0=np.minimum(F.query_dist(tree,P).reshape(len(F.D_ORIGIN),-1),1.0).mean(1)
    d0=float(F.D_ORIGIN[int(np.argmin(c0))])
    dc=np.clip(np.array([d0-0.5,d0,d0+0.5]),-2,2)
    off=np.outer(dc,lat)
    P=np.empty((len(F.S_COARSE)*len(dc)*len(qc),3))
    for a,s in enumerate(F.S_COARSE):
        P[a*len(dc)*len(qc):(a+1)*len(dc)*len(qc)]=(bc[None]+(o+s*u+off)[:,None]).reshape(-1,3)
    cost=np.minimum(F.query_dist(tree,P).reshape(len(F.S_COARSE),len(dc),-1),1.0).mean(2)
    tc.append(time.perf_counter()-t0)
    t0=time.perf_counter(); nev=0; nb=0
    best=(np.inf,0,0)
    picks=[]
    for fl in np.argsort(cost,axis=None):
        a,b=np.unravel_index(int(fl),cost.shape)
        if all(abs(F.S_COARSE[a]-F.S_COARSE[pa])>=1.5 for pa,_ in picks): picks.append((a,b))
        if len(picks)>=6: break
    for a,b in picks:
        s0,dd0=float(F.S_COARSE[a]),float(dc[b])
        cp=bf+(o+s0*u+dd0*lat)
        di,ix=tree.query(cp,k=1,workers=4,distance_upper_bound=1.0); nev+=1
        di=np.where(np.isfinite(di),di,1.0)
        ins=np.isfinite(di)&(di>1e-9)&(di<1.0)
        if ins.any():
            g=cp[ins]-M[ix[ins]]; g/=np.maximum(di[ins],1e-12)[:,None]
            sl=0.5*np.abs(g@u)+0.4*np.abs(g@lat)
            bi=np.minimum(np.maximum(di[ins]-sl,0),1).mean()
            lb=(bi*ins.sum()+float((~ins).sum()))/len(di)
            if lb>=best[0]: continue
        nb+=1
        poses=[(s0+ds,dd0+d2) for ds in F.S_FINE for d2 in F.D_FINE if -10<=s0+ds<=10 and -2<=dd0+d2<=2]
        P=np.empty((len(poses)*len(qf),3))
        for t,(s,d2) in enumerate(poses): P[t*len(qf):(t+1)*len(qf)]=bf+(o+s*u+d2*lat)
        cf=np.minimum(F.query_dist(tree,P).reshape(len(poses),-1),1.0).mean(1); nev+=len(poses)
        t2=int(np.argmin(cf))
        if cf[t2]<best[0]: best=(float(cf[t2]),poses[t2][0],poses[t2][1])
    tf.append(time.perf_counter()-t0); nel.append((nev,nb))
tc=np.array(tc)*1000; tf=np.array(tf)*1000
print('预热后 %d 帧:' % len(tc[40:]))
print('  粗搜      中位 %6.1f ms' % np.median(tc[40:]))
print('  细搜+剪枝 中位 %6.1f ms' % np.median(tf[40:]))
print('  搜索合计  中位 %6.1f ms' % np.median((tc+tf)[40:]))
print('  细搜求值  均值 %.0f ；实际精搜区域 均值 %.2f/6' % (np.mean([x[0] for x in nel[40:]]),np.mean([x[1] for x in nel[40:]])))
EOF
