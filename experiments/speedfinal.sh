#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from scipy.spatial import cKDTree
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rtree=cKDTree(rpos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
SC=np.arange(-10,10.01,1.0); DC=np.arange(-2,2.001,0.5)
SF=np.arange(-0.5,0.501,0.25); DF=np.arange(-0.4,0.401,0.1)
NC=1500; TOPK=6

def frame(i, workers=4):
    t={}
    t0=time.perf_counter()
    c=v7[i][:3,3]
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    clouds=[]
    for j in near:
        r=refs[j]; T=np.asarray(r['world_from_body'])
        w=vox(decode(r['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    t['map+query']=time.perf_counter()-t0
    QC=Q[:min(NC,len(Q))]; QF=Q[:min(4000,len(Q))]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
    off=np.outer(DC,v)
    t0=time.perf_counter()
    P=np.empty((len(SC)*len(DC)*len(QC),3))
    for a,s in enumerate(SC):
        P[a*len(DC)*len(QC):(a+1)*len(DC)*len(QC)]=(bc[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
    dd=tree.query(P,k=1,workers=workers)[0]
    cost=np.minimum(dd.reshape(len(SC),len(DC),-1),1.0).mean(2)
    t['coarse']=time.perf_counter()-t0
    picks=[]
    for flat in np.argsort(cost,axis=None):
        a,b=np.unravel_index(int(flat),cost.shape)
        if all(abs(SC[a]-SC[pa])>=1.5 for pa,_ in picks): picks.append((a,b))
        if len(picks)>=TOPK: break
    POSE=[]
    for a,b in picks:
        for ds in SF:
            s=SC[a]+ds
            if not(-10<=s<=10): continue
            for dd_ in DF:
                d2=DC[b]+dd_
                if -2<=d2<=2: POSE.append((s,d2))
    t0=time.perf_counter()
    P=np.empty((len(POSE)*len(QF),3))
    for k_,(s,d2) in enumerate(POSE):
        P[k_*len(QF):(k_+1)*len(QF)]=bf+(T0[:3,3]+s*u+d2*v)
    df=tree.query(P,k=1,workers=workers)[0]
    cf=np.minimum(df.reshape(len(POSE),-1),1.0).mean(1)
    t['fine']=time.perf_counter()-t0
    t['total']=sum(t.values())
    t['evals']=len(SC)*len(DC)+len(POSE)
    return t

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
sm=[]
for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): sm.extend(int(i) for i in idx[np.linspace(0,len(idx)-1,min(10,len(idx)),dtype=int)])
for w in (1,4):
    frame(sm[0],w)
    ts=[frame(i,w) for i in sm]
    print('\nworkers=%d  单进程 %d 帧  求值次数=%d' % (w,len(sm),ts[0]['evals']))
    for k in ('map+query','coarse','fine','total'):
        a=np.array([t[k] for t in ts])
        print('   %-10s 中位 %6.0f ms   均值 %6.0f ms' % (k, np.median(a)*1000, a.mean()*1000))
EOF
