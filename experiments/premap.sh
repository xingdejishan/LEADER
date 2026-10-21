#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from collections import OrderedDict
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
    s=np.fromfile(p,dtype=dt); return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]

class LRU:
    def __init__(self,n): self.n=n; self.d=OrderedDict(); self.hit=0; self.miss=0
    def get(self,k):
        d=self.d
        if k in d:
            d.move_to_end(k); self.hit+=1; return d[k]
        self.miss+=1; return None
    def put(self,k,v):
        d=self.d; d[k]=v; d.move_to_end(k)
        while len(d)>self.n: d.popitem(last=False)

REF=LRU(400)
MAP=LRU(300)
def ref_cloud(j):
    r=REF.get(j)
    if r is not None: return r
    T=np.asarray(refs[j]['world_from_body'])
    w=vox(decode(refs[j]['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
    REF.put(j,w); return w

SC=np.arange(-10,10.01,1.0)
SF=np.arange(-0.5,0.501,0.25); DF=np.arange(-0.4,0.401,0.1)
D0=np.arange(-2,2.001,0.25)
NC=1500; TOPK=6; W=4
MODE='ref'

def q(tree,P):
    d=tree.query(P,k=1,workers=W,distance_upper_bound=1.0)[0]
    return np.where(np.isfinite(d),d,1.0)

def frame(i):
    t={}; t0=time.perf_counter()
    c=v7[i][:3,3]
    cell=tuple(np.round(c).astype(np.int64))
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    key=(tuple(near),cell)
    M=MAP.get(key) if MODE=='map' else None
    t['取图']=0.0
    if M is None:
        clouds=[]
        for j in near:
            w=ref_cloud(j)
            clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
        M=vox(np.concatenate(clouds),0.3)
        if MODE=='map': MAP.put(key,M)
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    t['建图+读扫描']=time.perf_counter()-t0
    QC=Q[:min(NC,len(Q))]; QF=Q[:min(4000,len(Q))]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
    t0=time.perf_counter()
    P=np.empty((len(D0)*len(QC),3))
    for a,d in enumerate(D0): P[a*len(QC):(a+1)*len(QC)]=bc+(T0[:3,3]+d*v)
    c0=np.minimum(q(tree,P).reshape(len(D0),-1),1.0).mean(1)
    d0=float(D0[int(np.argmin(c0))])
    DC=np.clip(np.array([d0-0.5,d0,d0+0.5]),-2,2)
    off=np.outer(DC,v)
    P=np.empty((len(SC)*len(DC)*len(QC),3))
    for a,s in enumerate(SC):
        P[a*len(DC)*len(QC):(a+1)*len(DC)*len(QC)]=(bc[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
    cost=np.minimum(q(tree,P).reshape(len(SC),len(DC),-1),1.0).mean(2)
    t['粗搜']=time.perf_counter()-t0
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
    cf=np.minimum(q(tree,P).reshape(len(POSE),-1),1.0).mean(1)
    t['细搜']=time.perf_counter()-t0
    tt=int(np.argmin(cf)); s,d2=POSE[tt]
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d2*v
    t['总']=t['建图+读扫描']+t['粗搜']+t['细搜']
    t['误差']=float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))
    return t

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
sm=[]
for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): sm.extend(int(x) for x in idx[np.linspace(0,len(idx)-1,min(10,len(idx)),dtype=int)])
for mode in ('ref','map'):
    globals()['MODE']=mode
    REF.hit=REF.miss=MAP.hit=MAP.miss=0
    frame(sm[0])
    ts=[frame(i) for i in sm]
    print('\nMODE=%s  参考缓存命中 %.0f%%  地图缓存命中 %.0f%%'
          % (mode,100*REF.hit/max(1,REF.hit+REF.miss),100*MAP.hit/max(1,MAP.hit+MAP.miss)))
    for k in ('建图+读扫描','粗搜','细搜','总'):
        a=np.array([t[k] for t in ts])
        print('   %-10s 中位 %5.0f ms' % (k,np.median(a)*1000))
    e=np.array([t['误差'] for t in ts])
    print('   误差 中位 %.2f  进一米 %.0f%%' % (np.median(e),100*(e<=1).mean()))
EOF
