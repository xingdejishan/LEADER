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
    s=np.fromfile(p,dtype=dt); return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
S=np.arange(-10,10.01,1.0); DC=np.arange(-2,2.001,0.5)
NC=1500; W=4

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
sm=[]
for lo,hi in [(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): sm.append(int(idx[len(idx)//2]))

def prep(i):
    c=v7[i][:3,3]
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    clouds=[]
    for j in near:
        r=refs[j]; T=np.asarray(r['world_from_body'])
        w=vox(decode(r['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)[:NC]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    return M,Q,v7[i].copy(),u,v

def build_batch(Q,T0,u,v):
    R3=T0[:3,:3]; base=Q@R3.T; off=np.outer(DC,v)
    P=np.empty((len(S)*len(DC)*len(Q),3))
    for a,s in enumerate(S):
        P[a*len(DC)*len(Q):(a+1)*len(DC)*len(Q)]=(base[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
    return P

def grid_lookup(M,cs=1.0):
    k=np.floor(M/cs).astype(np.int64)
    key=(k[:,0]+(1<<20))*((1<<21)*(1<<21))+(k[:,1]+(1<<20))*(1<<21)+(k[:,2]+(1<<20))
    o=np.argsort(key); ks=key[o]; ps=M[o]
    uk,st,cnt=np.unique(ks,return_index=True,return_counts=True)
    return uk,ps,st,cnt,cs

def grid_query(P,G):
    uk,ps,st,cnt,cs=G
    k=np.floor(P/cs).astype(np.int64)
    best=np.full(len(P),1e9)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                ck=((k[:,0]+dx+(1<<20))*((1<<21)*(1<<21))+(k[:,1]+dy+(1<<20))*(1<<21)+(k[:,2]+dz+(1<<20)))
                j=np.searchsorted(uk,ck); jc=np.clip(j,0,len(uk)-1)
                hit=(j<len(uk))&(uk[jc]==ck)
                idx=np.where(hit)[0]
                for q in idx:
                    b=st[jc[q]]; n=cnt[jc[q]]
                    dd=np.linalg.norm(ps[b:b+n]-P[q],axis=1).min()
                    if dd<best[q]: best[q]=dd
    return best

print('%-22s %10s %10s' % ('方法','耗时(ms)','与基准最大差'))
for i in sm:
    M,Q,T0,u,v=prep(i)
    P=build_batch(Q,T0,u,v)
    tree=cKDTree(M)
    # baseline: plain query
    t0=time.perf_counter(); d0=tree.query(P,k=1,workers=W)[0]; t0=time.perf_counter()-t0
    # A: distance_upper_bound
    t1=time.perf_counter()
    d1=tree.query(P,k=1,workers=W,distance_upper_bound=1.0)[0]
    d1=np.where(np.isfinite(d1),d1,1.0)
    t1=time.perf_counter()-t1
    # B: grid
    G=grid_lookup(M,1.0)
    t2=time.perf_counter(); d2=grid_query(P,G); t2=time.perf_counter()-t2
    c0=np.minimum(d0,1.0).mean(); c1=np.minimum(d1,1.0).mean(); c2=np.minimum(d2,1.0).mean()
    print('frame %4d  基准 %8.1f | 上界 %8.1f (差%.1e) | 网格 %8.1f (差%.1e)'
          % (i,t0*1000,t1*1000,abs(c1-c0),t2*1000,abs(c2-c0)))
EOF
