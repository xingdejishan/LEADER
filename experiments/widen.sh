#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, numpy as np, time
from scipy.spatial import cKDTree
from multiprocessing import Pool
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
DC=np.arange(-2,2.001,0.5)
SF=np.arange(-0.5,0.501,0.25); DF=np.arange(-0.4,0.401,0.1)
LEVELS=[(1.0,0.5),(2.0,0.5),(6.0,1.0),(10.0,1.0)]
NC=1500; TOPK=6; W=4
MODE='fixed'

def run(i):
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
    QC=Q[:min(NC,len(Q))]; QF=Q[:min(4000,len(Q))]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
    off=np.outer(DC,v)
    nev=0; best=None
    # 覆盖度：每个候选中点附近 60 米内的库点数（只计数，不取索引）
    def coarse_for(S):
        nonlocal nev
        if len(S)==0: return None
        P=np.empty((len(S)*len(DC)*len(QC),3))
        for a,s in enumerate(S):
            P[a*len(DC)*len(QC):(a+1)*len(DC)*len(QC)]=(bc[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
        dd=tree.query(P,k=1,workers=W)[0]
        nev+=len(S)*len(DC)
        return np.minimum(dd.reshape(len(S),len(DC),-1),1.0).mean(2)
    if MODE=='fixed':
        S=np.arange(-10,10.01,1.0)
        cost=coarse_for(S)
        best=(cost,S)
    else:
        allowed=None
        for L,step in LEVELS:
            S=np.arange(-L,L+1e-9,step)
            if allowed is not None:
                S=np.array([s for s in S if abs(s)<=allowed])
            cost=coarse_for(S)
            if cost is None: break
            best=(cost,S)
            a,b=np.unravel_index(int(np.argmin(cost)),cost.shape)
            edge=(abs(S[a])>=S[np.argmax(np.abs(S))]-1e-9)
            if not edge: break
    cost,S=best
    picks=[]
    for flat in np.argsort(cost,axis=None):
        a,b=np.unravel_index(int(flat),cost.shape)
        if all(abs(S[a]-S[pa])>=1.5 for pa,_ in picks): picks.append((a,b))
        if len(picks)>=TOPK: break
    POSE=[]
    for a,b in picks:
        for ds in SF:
            s=S[a]+ds
            if not(-10<=s<=10): continue
            for dd_ in DF:
                d2=DC[b]+dd_
                if -2<=d2<=2: POSE.append((s,d2))
    P=np.empty((len(POSE)*len(QF),3))
    for t,(s,d2) in enumerate(POSE):
        P[t*len(QF):(t+1)*len(QF)]=bf+(T0[:3,3]+s*u+d2*v)
    df=tree.query(P,k=1,workers=W)[0]
    cf=np.minimum(df.reshape(len(POSE),-1),1.0).mean(1)
    t=int(np.argmin(cf)); s,d2=POSE[t]
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d2*v
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), nev+len(POSE)

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    for mode in ('fixed','widen'):
        globals()['MODE']=mode
        t0=time.perf_counter()
        with Pool(12) as p: res=p.map(run, range(len(keys)))
        e=np.array([r[0] for r in res]); nv=np.mean([r[1] for r in res])
        line='MODE=%-6s MPE=%.4f med=%.4f <=1m=%.0f%% 平均求值%.0f次  %.0fs墙钟'
        print(line % (mode,e.mean(),np.median(e),100*(e<=1).mean(),nv,time.perf_counter()-t0))
        for lo,hi in [(0,1),(1,3),(3,5),(5,8)]:
            m=(dep>=lo)&(dep<hi)&(~rep)
            if m.sum(): print('    dep %d-%d m: med=%.2f <=1m=%.0f%% n=%d'
                              % (lo,hi,np.median(e[m]),100*(e[m]<=1).mean(),int(m.sum())))
EOF
