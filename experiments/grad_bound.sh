#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
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
    s=np.fromfile(p,dtype=dt); return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
SC=np.arange(-10,10.01,1.0)
SF=np.arange(-0.5,0.501,0.25); DF=np.arange(-0.4,0.401,0.1)
D0=np.arange(-2,2.001,0.25)
NC=1500; TOPK=6; W=4
A=0.5; B=0.4      # 细窗口半宽（沿轨、横向）
MODE='grad'

def q(tree,P):
    d=tree.query(P,k=1,workers=W,distance_upper_bound=1.0)[0]
    return np.where(np.isfinite(d),d,1.0)

def run(i):
    c=v7[i][:3,3]
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    clouds=[]
    for j in near:
        T=np.asarray(refs[j]['world_from_body'])
        w=vox(decode(refs[j]['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    QC=Q[:min(NC,len(Q))]; QF=Q[:min(4000,len(Q))]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
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
    picks=[]
    for flat in np.argsort(cost,axis=None):
        a,b=np.unravel_index(int(flat),cost.shape)
        if all(abs(SC[a]-SC[pa])>=1.5 for pa,_ in picks): picks.append((a,b))
        if len(picks)>=TOPK: break
    best=(np.inf,None,None); used=0; nref=0
    for a,b in picks:
        sc=SC[a]; dc=DC[b]
        centre=bf+(T0[:3,3]+sc*u+dc*v)
        dd,ii=tree.query(centre,k=1,workers=W,distance_upper_bound=1.0)
        dd=np.where(np.isfinite(dd),dd,1.0); used+=1
        ccen=float(np.minimum(dd,1.0).mean())
        ok=np.isfinite(dd)&(dd>1e-9)&(dd<1.0)
        if MODE=='grad' and ok.any():
            g=centre[ok]-M[ii[ok]]; g/=np.maximum(dd[ok],1e-12)[:,None]
            slack=A*np.abs(g@u)+B*np.abs(g@v)
            lbv=np.minimum(np.maximum(dd[ok]-slack,0.0),1.0).mean()
            n_out=(~ok).sum()
            lb=(lbv*ok.sum()+1.0*n_out)/len(dd)
        else:
            lb=ccen-0.5*np.hypot(2*A,2*B)
        if lb>=best[0]:
            continue
        nref+=1
        POSE=[]
        for ds in SF:
            s=sc+ds
            if not(-10<=s<=10): continue
            for dd_ in DF:
                d2=dc+dd_
                if -2<=d2<=2: POSE.append((s,d2))
        P=np.empty((len(POSE)*len(QF),3))
        for k_,(s,d2) in enumerate(POSE):
            P[k_*len(QF):(k_+1)*len(QF)]=bf+(T0[:3,3]+s*u+d2*v)
        cf=np.minimum(q(tree,P).reshape(len(POSE),-1),1.0).mean(1); used+=len(POSE)
        t=int(np.argmin(cf))
        if cf[t]<best[0]: best=(float(cf[t]),POSE[t][0],POSE[t][1])
    s,d2=best[1],best[2]
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d2*v
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), used, nref

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    t0=time.perf_counter()
    with Pool(12) as p: R=p.map(run, range(len(keys)))
    e=np.array([r[0] for r in R]); nv=np.array([r[1] for r in R]); nr=np.array([r[2] for r in R])
    print('梯度界: MPE=%.4f med=%.4f <=1m=%.0f%%  细搜求值 均值%.0f  平均精搜区域 %.2f/6  %.0fs'
          % (e.mean(),np.median(e),100*(e<=1).mean(),nv.mean(),nr.mean(),time.perf_counter()-t0))
    for lo,hi in [(3,5),(5,8)]:
        m=(dep>=lo)&(dep<hi)&(~rep)
        if m.sum(): print('   dep %d-%d: med=%.2f <=1m=%.0f%%' % (lo,hi,np.median(e[m]),100*(e[m]<=1).mean()))
    print('对照 全量细搜: MPE=1.5693 270次 / 粗界: 269次')
EOF
