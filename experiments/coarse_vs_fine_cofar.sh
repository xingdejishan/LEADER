#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, sys, time, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
sys.path.insert(0,'/home/zhangcy/projects/r-gate-20261018-v1')
import geometry_local as GL
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

def q(tree,P):
    d=tree.query(P,k=1,workers=W,distance_upper_bound=1.0)[0]
    return np.where(np.isfinite(d),d,1.0)

def run(i):
    c=v7[i][:3,3]
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    clouds=[]
    for j in near:
        r=refs[j]; T=np.asarray(r['world_from_body'])
        w=vox(decode(r['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Qv=decode(qmap[keys[i]]['scan_path']); Qv=Qv[(np.linalg.norm(Qv,axis=1)>1)&(np.linalg.norm(Qv,axis=1)<60)]
    Qv=vox(Qv,0.3)
    QC=Qv[:min(NC,len(Qv))]; QF=Qv[:min(4000,len(Qv))]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
    # d0
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
    a0,b0=picks[0]
    Pc=T0.copy(); Pc[:3,3]=T0[:3,3]+SC[a0]*u+DC[b0]*v
    POSE=[]
    for a,b in picks:
        for ds in SF:
            s=SC[a]+ds
            if not(-10<=s<=10): continue
            for dd_ in DF:
                d2=DC[b]+dd_
                if -2<=d2<=2: POSE.append((s,d2))
    P=np.empty((len(POSE)*len(QF),3))
    for t,(s,d2) in enumerate(POSE):
        P[t*len(QF):(t+1)*len(QF)]=bf+(T0[:3,3]+s*u+d2*v)
    cf=np.minimum(q(tree,P).reshape(len(POSE),-1),1.0).mean(1)
    t=int(np.argmin(cf)); s,d2=POSE[t]
    Pf=T0.copy(); Pf[:3,3]=T0[:3,3]+s*u+d2*v
    # CoFAR
    row=qmap[keys[i]]
    qraw=GL.raw_scan(row)
    t0=time.perf_counter()
    Cc,_=GL.refine_raw(Pc,qraw,refs,rpos); tc=time.perf_counter()-t0
    t0=time.perf_counter()
    Cf,_=GL.refine_raw(Pf,qraw,refs,rpos); tf=time.perf_counter()-t0
    e=lambda T: float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))
    return dict(i=i,dep=e(v7[i]),coarse=e(Pc),fine=e(Pf),
                coarse_cf=e(Cc),fine_cf=e(Cf),tc=tc,tf=tf,
                rot_c=float(np.degrees(np.arccos(np.clip((np.trace(Cc[:3,:3].T@gt[i][:3,:3])-1)/2,-1,1)))),
                rot_f=float(np.degrees(np.arccos(np.clip((np.trace(Cf[:3,:3].T@gt[i][:3,:3])-1)/2,-1,1)))))

if __name__=='__main__':
    t0=time.perf_counter()
    with Pool(12) as p: R=p.map(run, range(len(keys)))
    el=time.perf_counter()-t0
    for k in ('coarse','fine','coarse_cf','fine_cf'):
        a=np.array([r[k] for r in R])
        print('%-10s MPE=%.4f  中位=%.4f  进一米=%.0f%%' % (k,a.mean(),np.median(a),100*(a<=1).mean()))
    print('CoFAR 单帧耗时 中位 %.0f ms' % (np.median([r['tc'] for r in R])*1000))
    print('墙钟 %.0f s (12 procs)' % el)
    keep=[r for r in R if not r['i'] in set()]
    dep=np.array([r['dep'] for r in R])
    for lo,hi in [(0,1),(1,3),(3,5),(5,8)]:
        m=(dep>=lo)&(dep<hi)
        if m.sum():
            a=np.array([r['coarse_cf'] for r in R])[m]; b=np.array([r['fine_cf'] for r in R])[m]
            print('  dep %d-%d m (n=%d): 粗+CF %.2f / %.0f%%   粗+细+CF %.2f / %.0f%%'
                  % (lo,hi,m.sum(),np.median(a),100*(a<=1).mean(),np.median(b),100*(b<=1).mean()))
EOF
