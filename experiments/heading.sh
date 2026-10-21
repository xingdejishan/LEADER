#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
$PY - <<'EOF'
import json, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
R='/data2/zhangcy/r4-full-20261019-v1/results/main/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rdates=np.array([r['key'].split('/')[0] for r in refs])
rts=np.array([int(r['key'].split('/')[1]) for r in refs])
rtree=cKDTree(rpos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
SC=np.arange(-10,10.01,1.0); DC=np.arange(-2,2.001,0.5)
SF=np.arange(-0.5,0.501,0.25); DF=np.arange(-0.4,0.401,0.1)
W=4; NC=1500; TOPK=6
MODE='heading'; AXIS=0

def run(i):
    c=v7[i][:3,3]
    try:
        if MODE=='heading':
            u=v7[i][:3,:3][:,AXIS].copy()
        else:
            j=int(rtree.query(c)[1]); d=rdates[j]
            mm=(rdates==d)&(np.linalg.norm(rpos-c,axis=1)<40.0)
            P=rpos[mm]; T=rts[mm].astype(float); o=np.argsort(T); P=P[o]; T=T[o]
            t=(T-T.mean())/1e3
            A=np.column_stack([np.ones_like(t),t,t*t])
            coef,*_=np.linalg.lstsq(A,P,rcond=None)
            u=coef[1]
        u=u/np.linalg.norm(u)
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
        v=np.cross(u,[0,0,1.0]); v=v/np.linalg.norm(v)
        T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
        bc=QC@R3.T; bf=QF@R3.T
        off=np.outer(DC,v)
        P2=np.empty((len(SC)*len(DC)*len(QC),3))
        for a,s in enumerate(SC):
            P2[a*len(DC)*len(QC):(a+1)*len(DC)*len(QC)]=(bc[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
        dd=tree.query(P2,k=1,workers=W)[0]
        cost=np.minimum(dd.reshape(len(SC),len(DC),-1),1.0).mean(2)
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
        P3=np.empty((len(POSE)*len(QF),3))
        for t,(s,d2) in enumerate(POSE):
            P3[t*len(QF):(t+1)*len(QF)]=bf+(T0[:3,3]+s*u+d2*v)
        df=tree.query(P3,k=1,workers=W)[0]
        cf=np.minimum(df.reshape(len(POSE),-1),1.0).mean(1)
        t=int(np.argmin(cf)); s,d2=POSE[t]
        T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d2*v
        return float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))
    except Exception as ex:
        return float('nan')

if __name__=='__main__':
    for axis in (0,1,2):
        globals()['AXIS']=axis
        with Pool(12) as p: e=np.array(p.map(run, range(len(keys))))
        fin=np.isfinite(e)
        print('MODE=heading axis=%d: MPE=%.4f med=%.4f <=1m=%.0f%% (finite %d/%d)'
              % (axis, np.nanmean(e), np.nanmedian(e), 100*(e[fin]<=1).mean(), fin.sum(), len(e)))
EOF
