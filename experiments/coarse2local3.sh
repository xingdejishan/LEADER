#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
R='/data2/zhangcy/r4-full-20261019-v1/results/main/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
qmap={str(q['key']):q for q in qs}
pos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rtree=cKDTree(pos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
W=4; CFG={}
def run(i):
    c=v7[i][:3,3]; k=keys[i]
    m=json.load(open(R+k.replace('/','__')+'.json'))
    u=np.asarray(m['u'],float); u=u/np.linalg.norm(u)
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(pos[j]-c))[:3]
    clouds=[]
    for j in near:
        r=refs[j]; T=np.asarray(r['world_from_body'])
        w=vox(decode(r['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Q=decode(qmap[k]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    QC=Q[:min(CFG['NC'],len(Q))]; QF=Q[:min(4000,len(Q))]
    v=np.cross(u,[0,0,1.0]); v=v/np.linalg.norm(v)
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    bc=QC@R3.T; bf=QF@R3.T
    SC=np.arange(-10,10.0001,CFG['ds']); DC=np.arange(-2,2.0001,0.5)
    off=np.outer(DC,v)
    P=np.empty((len(SC)*len(DC)*len(QC),3))
    for a,s in enumerate(SC):
        P[a*len(DC)*len(QC):(a+1)*len(DC)*len(QC)]=(bc[None,:,:]+(T0[:3,3]+s*u+off)[:,None,:]).reshape(-1,3)
    dd=tree.query(P,k=1,workers=W)[0]
    cost=np.minimum(dd.reshape(len(SC),len(DC),-1),1.0).mean(2)
    picks=[]
    for flat in np.argsort(cost,axis=None):
        a,b=np.unravel_index(int(flat),cost.shape)
        if all(abs(SC[a]-SC[pa])>=CFG['ds']*1.5 for pa,_ in picks): picks.append((a,b))
        if len(picks)>=CFG['TOPK']: break
    SF=np.arange(-CFG['hw'],CFG['hw']+1e-9,0.25); DF=np.arange(-CFG['hd'],CFG['hd']+1e-9,0.1)
    POSE=[]
    for a,b in picks:
        for ds_ in SF:
            s=SC[a]+ds_
            if not(-10<=s<=10): continue
            for dd_ in DF:
                d=DC[b]+dd_
                if -2<=d<=2: POSE.append((s,d))
    P=np.empty((len(POSE)*len(QF),3))
    for t,(s,d) in enumerate(POSE):
        P[t*len(QF):(t+1)*len(QF)]=bf+(T0[:3,3]+s*u+d*v)
    df=tree.query(P,k=1,workers=W)[0]
    cf=np.minimum(df.reshape(len(POSE),-1),1.0).mean(1)
    t=int(np.argmin(cf)); s,d=POSE[t]
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d*v
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    sm=[]
    for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
        idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
        if len(idx): sm.extend(int(i) for i in idx[np.linspace(0,len(idx)-1,min(10,len(idx)),dtype=int)])
    cfgs=[dict(ds=1.0,TOPK=6,NC=1500,hw=0.75,hd=0.6),
          dict(ds=0.75,TOPK=6,NC=1500,hw=0.5,hd=0.4),
          dict(ds=1.0,TOPK=6,NC=2000,hw=0.75,hd=0.6)]
    for cfg in cfgs:
        CFG.update(cfg)
        t0=time.perf_counter(); out=[run(i) for i in sm]; per=(time.perf_counter()-t0)/len(sm)
        e1=np.array(out); dd=dep[sm]
        b=[]
        for lo,hi in [(1,3),(3,8)]:
            mk=(dd>=lo)&(dd<hi); b.append('%d-%d:%.0f%%'%(lo,hi,100*(e1[mk]<=1).mean()))
        with Pool(12) as p: res=p.map(run, range(len(keys)))
        e=np.array(res)
        print('ds=%.2f TOPK=%d NC=%d win=%.2f/%.2f | %4.0f ms | MPE=%.4f med=%.4f <=1m=%.0f%% | %s'
              % (cfg['ds'],cfg['TOPK'],cfg['NC'],cfg['hw'],cfg['hd'],per*1000,e.mean(),np.median(e),100*(e<=1).mean(),' '.join(b)))
EOF
