#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
cd /home/zhangcy/projects/r5b-lossless-20261019-v1
$PY - <<'EOF'
import json, sys, time, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
sys.path.insert(0,'/home/zhangcy/projects/r5b-lossless-20261019-v1')
import fast_lib as F
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
MS=None; MODE='anchor'
def init():
    global MS
    MS=F.MapSource(refs,rpos)

def run(i):
    c=v7[i][:3,3]; M=MS.local(c)
    q=F.decode(qmap[keys[i]]['scan_path'])
    q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]
    q=F.voxel(q,0.3)
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    lat=np.cross(u,[0,0,1.0]); lat/=np.linalg.norm(lat)
    tree=cKDTree(M); R3=v7[i][:3,:3]; o=v7[i][:3,3]
    qc=q[:1500]; qf=q[:4000]; bc=qc@R3.T; bf=qf@R3.T
    if MODE=='anchor':
        P=np.empty((len(F.D_ORIGIN)*len(qc),3))
        for a,dd in enumerate(F.D_ORIGIN): P[a*len(qc):(a+1)*len(qc)]=bc+(o+dd*lat)
        c0=np.minimum(F.query_dist(tree,P).reshape(len(F.D_ORIGIN),-1),1.0).mean(1)
        dh=float(F.D_ORIGIN[int(np.argmin(c0))])
        DC=np.clip(np.array([dh-0.5,dh,dh+0.5]),-2,2)
        GRID=[(s,float(dd)) for s in F.S_COARSE for dd in DC]
        n0=len(F.D_ORIGIN)
    else:
        step=0.5 if MODE=='per9' else 0.25
        DC=np.arange(-2,2.0001,step)
        GRID=[(s,float(dd)) for s in F.S_COARSE for dd in DC]
        n0=0
    P=np.empty((len(GRID)*len(qc),3))
    for a,(s,dd) in enumerate(GRID): P[a*len(qc):(a+1)*len(qc)]=bc+(o+s*u+dd*lat)
    cc=np.minimum(F.query_dist(tree,P).reshape(len(GRID),-1),1.0).mean(1)
    order=np.argsort(cc); picks=[]
    for t in order:
        s,dd=GRID[t]
        if all(abs(s-p[0])>=1.5 for p in picks): picks.append((s,dd))
        if len(picks)>=6: break
    best=(np.inf,0,0); nev=n0
    for s0,d0 in picks:
        cp=bf+(o+s0*u+d0*lat)
        di,ix=tree.query(cp,k=1,workers=4,distance_upper_bound=1.0); nev+=1
        di=np.where(np.isfinite(di),di,1.0)
        ins=np.isfinite(di)&(di>1e-9)&(di<1.0)
        if ins.any():
            g=cp[ins]-M[ix[ins]]; g/=np.maximum(di[ins],1e-12)[:,None]
            sl=0.5*np.abs(g@u)+0.4*np.abs(g@lat)
            bi=np.minimum(np.maximum(di[ins]-sl,0),1).mean()
            lb=(bi*ins.sum()+float((~ins).sum()))/len(di)
            if lb>=best[0]: continue
        poses=[(s0+ds,d0+d2) for ds in F.S_FINE for d2 in F.D_FINE if -10<=s0+ds<=10 and -2<=d0+d2<=2]
        P=np.empty((len(poses)*len(qf),3))
        for t,(s,d2) in enumerate(poses): P[t*len(qf):(t+1)*len(qf)]=bf+(o+s*u+d2*lat)
        cf=np.minimum(F.query_dist(tree,P).reshape(len(poses),-1),1.0).mean(1); nev+=len(poses)
        t2=int(np.argmin(cf))
        if cf[t2]<best[0]: best=(float(cf[t2]),poses[t2][0],poses[t2][1])
    s,d2=best[1],best[2]
    T=v7[i].copy(); T[:3,3]=o+s*u+d2*lat
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), nev

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    res={}
    for mode in ('anchor','per9','per17'):
        globals()['MODE']=mode
        t0=time.perf_counter()
        with Pool(12,initializer=init) as p: R=p.map(run, range(len(keys)))
        e=np.array([r[0] for r in R]); nv=np.array([r[1] for r in R])
        res[mode]=e
        print('MODE=%-7s 求值均值%5.0f  MPE=%.4f 中位=%.4f <=1m=%.0f%%  粗搜额外求值 %d  %.0fs'
              % (mode,nv.mean(),e.mean(),np.median(e),100*(e<=1).mean(),
                 0 if mode=='anchor' else (189 if mode=='per9' else 357),time.perf_counter()-t0))
        for lo,hi in [(1,3),(3,5),(5,8)]:
            m=(dep>=lo)&(dep<hi)&(~rep)
            if m.sum(): print('    dep %d-%d: med=%.2f <=1m=%.0f%%' % (lo,hi,np.median(e[m]),100*(e[m]<=1).mean()))
    a,b=res['anchor'],res['per17']
    print('\nanchor vs per17: 逐帧中位差 %.3f m，per17 更好 %d 帧、更差 %d 帧、相同 %d 帧'
          % (np.median(np.abs(a-b)),int((b<a-1e-9).sum()),int((b>a+1e-9).sum()),int(np.isclose(a,b).sum())))
EOF
