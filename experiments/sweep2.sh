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
MS=None; CFG={}
def init():
    global MS
    MS=F.MapSource(refs,rpos)

def run(i):
    cfg=CFG
    c=v7[i][:3,3]; M=MS.local(c)
    q=F.decode(qmap[keys[i]]['scan_path'])
    q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]
    q=F.voxel(q,0.3)
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    lat=np.cross(u,[0,0,1.0]); lat/=np.linalg.norm(lat)
    tree=cKDTree(M); R3=v7[i][:3,:3]; o=v7[i][:3,3]
    qc=q[:cfg['cpts']]; qf=q[:4000]; bc=qc@R3.T; bf=qf@R3.T
    DO=np.arange(-cfg['lrange'],cfg['lrange']+1e-9,cfg['lstep'])
    P=np.empty((len(DO)*len(qc),3))
    for a,dd in enumerate(DO): P[a*len(qc):(a+1)*len(qc)]=bc+(o+dd*lat)
    c0=np.minimum(F.query_dist(tree,P).reshape(len(DO),-1),1.0).mean(1)
    dh=float(DO[int(np.argmin(c0))])
    k=cfg['lcount']//2
    DC=np.clip(dh+np.array([(i-k)*cfg['lsp'] for i in range(cfg['lcount'])]),-2,2)
    SC=np.arange(-cfg['srange'],cfg['srange']+1e-9,cfg['sstep'])
    GRID=[(s,float(dd)) for s in SC for dd in DC]
    P=np.empty((len(GRID)*len(qc),3))
    for a,(s,dd) in enumerate(GRID): P[a*len(qc):(a+1)*len(qc)]=bc+(o+s*u+dd*lat)
    cc=np.minimum(F.query_dist(tree,P).reshape(len(GRID),-1),1.0).mean(1)
    picks=[]
    for t in np.argsort(cc):
        s,dd=GRID[t]
        if all(abs(s-p[0])>=max(1.5,cfg['sstep']) for p in picks): picks.append((s,dd))
        if len(picks)>=6: break
    SF=np.arange(-cfg['sfine'],cfg['sfine']+1e-9,0.25)
    best=(np.inf,0,0); nev=len(DO)
    for s0,d0 in picks:
        cp=bf+(o+s0*u+d0*lat)
        di,ix=tree.query(cp,k=1,workers=4,distance_upper_bound=1.0); nev+=1
        di=np.where(np.isfinite(di),di,1.0)
        ins=np.isfinite(di)&(di>1e-9)&(di<1.0)
        if ins.any():
            g=cp[ins]-M[ix[ins]]; g/=np.maximum(di[ins],1e-12)[:,None]
            sl=cfg['sfine']*np.abs(g@u)+0.4*np.abs(g@lat)
            bi=np.minimum(np.maximum(di[ins]-sl,0),1).mean()
            lb=(bi*ins.sum()+float((~ins).sum()))/len(di)
            if lb>=best[0]: continue
        poses=[(s0+ds,d0+d2) for ds in SF for d2 in F.D_FINE if -10<=s0+ds<=10 and -2<=d0+d2<=2]
        P=np.empty((len(poses)*len(qf),3))
        for t,(s,d2) in enumerate(poses): P[t*len(qf):(t+1)*len(qf)]=bf+(o+s*u+d2*lat)
        cf=np.minimum(F.query_dist(tree,P).reshape(len(poses),-1),1.0).mean(1); nev+=len(poses)
        t2=int(np.argmin(cf))
        if cf[t2]<best[0]: best=(float(cf[t2]),poses[t2][0],poses[t2][1])
    s,d2=best[1],best[2]
    T=v7[i].copy(); T[:3,3]=o+s*u+d2*lat
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), nev

BASE=dict(lrange=2.0,lstep=0.25,cpts=1500,lcount=3,lsp=0.5,srange=10.0,sstep=1.0,sfine=0.5)
def cfg(**kw):
    c=dict(BASE); c.update(kw); return c
TESTS=[
 ('基准',            cfg()),
 ('横向步长 0.5',     cfg(lstep=0.5)),
 ('横向步长 1.0',     cfg(lstep=1.0)),
 ('横向步长 1.5',     cfg(lstep=1.5)),
 ('横向步长 2.0',     cfg(lstep=2.0)),
 ('沿轨步长 2.0/窗1.0',cfg(sstep=2.0,sfine=1.0)),
 ('沿轨步长 2.5/窗1.25',cfg(sstep=2.5,sfine=1.25)),
 ('沿轨步长 3.0/窗1.5',cfg(sstep=3.0,sfine=1.5)),
 ('沿轨步长 4.0/窗2.0',cfg(sstep=4.0,sfine=2.0)),
]
if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    for name,c in TESTS:
        CFG.clear(); CFG.update(c)
        t0=time.perf_counter()
        with Pool(16,initializer=init) as p: R=p.map(run, range(len(keys)))
        e=np.array([r[0] for r in R]); nv=np.mean([r[1] for r in R])
        b=[]
        for lo,hi in [(1,3),(3,5),(5,8)]:
            m=(dep>=lo)&(dep<hi)&(~rep)
            if m.sum(): b.append('%d-%d:%.0f%%'%(lo,hi,100*(e[m]<=1).mean()))
        print('%-22s MPE=%.4f med=%.4f <=1m=%.0f%%  求值%4.0f  %.0fs  %s'
              % (name,e.mean(),np.median(e),100*(e<=1).mean(),nv,time.perf_counter()-t0,' '.join(b)), flush=True)
EOF
