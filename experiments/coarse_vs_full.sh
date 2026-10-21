#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, os, time, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
R='/data2/zhangcy/r4-full-20261019-v1/results/main/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']
qmap={str(q['key']):q for q in qs}
pos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1); rep=z['replaced'].astype(bool)
cand=[]
for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): cand.extend(int(i) for i in idx[np.linspace(0,len(idx)-1,min(8,len(idx)),dtype=int)])
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
S=np.arange(-10,10.01,0.5); DS=np.arange(-2,2.001,0.5)
rtree=cKDTree(pos)

def work(i):
    k=keys[i]; c=v7[i][:3,3]
    idx=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(pos[j]-c))[:3]
    maps=[]
    for j in idx:
        r=refs[j]; raw=vox(decode(r['scan_path']),0.3); T=np.asarray(r['world_from_body'])
        w=raw@T[:3,:3].T+T[:3,3]; maps.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(maps),0.3)
    Q=decode(qmap[k]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)[:4000]
    nxt=min(i+1,len(keys)-1); prv=max(i-1,0)
    u=gt[nxt][:3,3]-gt[prv][:3,3]; u=u/np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v=v/np.linalg.norm(v)
    T0=v7[i].copy(); tree=cKDTree(M)
    def cost(s,d):
        T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d*v
        W=Q@T[:3,:3].T+T[:3,3]
        return float(np.minimum(tree.query(W,k=1,workers=1)[0],1.0).mean())
    best=(1e9,0,0)
    for s in S:
        cs=[cost(s,d) for d in DS]; kk=int(np.argmin(cs)); bc,pd=cs[kk],DS[kk]
        for d in (pd-0.25,pd-0.1,pd+0.1,pd+0.25):
            if -2<=d<=2:
                cc=cost(s,d)
                if cc<bc: bc,pd=cc,d
        if bc<best[0]: best=(bc,s,pd)
    _,s,d=best
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d*v
    e=float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))
    # stored full-grid result
    f=R+keys[i].replace('/','__')+'.json'
    e_full=float(json.load(open(f))['sel_err']) if os.path.exists(f) else np.nan
    return (i, float(dep[i]), e, e_full)

if __name__=='__main__':
    t0=time.perf_counter()
    nproc=8
    with Pool(nproc) as p: rows=p.map(work, cand)
    el=(time.perf_counter()-t0)*nproc/len(cand)
    print('frames=%d  approx single-thread %.2f s/frame' % (len(cand), el))
    d_=np.array([r[1] for r in rows]); e_=np.array([r[2] for r in rows]); f_=np.array([r[3] for r in rows])
    print('\n%-7s %8s %8s %9s %9s' % ('dep','coarse','fullgrid','diff','same_ok'))
    for i,d,e,f in rows:
        print('%-7.2f %8.2f %8.2f %9.2f %9s' % (d,e,f,e-f,'yes' if (e<=1)==(f<=1) else 'NO'))
    print('\ncoarse: med=%.2f  <=1m=%.0f%%' % (np.median(e_),100*(e_<=1).mean()))
    print('full  : med=%.2f  <=1m=%.0f%%' % (np.nanmedian(f_),100*(f_<=1).mean()))
    same=np.sum((e_<=1)==(f_<=1))
    print('same success/fail decision: %d/%d   median |diff|=%.3f m' % (same,len(rows),np.median(np.abs(e_-f_))))
    for lo,hi in [(0,1),(1,3),(3,8)]:
        m=(d_>=lo)&(d_<hi)
        if m.sum(): print('  dep %d-%d: coarse<=1m=%.0f%%  full<=1m=%.0f%%  n=%d'
                          % (lo,hi,100*(e_[m]<=1).mean(),100*(f_[m]<=1).mean(),int(m.sum())))
EOF
