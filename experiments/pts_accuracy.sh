#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from scipy.spatial import cKDTree
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']
qmap={str(q['key']):q for q in qs}
pos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rtree=cKDTree(pos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1); rep=z['replaced'].astype(bool)
cand=[]
for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): cand.extend(int(i) for i in idx[np.linspace(0,len(idx)-1,min(7,len(idx)),dtype=int)])
S=np.arange(-10,10.01,0.5); D=np.arange(-2,2.001,0.1)

def run_frame_once(i, npts):
    k=keys[i]; c=v7[i][:3,3]
    idx=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(pos[j]-c))[:3]
    maps=[]
    for j in idx:
        r=refs[j]; raw=vox(decode(r['scan_path']),0.3); T=np.asarray(r['world_from_body'])
        w=raw@T[:3,:3].T+T[:3,3]; maps.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(maps),0.3)
    Q=decode(qmap[k]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)[:npts]
    nxt=min(i+1,len(keys)-1); prv=max(i-1,0)
    u=gt[nxt][:3,3]-gt[prv][:3,3]; u=u/np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v=v/np.linalg.norm(v)
    T0=v7[i].copy(); tree=cKDTree(M)
    def cost(s,d):
        T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d*v
        W=Q@T[:3,:3].T+T[:3,3]
        return float(np.minimum(tree.query(W,k=1,workers=1)[0],1.0).mean())
    d0=min(((cost(0.0,d),d) for d in D))[1]
    best=None
    for s in S:
        cst=cost(s,d0)
        if best is None or cst<best[0]: best=(cst,s)
    s=best[1]
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d0*v
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))

dd=dep[cand]
for npts in (4000,2000,1000,500):
    t0=time.perf_counter(); errs=np.array([run_frame_once(i,npts) for i in cand])
    sec=(time.perf_counter()-t0)/len(cand)
    print('\n### npts=%-4d  %.0f ms/frame (41 evals, single-thread)' % (npts, sec*1000))
    print('   all    : med=%5.2f  <=1m=%3.0f%%   (deploy med=%5.2f)'
          % (np.median(errs), 100*(errs<=1).mean(), np.median(dd)))
    for lo,hi in [(0,1),(1,3),(3,8)]:
        m=(dd>=lo)&(dd<hi)
        if m.sum():
            print('   %d-%d m : med=%5.2f  <=1m=%3.0f%%  n=%d'
                  % (lo,hi,np.median(errs[m]),100*(errs[m]<=1).mean(),int(m.sum())))
EOF
