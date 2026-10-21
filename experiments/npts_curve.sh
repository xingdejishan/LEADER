#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, numpy as np
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

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
buckets=[(0.2,1.0),(1.0,2.0),(2.0,3.0),(3.0,5.0),(5.0,9.0)]
picks=[]
for lo,hi in buckets:
    idx=np.where((dep>=lo)&(dep<hi))[0]
    if len(idx): picks.append(int(idx[len(idx)//2]))
print('frames picked:', [(i, round(float(dep[i]),2)) for i in picks])

S=np.arange(-10,10.01,0.5)
def cost_curve(M, Q, T0, u, npts):
    P=Q[:npts]
    tree=cKDTree(M)
    out=[]
    for s in S:
        T=T0.copy(); T[:3,3]=T0[:3,3]+s*u
        W=P@T[:3,:3].T+T[:3,3]
        d=tree.query(W,k=1,workers=1)[0]
        out.append(float(np.minimum(d,1.0).mean()))
    return np.asarray(out)

print('\n%-6s %-8s %-9s %-9s %-9s %-9s %-9s' % ('frame','dep','argmin4k','argmin1k','argmin500','corr1k','corr500'))
for i in picks:
    k=keys[i]; c=v7[i][:3,3]
    idx=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(pos[j]-c))[:3]
    maps=[]
    for j in idx:
        r=refs[j]; raw=vox(decode(r['scan_path']),0.3); T=np.asarray(r['world_from_body'])
        w=raw@T[:3,:3].T+T[:3,3]; maps.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(maps),0.3)
    Q=decode(qmap[k]['scan_path'])
    Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    # along-track direction: truth tangent (diagnostic only)
    nxt=min(i+1,len(keys)-1); prv=max(i-1,0)
    u=gt[nxt][:3,3]-gt[prv][:3,3]; u=u/np.linalg.norm(u)
    T0=v7[i].copy()
    c4=cost_curve(M,Q,T0,u,4000)
    c1=cost_curve(M,Q,T0,u,1000)
    c5=cost_curve(M,Q,T0,u,500)
    a4,a1,a5=[float(S[np.argmin(c)]) for c in (c4,c1,c5)]
    r1=np.corrcoef(c4,c1)[0,1]; r5=np.corrcoef(c4,c5)[0,1]
    print('%-6d %-8.2f %-9.1f %-9.1f %-9.1f %-9.3f %-9.3f  npts=%d/%d/%d'
          % (i, dep[i], a4, a1, a5, r1, r5, min(4000,len(Q)), min(1000,len(Q)), min(500,len(Q))))
EOF
