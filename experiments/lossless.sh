#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from scipy.spatial import cKDTree
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
S=np.arange(-10,10.01,0.5); D=np.arange(-2,2.001,0.05)

def prep(i):
    k=keys[i]; c=v7[i][:3,3]
    m=json.load(open(R+k.replace('/','__')+'.json'))
    u=np.asarray(m['u'],float); u=u/np.linalg.norm(u)
    idx=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(pos[j]-c))[:3]
    maps=[]
    for j in idx:
        r=refs[j]; raw=vox(decode(r['scan_path']),0.3); T=np.asarray(r['world_from_body'])
        w=raw@T[:3,:3].T+T[:3,3]; maps.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(maps),0.3)
    Q=decode(qmap[k]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)[:4000]
    v=np.cross(u,[0,0,1.0]); v=v/np.linalg.norm(v)
    return M,Q,v7[i].copy(),u,v

def full_grid(i, workers):
    M,Q,T0,u,v=prep(i)
    tree=cKDTree(M); R3=T0[:3,:3]
    base=Q@R3.T
    offs=np.outer(D,v)                       # 81 x 3
    best=(1e18,0.0,0.0)
    for s in S:
        P=(base[None,:,:]+(T0[:3,3]+s*u+offs)[:,None,:]).reshape(-1,3)
        dd=tree.query(P,k=1,workers=workers)[0]
        cost=np.minimum(dd.reshape(len(D),-1),1.0).mean(1)
        b=int(np.argmin(cost))
        if cost[b]<best[0]: best=(float(cost[b]),float(s),float(D[b]))
    c,s,d=best
    T=T0.copy(); T[:3,3]=T0[:3,3]+s*u+d*v
    return (s,d,float(np.linalg.norm(T[:3,3]-gt[i][:3,3])))

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
sm=[]
for lo,hi in [(1,2),(3,5)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): sm.append(int(idx[len(idx)//2]))
print('frames:', [(i,round(float(dep[i]),2)) for i in sm])

res={}
for w in (1,4,8):
    t0=time.perf_counter(); out=[full_grid(i,w) for i in sm]
    per=(time.perf_counter()-t0)/len(sm); res[w]=out
    print('\nworkers=%-2d : %.0f ms/frame  (single process, 41x81 grid, 4000 pts)' % (w,per*1000))
    for i,(s,d,e) in zip(sm,out):
        print('   frame %4d  s=%+5.1f d=%+5.2f err=%.6f' % (i,s,d,e))

print('\n=== identical to workers=1? ===')
for w in (4,8):
    same=all((a==b) for a,b in zip(res[1],res[w]))
    print('  workers=%d : bit-identical=%s' % (w,same))
EOF
