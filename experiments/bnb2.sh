#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, heapq, numpy as np
from scipy.spatial import cKDTree
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
NPTS=4000; W=4

def ctx(i):
    c=v7[i][:3,3]
    near=sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3]
    clouds=[]
    for j in near:
        r=refs[j]; T=np.asarray(r['world_from_body'])
        w=vox(decode(r['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
        clouds.append(w[np.linalg.norm(w-c,axis=1)<=60])
    M=vox(np.concatenate(clouds),0.3)
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)[:NPTS]
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    v=np.cross(u,[0,0,1.0]); v/=np.linalg.norm(v)
    T0=v7[i].copy()
    return cKDTree(M), Q@T0[:3,:3].T, T0[:3,3], u, v

def bnb(tree,base,c0,u,v,SMIN,DMIN,MAXEVAL=4000):
    nev=[0]
    def probe(s,d):
        nev[0]+=1
        return tree.query(base+(c0+s*u+d*v),k=1,workers=W)[0]
    def cost(D): return float(np.minimum(D,1.0).mean())
    def lb(D,r): return float(np.minimum(np.maximum(D-r,0.0),1.0).mean())
    D0=probe(0.0,0.0); best=[cost(D0),0.0,0.0]
    pq=[(lb(D0,0.5*np.hypot(20.0,4.0)),-10.0,10.0,-2.0,2.0)]
    while pq and nev[0]<MAXEVAL:
        L,slo,shi,dlo,dhi=heapq.heappop(pq)
        if L>=best[0]-1e-12: break
        es,ed=shi-slo,dhi-dlo
        if es<=SMIN and ed<=DMIN: continue
        sc=0.5*(slo+shi); dc=0.5*(dlo+dhi)
        if es/20.0>=ed/4.0:
            mid=0.5*(slo+shi)
            kids=[(slo,mid,dlo,dhi),(mid,shi,dlo,dhi)]
        else:
            mid=0.5*(dlo+dhi)
            kids=[(slo,shi,dlo,mid),(slo,shi,mid,dhi)]
        for a,b,dl,dh in kids:
            cs=0.5*(a+b); ds=0.5*(dl+dh)
            Dk=probe(cs,ds); ck=cost(Dk)
            if ck<best[0]-1e-15: best=[ck,cs,ds]
            r=0.5*np.hypot(b-a,dh-dl); Lk=lb(Dk,r)
            if Lk<best[0]-1e-12: heapq.heappush(pq,(Lk,a,b,dl,dh))
    return best,nev[0]

dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
sm=[]
for lo,hi in [(0.5,1),(1,2),(2,3),(3,5),(5,8)]:
    idx=np.where((dep>=lo)&(dep<hi)&(~rep))[0]
    if len(idx): sm.append(int(idx[len(idx)//2]))
print('frames:',[(i,round(float(dep[i]),2)) for i in sm])
for SMIN,DMIN in ((0.5,0.1),(1.0,0.25),(2.0,0.5)):
    tot=0; out=[]
    for i in sm:
        tr,base,c0,u,v=ctx(i)
        (c,s,d),n=bnb(tr,base,c0,u,v,SMIN,DMIN)
        T=v7[i].copy(); T[:3,3]=c0+s*u+d*v
        out.append(float(np.linalg.norm(T[:3,3]-gt[i][:3,3]))); tot+=n
    print('SMIN=%.2f DMIN=%.2f: 平均求值 %.0f  误差 %s' % (SMIN,DMIN,tot/len(sm),['%.2f'%e for e in out]))
EOF
