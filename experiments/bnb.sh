#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, heapq, numpy as np, time
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rtree=cKDTree(rpos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]
NPTS=4000; W=4; MAXEVAL=1600
SMIN=0.125; DMIN=0.05

def run(i):
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
    T0=v7[i].copy(); R3=T0[:3,:3]; tree=cKDTree(M)
    base=Q@R3.T; c0=T0[:3,3]
    nev=[0]
    def probe(s,d):
        nev[0]+=1
        D=tree.query(base+(c0+s*u+d*v),k=1,workers=W)[0]
        return D                      # 每点最近距离
    def cost_of(D):  return float(np.minimum(D,1.0).mean())
    def lb_of(D,r):
        return float(np.minimum(np.maximum(D-r,0.0),1.0).mean())
    # BnB
    root=( -10.0,10.0,-2.0,2.0 )
    best=(np.inf,0.0,0.0)
    D=probe(0.0,0.0); cst=cost_of(D)
    if cst<best[0]: best=(cst,0.0,0.0)
    r=0.5*np.hypot(20.0,4.0)
    pq=[(lb_of(D,r),-10.0,10.0,-2.0,2.0,0.0,0.0,D,r)]
    while pq and nev[0]<MAXEVAL:
        lb,slo,shi,dlo,dhi,sc,dc,Dc,rc=heapq.heappop(pq)
        if lb>=best[0]-1e-12: break          # 队列按下界排序，队首都不行则全体不行
        es,ed=shi-slo,dhi-dlo
        if es<=SMIN and ed<=DMIN:
            continue
        # 沿归一化更长的轴二分
        if es/20.0>=ed/4.0:
            mid=0.5*(slo+shi)
            kids=[(slo,mid),(mid,shi)]
            for a,b in kids:
                cs=0.5*(a+b)
                Dk=probe(cs,dc); ck=cost_of(Dk)
                if ck<best[0]: best=(ck,cs,dc)
                rk=0.5*np.hypot(b-a,ed)
                lk=lb_of(Dk,rk)
                if lk<best[0]-1e-12: heapq.heappush(pq,(lk,a,b,dlo,dhi,cs,dc,Dk,rk))
        else:
            mid=0.5*(dlo+dhi)
            for a,b in [(dlo,mid),(mid,dhi)]:
                dc2=0.5*(a+b)
                Dk=probe(sc,dc2); ck=cost_of(Dk)
                if ck<best[0]: best=(ck,sc,dc2)
                rk=0.5*np.hypot(es,b-a)
                lk=lb_of(Dk,rk)
                if lk<best[0]-1e-12: heapq.heappush(pq,(lk,slo,shi,a,b,sc,dc2,Dk,rk))
    cst,s,d=best
    T=T0.copy(); T[:3,3]=c0+s*u+d*v
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), nev[0]

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    t0=time.perf_counter()
    with Pool(12) as p: res=p.map(run, range(len(keys)))
    e=np.array([r[0] for r in res]); nv=np.array([r[1] for r in res])
    print('BnB  MPE=%.4f med=%.4f <=1m=%.0f%%  求值 中位%.0f 均值%.0f p90%.0f  %.0fs墙钟'
          % (e.mean(),np.median(e),100*(e<=1).mean(),np.median(nv),nv.mean(),np.percentile(nv,90),time.perf_counter()-t0))
    for lo,hi in [(0,1),(1,3),(3,5),(5,8)]:
        m=(dep>=lo)&(dep<hi)&(~rep)
        if m.sum(): print('   dep %d-%d m: med=%.2f <=1m=%.0f%% n=%d' % (lo,hi,np.median(e[m]),100*(e[m]<=1).mean(),int(m.sum())))
    print('对照 固定网格: MPE=1.5612 429次(1500/4000点) ; 穷举: 1.5403 3321次')
EOF
