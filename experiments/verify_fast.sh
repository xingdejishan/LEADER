#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
cd /home/zhangcy/projects/r5b-lossless-20261019-v1
$PY - <<'EOF'
import json, sys, time, numpy as np
sys.path.insert(0,'/home/zhangcy/projects/r5b-lossless-20261019-v1')
import fast_lib as F
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rrt=cKDTree(rpos)

# ---------- (a) 体素降采样等价性 ----------
print('=== 体素降采样：打包键 vs 按行去重 ===')
ok=0
for i in range(0,30,3):
    raw=F.decode(qmap[keys[i]]['scan_path'])
    raw=raw[(np.linalg.norm(raw,axis=1)>1)&(np.linalg.norm(raw,axis=1)<60)]
    a=F.voxel(raw,0.3,packed=True); b=F.voxel(raw,0.3,packed=False)
    ok+= int(a.shape==b.shape and np.array_equal(a,b))
print('  逐位相同 %d/10' % ok)

# ---------- (b) 全量精度 ----------
MS=None
def init():
    global MS
    MS=F.MapSource(refs,rpos)
def run(i):
    c=v7[i][:3,3]
    ca=MS.nearby(c); M=MS.local(c,ca)
    q=F.decode(qmap[keys[i]]['scan_path'])
    q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]
    q=F.voxel(q,0.3)
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    s,d,cost,ev=F.solve(M,q,v7[i],u)
    T=v7[i].copy(); T[:3,3]=T[:3,3]+s*u+d*np.cross(u,[0,0,1.0])/np.linalg.norm(np.cross(u,[0,0,1.0]))
    return float(np.linalg.norm(T[:3,3]-gt[i][:3,3])), ev

if __name__=='__main__':
    dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
    t0=time.perf_counter()
    with Pool(12,initializer=init) as p: R=p.map(run, range(len(keys)))
    el=time.perf_counter()-t0
    e=np.array([r[0] for r in R]); ev=np.array([r[1] for r in R])
    print('\n=== 全量 1502（新实现） ===')
    print('MPE=%.4f  中位=%.4f  进一米=%.0f%%  细搜求值 均值%.0f  %.0fs'
          % (e.mean(),np.median(e),100*(e<=1).mean(),ev.mean(),el))
    for lo,hi in [(0,1),(1,3),(3,5),(5,8)]:
        m=(dep>=lo)&(dep<hi)&(~rep)
        if m.sum(): print('   dep %d-%d m: med=%.2f <=1m=%.0f%%' % (lo,hi,np.median(e[m]),100*(e[m]<=1).mean()))
    print('对照（旧实现，无缓存无剪枝）: MPE=1.5693 中位=0.3528 进一米=83% 求值270')
EOF
