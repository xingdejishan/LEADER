#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
rtree=cKDTree(rpos)
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
def decode(p):
    s=np.fromfile(p,dtype=dt); return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100
def vox(p,s):
    k=np.floor(p/s).astype(np.int64); _,i=np.unique(k,axis=0,return_index=True); return p[np.sort(i)]

# 连续段：取一个日期里按时间连续的 300 帧
idx_by_date={}
for i,k in enumerate(keys):
    idx_by_date.setdefault(k.split('/')[0],[]).append((int(k.split('/')[1]),i))
d=sorted(idx_by_date)[0]
seq=[i for _,i in sorted(idx_by_date[d])][:300]
print('连续段: %s  %d 帧' % (d,len(seq)))

REF=OrderedDict(); TRI=OrderedDict()
def ref_cloud(j):
    if j in REF: return REF[j]
    T=np.asarray(refs[j]['world_from_body'])
    w=vox(decode(refs[j]['scan_path']),0.3)@T[:3,:3].T+T[:3,3]
    REF[j]=w
    if len(REF)>600: REF.popitem(last=False)
    return w

t_map=[]; t_q=[]; hit=[0,0]; npts=[]
for i in seq:
    c=v7[i][:3,3]
    near=tuple(sorted(rtree.query_ball_point(c,30.0), key=lambda j: np.linalg.norm(rpos[j]-c))[:3])
    t0=time.perf_counter()
    if near in TRI:
        hit[0]+=1
        full=TRI[near]
    else:
        hit[1]+=1
        full=vox(np.concatenate([ref_cloud(j) for j in near]),0.3)   # 合并+体素只做一次，与位置无关
        TRI[near]=full
        if len(TRI)>400: TRI.popitem(last=False)
    M=full[np.linalg.norm(full-c,axis=1)<=60]                        # 每帧只做半径筛选
    t_map.append(time.perf_counter()-t0); npts.append(len(M))
    t0=time.perf_counter()
    Q=decode(qmap[keys[i]]['scan_path']); Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
    Q=vox(Q,0.3)
    t_q.append(time.perf_counter()-t0)
t_map=np.array(t_map)*1000; t_q=np.array(t_q)*1000
print('三元组缓存命中 %d/%d = %.0f%%' % (hit[0],sum(hit),100*hit[0]/sum(hit)))
print('地图（取缓存+半径筛选） 中位 %5.1f ms  均值 %5.1f ms  p90 %5.1f ms' % (np.median(t_map),t_map.mean(),np.percentile(t_map,90)))
print('查询扫描（解码+体素）   中位 %5.1f ms  均值 %5.1f ms  p90 %5.1f ms' % (np.median(t_q),t_q.mean(),np.percentile(t_q,90)))
print('地图点数 中位 %d' % np.median(npts))
EOF
