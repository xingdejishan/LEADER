#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "waiting for low load..."
for k in $(seq 1 20); do
  L=$(uptime | sed 's/.*load average: //' | cut -d, -f1 | tr -d ' ')
  echo "  load=$L"
  ok=$(awk -v l="$L" 'BEGIN{print (l<45)?1:0}')
  [ "$ok" = "1" ] && break
  sleep 25
done
echo "start at load=$L"
cd /home/zhangcy/projects/r5b-lossless-20261019-v1
$PY - <<'EOF'
import json, sys, time, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/home/zhangcy/projects/r5b-lossless-20261019-v1')
import fast_lib as F
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); refs=sel['references']; qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']
qmap={str(q['key']):q for q in qs}
rpos=np.asarray([r['world_from_body'] for r in refs])[:,:3,3]
per={}
for i,k in enumerate(keys): per.setdefault(k.split('/')[0],[]).append((int(k.split('/')[1]),i))
d=sorted(per)[0]; seq=[i for _,i in sorted(per[d])][:260]
MS=F.MapSource(refs,rpos)
lat=[]; ev=[]; ph=[]; ne=[]
for n,i in enumerate(seq):
    t0=time.perf_counter()
    c=v7[i][:3,3]; ca=MS.nearby(c); M=MS.local(c,ca)
    q=F.decode(qmap[keys[i]]['scan_path'])
    q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]
    q=F.voxel(q,0.3)
    t1=time.perf_counter()
    u=v7[i][:3,:3][:,0].copy(); u/=np.linalg.norm(u)
    s,dd,cost,nev=F.solve(M,q,v7[i],u)
    t2=time.perf_counter()
    if n>=40:                      # 丢弃缓存冷启动
        lat.append(t2-t0); ph.append(t1-t0); ev.append(t2-t1); ne.append(nev)
lat=np.array(lat)*1000; ph=np.array(ph)*1000; ev=np.array(ev)*1000
print('\n地图缓存命中 %d/%d = %.0f%%' % (MS.hit,MS.hit+MS.miss,100*MS.hit/max(1,MS.hit+MS.miss)))
print('预热后 %d 帧:' % len(lat))
print('  预处理(地图+扫描)  中位 %5.1f ms' % np.median(ph))
print('  搜索(粗+细+剪枝)   中位 %5.1f ms' % np.median(ev))
print('  端到端             中位 %5.1f ms  均值 %5.1f ms  p90 %5.1f ms' % (np.median(lat),lat.mean(),np.percentile(lat,90)))
print('  细搜求值 均值 %.0f' % np.mean(ne))
EOF
