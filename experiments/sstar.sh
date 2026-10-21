#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average : //' 2>/dev/null || uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
R='/data2/zhangcy/r4-full-20261019-v1/results/main/'
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]; v7=z['refined_V7_poses']; gt=z['reference']; rep=z['replaced'].astype(bool)
dep=np.linalg.norm(v7[:,:3,3]-gt[:,:3,3],axis=1)
ss=[]
for k in keys:
    ss.append(abs(json.load(open(R+k.replace('/','__')+'.json'))['s_star']))
ss=np.array(ss)
print('=== |s*| 分布（按部署误差分档，剔除可疑窗）===')
for lo,hi in [(0,1),(1,2),(2,3),(3,5),(5,8),(8,99)]:
    m=(dep>=lo)&(dep<hi)&(~rep)
    if m.sum()==0: continue
    print('  dep %4.1f-%4.1f m  n=%4d   |s*| 中位 %5.2f  <=2m %3.0f%%  <=5m %3.0f%%'
          % (lo,hi,m.sum(),np.median(ss[m]),100*(ss[m]<=2).mean(),100*(ss[m]<=5).mean()))
EOF
