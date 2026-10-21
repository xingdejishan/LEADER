#!/bin/bash
PY=/home/zhangcy/.conda/envs/leader/bin/python
echo "load: $(uptime | sed 's/.*load average : //' 2>/dev/null || uptime | sed 's/.*load average: //')"
$PY - <<'EOF'
import json, time, numpy as np
H='/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel=json.load(open(H+'selection.json')); qs=sel['queries']
z=np.load(H+'scored.npz', allow_pickle=True)
keys=[str(k) for k in z['keys']]
qmap={str(q['key']):q for q in qs}
dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])

def dec_old(p):
    s=np.fromfile(p,dtype=dt)
    return np.column_stack([s[n] for n in ['x','y','z']]).astype(np.float64)*.005-100

def dec_new(p):
    s=np.fromfile(p,dtype=dt)
    out=np.empty((len(s),3),np.float64)
    for c,n in enumerate(('x','y','z')):
        np.multiply(s[n],0.005,out=out[:,c])
        out[:,c]-=100.0
    return out

def vox_old(p,s=0.3):
    k=np.floor(p/s).astype(np.int64)
    _,i=np.unique(k,axis=0,return_index=True)
    return p[np.sort(i)]

def vox_pack(p,s=0.3):
    k=np.floor(p/s).astype(np.int64)
    k-=k.min(0)
    A=int(k.max())+2
    key=(k[:,0]*A+k[:,1])*A+k[:,2]
    _,i=np.unique(key,return_index=True)
    return p[np.sort(i)]

sample=[keys[i] for i in range(0,60,3)]
print('%-28s %10s' % ('步骤','中位 ms'))
for name,f in (('解码（原）',dec_old),('解码（改）',dec_new)):
    ts=[]
    for k in sample:
        t0=time.perf_counter(); q=f(qmap[k]['scan_path']); ts.append(time.perf_counter()-t0)
    print('%-28s %10.1f' % (name,np.median(ts)*1000))
Q=dec_new(qmap[sample[0]]['scan_path'])
Q=Q[(np.linalg.norm(Q,axis=1)>1)&(np.linalg.norm(Q,axis=1)<60)]
for name,f in (('体素（原，axis=0）',vox_old),('体素（改，打包键）',vox_pack)):
    f(Q); ts=[]
    for _ in range(5):
        t0=time.perf_counter(); f(Q); ts.append(time.perf_counter()-t0)
    print('%-28s %10.1f   点数 %d -> %d' % (name,np.median(ts)*1000,len(Q),len(f(Q))))
print()
for tag,df,vf in (('原链路',dec_old,vox_old),('改后',dec_new,vox_pack)):
    ts=[]
    for k in sample:
        t0=time.perf_counter()
        q=df(qmap[k]['scan_path']); q=q[(np.linalg.norm(q,axis=1)>1)&(np.linalg.norm(q,axis=1)<60)]; q=vf(q)
        ts.append(time.perf_counter()-t0)
    a=np.array(ts)*1000
    print('%-8s 解码+过滤+体素 中位 %5.1f ms  均值 %5.1f ms' % (tag,np.median(a),a.mean()))
EOF
