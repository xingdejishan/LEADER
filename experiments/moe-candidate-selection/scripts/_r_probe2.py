import json, os
from pathlib import Path
import numpy as np
P = Path('/data2/zhangcy/sp-provider-cls5-sc2pos5-20261015-v1')
print('== data dir', P, P.is_dir())
if P.is_dir():
    for n in sorted(os.listdir(P)):
        p = P/n
        print('  ', n, p.stat().st_size, 'dir' if p.is_dir() else '')
Q = Path('/home/zhangcy/projects/sp-provider-cls5-sc2pos5-20261015-v1')
print('== proj dir', Q, Q.is_dir())
if Q.is_dir():
    for n in sorted(os.listdir(Q)):
        print('  ', n, (Q/n).stat().st_size)
z = np.load(P/'provider.npz', allow_pickle=False)
print('provider keys:', z.files)
for k in z.files:
    print('  ', k, z[k].shape, z[k].dtype)
print('first keys', [str(x) for x in z['keys'][:3]])
print('offsets[:5]', z['offsets'][:5])
print('indices[:12]', z['indices'][:12])
# reference CLS?
for cand in ('ref-cls.npy','library-cls.npy','reference-cls.npy'):
    print(cand, (P/cand).exists(), (Q/cand).exists())
