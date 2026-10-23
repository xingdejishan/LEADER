import json, torch, numpy as np, hashlib
from pathlib import Path

CFG = '/home/zhangcy/projects/leader-far-covolume-standard-20261001/config-v2.json'
c = json.load(open(CFG))
print('CONFIG KEYS:')
for k, v in c.items():
    print('  ', k, '=', str(v)[:260])

CK = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
h = hashlib.sha256(open(CK, 'rb').read()).hexdigest()
print('\nCKPT sha256 =', h)
st = torch.load(CK, map_location='cpu')
print('CKPT top keys =', list(st.keys()))
if 'model' in st:
    md = st['model']
    print('model entries =', len(md))
    for k in list(md.keys())[:12]:
        print('   ', k, tuple(md[k].shape), md[k].dtype)
for k, v in st.items():
    if k != 'model':
        print('  meta', k, '=', str(v)[:200])

print('\nINITIAL.NPZ:')
with np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=False) as a:
    for k in a.files:
        arr = a[k]
        print('  ', k, arr.shape, arr.dtype)
    keys = a['keys'] if 'keys' in a.files else None
    if keys is not None:
        print('  keys[:4] =', [str(x) for x in keys[:4]])
        print('  nkeys =', len(keys))
    if 'Poses' in a.files:
        print('  Poses[0] =\n', a['Poses'][0])
