import glob, json, os, sys
import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
z = np.load(os.path.join(P, 'results/full-online.npz'), allow_pickle=True)
print('keys:', list(z.keys()))
for k in z.keys():
    a = z[k]
    print(' ', k, getattr(a, 'shape', None), a.dtype)
print('first keys:', [str(x) for x in z['keys'][:5]])
print('n frames:', len(z['keys']))
if 'poses' in z:
    print('poses shape', z['poses'].shape, z['poses'][0])

print('\nresults files:')
for p in sorted(glob.glob(os.path.join(P, 'results', '*online*'))):
    print(' ', p, os.path.getsize(p))
print('\nbackup dir:')
for p in sorted(glob.glob(os.path.join(P, '.backups/remove-rotation-20261022-182520', '**', '*.py'), recursive=True)):
    print(' ', p)
