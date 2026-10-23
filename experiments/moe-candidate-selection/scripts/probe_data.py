import json, os, glob
import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
cat = json.load(open(os.path.join(DATA, 'catalog.json')))
print('catalog type', type(cat), list(cat)[:10] if isinstance(cat, dict) else len(cat))
s = json.dumps(cat, default=str)[:1500]
print(s)

for sub in ('completed', 'selection', 'initial'):
    p = os.path.join(DATA, sub)
    print('\n', sub, os.path.isdir(p), os.listdir(p)[:10] if os.path.isdir(p) else '')

print('\nNCLT dates:', os.listdir(os.path.join(DATA, 'NCLT'))[:10])
d0 = os.path.join(DATA, 'NCLT', '2012-02-12')
print(os.listdir(d0)[:20])
sub = os.path.join(d0, 'velodyne')
if os.path.isdir(sub):
    fs = sorted(os.listdir(sub))
    print('velodyne n=', len(fs), fs[:3])

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
refs = json.load(open(os.path.join(P, 'inputs/refs.json')))
print('\nrefs n=', len(refs), 'keys:', list(refs[0].keys()))
print('ref0', refs[0]['key'], refs[0]['scan_path'], np.asarray(refs[0]['world_from_body'])[:3, 3])
frames = json.load(open(os.path.join(P, 'inputs/frames.json')))
print('\nframes n=', len(frames), 'keys:', list(frames[0].keys()))
print('frame0', frames[0])
