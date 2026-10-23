import json, os, sys, time
import numpy as np
ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
print('--- results dir')
for f in sorted(os.listdir(os.path.join(ROOT, 'results'))):
    p = os.path.join(ROOT, 'results', f)
    print('%-55s %10d' % (f, os.path.getsize(p)))
z = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
print('npz keys', z.files)
print('poses', z['poses'].shape, 'keys', len(z['keys']), z['keys'][:2])
frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
print('frames', len(frames), 'refs', len(refs))
print('frame0 fields', sorted(frames[0].keys()))
print('ref0 fields', sorted(refs[0].keys()))
init = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
print('init keys', init.files, init['poses'].shape, init['keys'][:2])
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
import fast_lib as F
print('TRUNC', F.TRUNC)
for j in (0, 1, 2):
    t = time.time()
    c = F.decode(refs[j]['scan_path'])
    print('ref', j, c.shape, '%.3f s' % (time.time() - t))
rec = frames[0]
t = time.time()
p = F.decode(rec['scan_path'])
print('scan0', p.shape, '%.3f s' % (time.time() - t))
print('mem:')
os.system("free -g; nproc")
d = os.path.join(ROOT, 'results/three_route_merged_diag.jsonl')
if os.path.exists(d):
    rows = [json.loads(l) for l in open(d)]
    print('diag rows', len(rows), 'keys', sorted(rows[0].keys()))
    print(json.dumps(rows[0]))
    print(json.dumps(rows[1]))
    nv = sum(1 for r in rows if r.get('visual'))
    ng = sum(1 for r in rows if 'ratio' in r)
    print('visual', nv, 'with-ratio', ng)
