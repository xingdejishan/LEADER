"""Probe: truth coverage, scan paths, and a timing benchmark on a small sample."""
import json
import os
import sys
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(P, 'stages/fast_search'))
sys.path.insert(0, os.path.join(P, 'stages'))

TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
DATASET = '/data2/dataset/NCLT'

z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    raw = json.load(open(os.path.join(TRUTH, f)))
    for ts, v in raw.items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)
print('truth dates:', sorted({k[0] for k in truth}), 'total', len(truth))

def truth_of(k):
    d, ts = k.split('/')
    return truth.get((d, ts))

hit = sum(1 for k in keys if truth_of(k) is not None)
print('npz keys: %d  with truth: %d  missing: %d' % (len(keys), hit, len(keys) - hit))

miss = [k for k in keys if truth_of(k) is None][:5]
print('sample missing:', miss)

# scan path existence
def scan_path(k):
    d, ts = k.split('/')
    return '%s/%s/velodyne_sync/%s.bin' % (DATASET, d, ts)

sel = [0, len(keys)//3, 2*len(keys)//3, len(keys)-1]
for i in sel:
    p = scan_path(keys[i])
    print('scan exists', os.path.exists(p), p)

# expected position error of t0 on the full set (for pass ordering)
e0 = np.full(len(keys), np.nan)
for i, k in enumerate(keys):
    g = truth_of(k)
    if g is not None:
        e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
print('t0 pos err: mean %.4f median %.4f  >0.2m %d  >1m %d'
      % (np.nanmean(e0), np.nanmedian(e0), int((e0 > 0.2).sum()), int((e0 > 1).sum())))

# ---------------------------------------------------------------- timing
sys.path.insert(0, os.path.join(P, '.backups/remove-rotation-20261022-182520'))
import fast_lib as F
import pipeline as Pn
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import importlib.util

spec = importlib.util.spec_from_file_location(
    'tru_bak', os.path.join(P, '.backups/remove-rotation-20261022-182520/three_route_unified.py'))
# do not exec: it would import gate etc. Instead pull rot_only out by exec of just that function.
src = open(os.path.join(P, '.backups/remove-rotation-20261022-182520/three_route_unified.py')).read()
start = src.index('def rot_only(')
end = src.index('def one(')
ns = {'np': np, 'Rotation': Rotation}
exec(compile(src[start:end], 'rot_only', 'exec'), ns)
rot_only = ns['rot_only']

refs = json.load(open(os.path.join(P, 'inputs/refs.json')))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
maps = F.MapSource(refs, pos, radius=60.0)
cfg = Pn.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=1)

import math
sample = [i for i in np.linspace(0, len(keys)-1, 12).astype(int)]
tt = dict(decode=0.0, mapcrop=0.0, coarse=0.0, solve=0.0, rot=0.0)
n = 0
for i in sample:
    k = keys[i]
    rec = {'key': k, 'scan_path': scan_path(k), 'world_from_body': t0[i].tolist()}
    a = time.time()
    scan = Pn.prepare_scan(rec, cfg)
    b = time.time(); tt['decode'] += b - a
    T0 = t0[i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
    c = time.time(); tt['mapcrop'] += c - b
    u = Pn.along_track_direction(T0)
    cs = F.coarse_stage(mp, scan, T0, u, workers=1, n_coarse=1000, topk=6)
    d = time.time(); tt['coarse'] += d - c
    pose, diag = Pn.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
    e = time.time(); tt['solve'] += e - d
    tree = cKDTree(mp)
    q = scan[np.linalg.norm(scan, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]
    T = np.asarray(pose, float)
    for eps in (1e-6, 0.005, 0.01, 0.02, 0.05):
        _ = rot_only(tree, mp, q, T)
    _ = rot_only(tree, mp, q, T0)
    f = time.time(); tt['rot'] += f - e
    n += 1
    print('frame %d: scan %d map %d  s=%.3f cost=%.4f  total %.3fs'
          % (i, len(scan), len(mp), diag['s'], diag['cost'], f - a))

print('n=%d  per-frame ms: ' % n + '  '.join('%s=%.1f' % (k, 1000*v/n) for k, v in tt.items()))
print('sum per frame ms %.1f  -> 8 workers, 104728 frames: %.1f min'
      % (1000*sum(tt.values())/n, 104728*sum(tt.values())/n/8/60))
