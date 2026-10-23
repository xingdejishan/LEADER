#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe 4: ledger row shape, scan path resolution, per-frame timing pilot."""
import json
import os
import sys
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
RES = P + '/results/'

print('[ledger]')
for name in ('full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl', 'full-shard1.jsonl'):
    path = RES + name
    if not os.path.exists(path):
        print('  MISSING', name)
        continue
    with open(path) as fh:
        line = fh.readline()
    r = json.loads(line)
    print('  ==', name)
    print('    row keys:', sorted(r.keys()))
    print('    key=%s' % r.get('key'))
    print('    scan_path=%s exists=%s' % (r.get('scan_path'), os.path.exists(str(r.get('scan_path')))))
    print('    seconds=%s skipped=%s t0?=%s pose?=%s' % (r.get('seconds'), r.get('skipped'),
                                                         't0' in r, 'pose' in r))
    print('    gain=%s s=%s d=%s depth?=%s' % (r.get('gain'), r.get('s'), r.get('d'),
                                              'depth' in r))

# count scan paths resolvable from the NCLT convention for all keys
z = np.load(RES + 'full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
n_ok = 0
missing = []
for k in keys[::500]:
    d, ts = k.split('/')
    p = '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin' % (d, ts)
    if os.path.exists(p):
        n_ok += 1
    else:
        missing.append(k)
print('\n[convention] sampled %d keys, found %d, missing %d' % (len(keys[::500]), n_ok, len(missing)))
print('   missing sample:', missing[:5])

# --- per-frame timing pilot using the project's own code
sys.path.insert(0, P)
sys.path.insert(0, os.path.join(P, 'stages', 'fast_search'))
sys.path.insert(0, os.path.join(P, 'stages'))
import fast_lib as F
import pipeline as PL
import adaptive

refs = json.load(open(P + '/inputs/refs.json'))
for row in refs:
    row['scan_path'] = os.path.realpath(row['scan_path'])
positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
t0s = np.asarray(z['t0'], np.float64)

maps = F.MapSource(refs, positions, radius=60.0)
cfg = PL.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=1)

# pick 20 frames spread over the set
pick = np.linspace(0, len(keys) - 1, 20).astype(int)
tt = dict(decode=0.0, prep=0.0, map=0.0, coarse=0.0, gain=0.0)
dts = []
surfmins = []
for i in pick:
    k = keys[i]
    d, ts = k.split('/')
    sp = '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin' % (d, ts)
    a = time.time()
    pts = F.decode(sp)
    tt['decode'] += time.time() - a
    a = time.time()
    rad = np.linalg.norm(pts, axis=1)
    scan = F.voxel(pts[(rad > 1.0) & (rad < 60.0)], 0.3)
    tt['prep'] += time.time() - a
    T0 = t0s[i]
    centre = T0[:3, 3]
    a = time.time()
    triple = maps.nearby(centre)[:3]
    mapp = maps.local(centre, triple)
    tt['map'] += time.time() - a
    u = PL.along_track_direction(T0)
    a = time.time()
    coarse = F.coarse_stage(mapp, scan, T0, u, workers=1, n_coarse=1000, topk=6)
    tt['coarse'] += time.time() - a
    tt['gain'] += 0.0
    dts.append(time.time() - a)
    surfmins.append(float(coarse['surface'].min()))
print('\n[timing] n=%d  ' % len(pick) + json.dumps({k: round(v / len(pick), 4) for k, v in tt.items()}))
print('[surface] mins', np.round(surfmins, 4))
print('[range]   S_COARSE', F.S_COARSE, '\n          D_COARSE', F.D_COARSE)
print('[scans]   n=%d  map=%d' % (len(scan), len(mapp)))
print('[cache]   hit=%d miss=%d' % (maps.hit, maps.miss))
np.savez('/home/zhangcy/.dsh_scratch/_pilot.npz', surface=np.stack([surfmins]))
