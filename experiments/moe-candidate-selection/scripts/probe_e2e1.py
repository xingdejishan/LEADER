#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe: sam cache roots, path resolution, frames.json ordering, merged npz."""
import json
import os
import sys

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'

for date in ('2012-02-12', '2012-02-19'):
    p = os.path.join(DATA, 'sam', date)
    print('sam dir', p, 'islink', os.path.islink(p), '->', os.path.realpath(p))
    cm = os.path.join(p, 'cache_manifest.json')
    print('  cache_manifest', os.path.exists(cm), os.path.islink(cm))
    print('  realpath', os.path.realpath(cm))

print('cam5 dir islink', os.path.islink(os.path.join(DATA, 'NCLT', '2012-02-12', 'cam5')))

frames = json.load(open(os.path.join(PROJ, 'inputs/frames.json')))
import numpy as np
z = np.load(os.path.join(PROJ, 'inputs/initial.npz'), allow_pickle=True)
keys = [str(k) for k in z['keys']]
fk = [r['key'] for r in frames]
print('frames', len(frames), 'initial', len(keys), 'same order', fk == keys)
print('frame0', frames[0])
print('dates hist', {d: sum(1 for k in keys if k.startswith(d)) for d in sorted({k.split('/')[0] for k in keys})})

merged = np.load(os.path.join(PROJ, 'results/three_route_merged.npz'), allow_pickle=True)
print('merged keys', merged['keys'].shape, merged['poses'].shape, merged['poses'].dtype)
print('merged key order same', [str(k) for k in merged['keys']] == keys)
print('first5 merged t', merged['poses'][:5, :3, 3].tolist())
print('first5 init  t', z['poses'][:5, :3, 3].tolist())
d = np.linalg.norm(merged['poses'][:5, :3, 3] - z['poses'][:5, :3, 3], axis=1)
print('first5 |merged-initial|', d.tolist())

refs = json.load(open(os.path.join(PROJ, 'inputs/refs.json')))
print('refs', len(refs), 'ref0 scan_path', refs[0]['scan_path'])
print('ref0 islink', os.path.islink(os.path.dirname(refs[0]['scan_path'])), os.path.realpath(refs[0]['scan_path']))
print('dirsize of results', sum(os.path.getsize(os.path.join(PROJ, 'results', f)) for f in os.listdir(os.path.join(PROJ, 'results'))))
print('cwd', os.getcwd())
print('SCAN sample path', os.path.join(PROJ, 'inputs/frames.json'))
