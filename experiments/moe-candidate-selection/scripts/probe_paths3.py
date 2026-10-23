#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only probe #3: per-key raw input availability (cam5 image, velodyne, SAM cache)."""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
NCLT = os.path.join(DATA, 'NCLT')
SAM = os.path.join(DATA, 'sam')
INIT = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'

with np.load(INIT, allow_pickle=False) as a:
    keys = [str(k) for k in a['keys']]

by_date = OrderedDict()
for k in keys:
    by_date.setdefault(k.split('/')[0], []).append(k)

def pick(per):
    out = []
    for d, lst in by_date.items():
        idxs = sorted(set(int(round(j * (len(lst) - 1) / float(per - 1))) for j in range(per)))
        out += [lst[j] for j in idxs]
    return out

# directory listing per date
for d in by_date:
    cam5 = os.path.join(NCLT, d, 'cam5')
    ent = sorted(os.listdir(cam5))[:3] if os.path.isdir(cam5) else 'MISSING'
    vel = os.path.join(NCLT, d, 'velodyne_sync')
    vent = sorted(os.listdir(vel))[:3] if os.path.isdir(vel) else 'MISSING'
    samm = os.path.join(SAM, d, 'cache_manifest.json')
    blob = json.load(open(samm)) if os.path.exists(samm) else None
    print('[dir] %s cam5=%s velodyne=%s sam_frames=%s' % (
        d, ent, vent, ('%d' % len(blob['frames'])) if blob else 'MISSING'))
    print('      sam_complete=', os.path.exists(os.path.join(SAM, d, 'complete.json')))

for per in (2, 6):
    sel = pick(per)
    print('=== per=%d n=%d' % (per, len(sel)))
    for k in sel:
        d, stem = k.split('/')
        blob = json.load(open(os.path.join(SAM, d, 'cache_manifest.json')))
        fr = blob['frames'].get(k)
        cam5 = os.path.join(NCLT, d, 'cam5')
        cands = [f for f in sorted(os.listdir(cam5)) if f.startswith(stem)] if os.path.isdir(cam5) else []
        img_ok = bool(cands)
        vel = os.path.join(NCLT, d, 'velodyne_sync', stem + '.bin')
        vel_real = os.path.realpath(vel)
        print('  %-32s sam_ok=%-5s sam_file=%-12s cam=%-5s img=%s vel=%s' % (
            k, fr is not None,
            (os.path.basename(fr['cache_file'])[:10] if fr else '-'),
            img_ok, cands[:1], os.path.exists(vel)))
sys.stdout.flush()
