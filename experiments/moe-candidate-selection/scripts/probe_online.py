#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only probe: initial.npz key layout, per-date counts, SAM cache / image / scan presence."""
import json
import os
import sys
from collections import Counter, OrderedDict

INIT = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'

import numpy as np

with np.load(INIT, allow_pickle=False) as a:
    print('[npz] fields=', list(a.keys()))
    keys = [str(k) for k in a['keys']]
    poses = np.asarray(a['poses'], float)
print('[npz] n_keys=%d poses=%s' % (len(keys), poses.shape))
print('[npz] first/last=', keys[0], keys[-1])

by_date = OrderedDict()
for i, k in enumerate(keys):
    d = k.split('/')[0]
    by_date.setdefault(d, []).append((i, k))
print('[dates]')
for d, lst in by_date.items():
    print('   %s n=%d  idx[%d..%d]' % (d, len(lst), lst[0][0], lst[-1][0]))

# evenly spaced picks: 6 per date for 24 frames, 2 per date for 8
for per in (2, 4, 6):
    sel = []
    for d, lst in by_date.items():
        idxs = [int(round(j * (len(lst) - 1) / float(per - 1))) for j in range(per)]
        idxs = sorted(set(idxs))
        sel += [lst[j][1] for j in idxs]
    print('[pick per=%d] n=%d' % (per, len(sel)))
    for s in sel:
        print('    ', s, 'sample_index=', keys.index(s))

# existence checks on a couple of picked keys
cands = []
for d, lst in by_date.items():
    cands.append(lst[0][1])
    cands.append(lst[len(lst) // 2][1])

print('[exists]')
for k in cands:
    date = k.split('/')[0]
    stem = k.split('/')[1]
    paths = {
        'image_jpg': '/data2/dataset/NCLT/%s/image_sync/%s.jpg' % (date, stem),
        'image_png': '/data2/dataset/NCLT/%s/image_sync/%s.png' % (date, stem),
        'velodyne_sync': '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin' % (date, stem),
        'sam_cache': '/data2/zhangcy/sam-cache/%s/%s.npz' % (date, stem),
    }
    print('  ', k, json.dumps({n: os.path.exists(p) for n, p in paths.items()}))
sys.stdout.flush()
