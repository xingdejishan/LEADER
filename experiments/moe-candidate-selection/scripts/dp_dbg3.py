#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug 3: single-process, small block, stored row vs live recomputation."""
import os
import shutil
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch/dp_dbg3'
shutil.rmtree(OUT, ignore_errors=True)
keys, t0, poses = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)

info = DP.build_profiles(OUT, order=order, processes=1, block=10, limit=40)
print('[pass1]', info)
print('[files]', sorted(os.listdir(OUT)))
path, n = DP.merge_blocks(OUT, os.path.join(OUT, 'm.npz'), order=order[:40])
prof = DP.load_profiles(path)
print('[index]', prof['index'])

DP.worker_init()
bad = 0
for i in range(n):
    r = DP.frame_profile(order[i])
    live = r['origin'] - float(np.min(r['surface']))
    mine = float(prof['origin'][i]) - float(np.nanmin(prof['surface'][i]))
    same = abs(live - mine) < 1e-5 and np.allclose(r['d_values'], prof['d_values'][i])
    if not same:
        bad += 1
        print('  MISMATCH i=%d key=%s  live %.6f  stored %.6f  d %s / %s'
              % (i, keys[int(order[i])], live, mine, r['d_values'], prof['d_values'][i]))
print('[check] %d/%d mismatched' % (bad, n))
