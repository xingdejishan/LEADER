#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pass 1 runner: build every frame's 17x3 cost surface, then merge to one npz.

    python dp_pass1.py [processes] [block]
"""
import os
import sys
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

PROC = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 500
OUT = '/home/zhangcy/.dsh_scratch/dp_profiles'

t = time.time()
keys, t0, poses = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
print('[plan] frames=%d processes=%d block=%d dates=%s'
      % (len(keys), PROC, BLOCK, [(b[0], b[2] - b[1]) for b in bounds]), flush=True)
info = DP.build_profiles(OUT, order=order, processes=PROC, block=BLOCK)
print('[pass1]', info, flush=True)
path, n = DP.merge_blocks(OUT, os.path.join(P + '/results/dp-profiles.npz'), order=order,
                          keys=keys, t0=t0)
print('[merge] %s  rows=%d  wall %.1f s' % (path, n, time.time() - t), flush=True)

prof = DP.load_profiles(path)
ok = prof['ok'].astype(bool)
print('[ok] %d/%d  thin_map=%d  no_reference=%d  scan mean %.0f  map mean %.0f'
      % (int(ok.sum()), n, int(((~ok) & (prof['map_points'] < 500)).sum()),
         int((~ok).sum()), float(np.mean(prof['scan_points'])), float(np.mean(prof['map_points']))),
      flush=True)
print('[seconds/frame] mean %.3f  total %.1f core-s' % (float(np.mean(prof['seconds'])),
                                                        float(np.sum(prof['seconds']))), flush=True)
print('[size] %.1f MB' % (os.path.getsize(path) / 1e6), flush=True)
