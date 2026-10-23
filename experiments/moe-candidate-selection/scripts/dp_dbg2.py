#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug 2: is the stored row aligned with the frame, and with the sweep row?"""
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

keys, t0, poses = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
prof = DP.load_profiles('/home/zhangcy/.dsh_scratch/dp_smoke/dp-profiles-smoke.npz')
z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
skeys = [str(k) for k in z['keys']]
sgain = np.asarray(z['gain'], np.float64)
sm = {k: i for i, k in enumerate(skeys)}

print('[index] first 5 %s  last 5 %s  sorted=%s  unique=%d'
      % (prof['index'][:5], prof['index'][-5:],
         bool(np.all(np.diff(prof['index']) > 0)), len(set(prof['index'].tolist()))))
print('[keys equal] %s' % (skeys[:3] == keys[:3],))

mine = prof['origin'].astype(float) - np.nanmin(prof['surface'], axis=(1, 2))
sel = np.array([sm[keys[int(order[i])]] for i in range(len(mine))])
print('[sel] equals order? %s' % bool(np.all(sel == order[:len(mine)])))

DP.worker_init()
for i in (438, 536, 858, 1072, 1130, 208):
    r = DP.frame_profile(order[i])
    live = r['origin'] - float(np.min(r['surface']))
    j = int(sel[i])
    print('\ni=%d key=%s' % (i, keys[int(order[i])]))
    print('  stored %.6f   live %.6f   sweep[sel] %.6f (%s)'
          % (mine[i], live, sgain[j], skeys[j]))
    d = np.abs(sgain[sel] - live if False else sgain - live)
    k = int(np.argmin(np.abs(sgain - mine[i])))
    print('  sweep row whose gain == stored: %d  key=%s  |d|=%.2e'
          % (k, skeys[k], abs(sgain[k] - mine[i])))
    print('  stored d_values %s' % (prof['d_values'][i],))
    print('  live   d_values %s' % (r['d_values'],))
    print('  stored surface row8 %s' % (np.round(prof['surface'][i][8], 6),))
    print('  live   surface row8 %s' % (np.round(r['surface'][8], 6),))
