#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug 4: is frame_profile deterministic for one frame, and does a warm cache change it?"""
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

keys, t0, poses = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)


def show(tag, i):
    r = DP.frame_profile(i)
    print('  %-22s i=%6d key=%s  d=%s  origin=%.6f  smin=%.6f  gain=%.6f  scan=%d map=%d'
          % (tag, i, keys[i], np.round(r['d_values'], 2), r['origin'],
             float(np.min(r['surface'])), r['origin'] - float(np.min(r['surface'])),
             r['scan_points'], r['map_points']))
    return r


i5, i10 = int(order[5]), int(order[10])
print('[keys] order[5]=%d order[10]=%d  same_date=%s'
      % (i5, i10, keys[i5].split('/')[0] == keys[i10].split('/')[0]))
DP.worker_init()
a = show('cold, order[5]', i5)
b = show('again, order[5]', i5)
c = show('order[10]', i10)
d = show('order[5] again', i5)
e = show('order[10] again', i10)
DP.worker_init()
f = show('fresh maps, order[5]', i5)
g = show('fresh maps, order[10]', i10)
print('[determinism] a==b %s  d==a %s  f==a %s  c==e %s  g==c %s'
      % (np.allclose(a['d_values'], b['d_values']), np.allclose(d['d_values'], a['d_values']),
         np.allclose(f['d_values'], a['d_values']), np.allclose(c['d_values'], e['d_values']),
         np.allclose(g['d_values'], c['d_values'])))
print('[centres] order[5]=%s  order[10]=%s' % (np.round(t0[i5][:3, 3], 2), np.round(t0[i10][:3, 3], 2)))
print('[cache] hit=%d miss=%d' % (DP._G['maps'].hit, DP._G['maps'].miss))
