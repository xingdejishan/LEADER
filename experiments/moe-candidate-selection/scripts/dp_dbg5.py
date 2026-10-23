#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug 5: emissions sanity - does cost[:,0] equal the origin and the grid min?"""
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

prof = DP.load_profiles('/home/zhangcy/.dsh_scratch/dp_smoke/dp-profiles-smoke.npz')
emis = DP.emissions(prof)
cost = emis['cost']
origin = prof['origin'].astype(float)
smin = np.nanmin(prof['surface'], axis=(1, 2))
gain = origin - smin

print('[cost]', cost.shape, 'nan=%d' % np.isnan(cost).sum())
print('[cost[:,0] == origin] %s' % np.allclose(cost[:, 0], origin, equal_nan=True))
print('[cost[:,1:].min(1) == smin] %s' % np.allclose(cost[:, 1:].min(1), smin, equal_nan=True))
print('[gain] >0 %.1f%%  >0.01 %.1f%%' % (100 * np.mean(gain > 0), 100 * np.mean(gain > 0.01)))
print('[cost[9] vs cost[0]] cost9<cost0 %.1f%%   min over states 9? %s'
      % (100 * np.mean(cost[:, 9] < cost[:, 0]), bool(np.allclose(cost[:, 9], 0))))
print('[argmin over 18] counts %s' % np.bincount(np.argmin(cost, axis=1), minlength=18).tolist())
print('[argmin over 17 cells, +1] counts %s'
      % np.bincount(np.argmin(cost[:, 1:], axis=1) + 1, minlength=18).tolist())
print('[sample cost row0] %s' % np.round(cost[0], 6).tolist())
print('[sample origin/smin] %.6f %.6f' % (origin[0], smin[0]))
print('[surface row0]\n%s' % np.round(prof['surface'][0], 5))
