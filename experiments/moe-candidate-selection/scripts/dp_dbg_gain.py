#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug: why does the reproduced gain differ from the sweep's gain?"""
import os
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
sbad = np.asarray(z['bad'], np.uint8)
sP = np.asarray(z['P_search'], np.float64)
sm = {k: i for i, k in enumerate(skeys)}

mine = prof['origin'].astype(float) - np.nanmin(prof['surface'], axis=(1, 2))
sel = np.array([sm[keys[int(order[i])]] for i in range(len(mine))])

DP.worker_init()
live = []
for i in range(6):
    r = DP.frame_profile(order[i])
    live.append(r['origin'] - float(np.min(r['surface'])))
live = np.array(live)

print('%6s %14s %14s %14s %10s %6s' % ('i', 'mine(stored)', 'mine(live)', 'sweep', 'd(sweep)', 'bad'))
for i in range(6):
    print('%6d %14.6f %14.6f %14.6f %10.2e %6d'
          % (i, mine[i], live[i], sgain[sel[i]], sgain[sel[i]] - live[i], sbad[sel[i]]))

d = sgain[sel] - mine
print('\noverall: mean %.3e  median %.3e  rms %.3e' % (d.mean(), np.median(d), np.sqrt((d ** 2).mean())))
top = np.argsort(-np.abs(d))[:10]
print('worst frames:')
for i in top:
    print('  i=%6d mine %.6f sweep %.6f  bad=%d  origin %.6f  smin %.6f  t0=%s'
          % (i, mine[i], sgain[sel[i]], sbad[sel[i]], prof['origin'][i],
             np.nanmin(prof['surface'][i]), keys[int(order[i])]))
print('\ngain>0.01: mine %d sweep %d' % ((mine > 0.01).sum(), (sgain[sel] > 0.01).sum()))
m = np.abs(mine) > 0.005
print('corr on |gain|>0.005 (n=%d): %.4f' % (m.sum(), np.corrcoef(mine[m], sgain[sel][m])[0, 1]))
