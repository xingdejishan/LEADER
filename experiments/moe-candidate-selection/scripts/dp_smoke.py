#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke: 2000-frame profile build, gain reproduction vs sweep_full, DP+both baselines."""
import os
import sys
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch/dp_smoke'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
PROC = int(sys.argv[2]) if len(sys.argv) > 2 else 8

keys, t0, poses = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
print('[table] frames=%d dates=%s' % (len(keys), [(b[0], b[2] - b[1]) for b in bounds]))

t = time.time()
info = DP.build_profiles(OUT, order=order, processes=PROC, block=500, limit=N)
print('[pass1]', info, 'wall %.1f s' % (time.time() - t))

path, n = DP.merge_blocks(OUT, os.path.join(OUT, 'dp-profiles-smoke.npz'), order=order[:N])
print('[merge]', path, n)
prof = DP.attach_motion(DP.load_profiles(path), keys, t0, order[:n])
print('[fields]', sorted(prof.keys()))
print('[ok] %d/%d  bad=%s' % (int(prof['ok'].sum()), n,
                              np.bincount(prof['argmin_a'][prof['ok'] == 0] + 1)))
print('[secs] mean %.4f  total %.1f' % (float(np.mean(prof['seconds'])),
                                        float(np.sum(prof['seconds']))))

# --- reproduce the sweep's descent depth (gain = origin - surface.min())
z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
skeys = [str(k) for k in z['keys']]
sgain = np.asarray(z['gain'], np.float64)
sm = {k: i for i, k in enumerate(skeys)}
sel = [sm[keys[int(order[i])]] for i in range(n)]
ref = sgain[sel]
mine = prof['origin'].astype(np.float64) - np.nanmin(prof['surface'], axis=(1, 2))
m = prof['ok'].astype(bool) & np.isfinite(ref)
print('\n[gain vs sweep] n=%d' % int(m.sum()))
print('  corr %.4f   mean abs diff %.2e   p99 abs diff %.2e   max %.2e'
      % (np.corrcoef(mine[m], ref[m])[0, 1], np.abs(mine[m] - ref[m]).mean(),
         np.percentile(np.abs(mine[m] - ref[m]), 99), np.abs(mine[m] - ref[m]).max()))
print('  gain>0.01 mine %d  sweep %d   both %d'
      % (int((mine[m] > 0.01).sum()), int((ref[m] > 0.01).sum()),
         int(((mine[m] > 0.01) & (ref[m] > 0.01)).sum())))

# --- per-frame argmin at the coarse cell vs the sweep's P_search (translation)
emis = DP.emissions(prof)
pos = np.arange(n)
chosen_argmin = (np.argmin(emis['cost'][:, 1:], axis=1) + 1).astype(np.int16)
chosen_argmin[~prof['ok'].astype(bool)] = 0
per_frame = DP.coarse_poses(t0[order[:n]], emis, chosen_argmin)
P_search = np.asarray(z['P_search'], np.float64)[sel]
d_shift = np.linalg.norm(per_frame[:, :3, 3] - P_search[:, :3, 3], axis=1)
print('\n[coarse argmin vs sweep P_search] median %.3f m  p90 %.3f  <0.6 m: %.1f%%'
      % (np.median(d_shift), np.percentile(d_shift, 90), 100.0 * (d_shift < 0.6).mean()))

# --- DP at a few settings: how much is non-zero?
print('\n[dp] lam / eta -> non-zero share, mean |shift| m')
for lam in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5):
    for eta in (0.0, 0.005, 0.01):
        t = time.time()
        ch = DP.viterbi(prof, emis=emis, lam=lam, eta=eta)
        s, d = DP.chosen_offsets(emis, ch)
        nz = ch != DP.STATE_ZERO
        print('  lam=%-6g eta=%-6g nonzero %6d (%5.1f%%)  mean|s| %.2f  states %s  %.1fs'
              % (lam, eta, int(nz.sum()), 100.0 * nz.mean(),
                 float(np.abs(s[nz]).mean()) if nz.any() else 0.0,
                 sorted(set(ch.tolist()))[:8], time.time() - t))
