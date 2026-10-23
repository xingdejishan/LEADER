#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pass 2: the DP sweep at coarse resolution + the reference rows.

Runs every (lambda, eta) of the specified and the calibrated grid, scores each with
the coarse cell pose (no fine refinement yet), and saves the chosen states so pass 3
can refine exactly the cells that were chosen.
"""
import json
import os
import sys
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch'
PROF = sys.argv[1] if len(sys.argv) > 1 else P + '/results/dp-profiles.npz'

# the grid as specified, and the calibrated one (the cost differences are 1e-3..1e-1 m)
SPEC_LAM = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SPEC_ETA = (0.0, 0.005, 0.01, 0.02)
CAL_LAM = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
CAL_ETA = (0.0, 0.001, 0.002, 0.005, 0.01)

keys, t0, poses_table = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
t = time.time()
prof = DP.attach_motion(DP.load_profiles(PROF), keys, t0, order)
print('[profiles] %d rows  load+attach %.1f s' % (len(prof['surface']), time.time() - t))
emis = DP.emissions(prof)
print('[ok] %d/%d' % (int(np.sum(emis['good'])), len(emis['good'])))

keys_seq = [keys[int(i)] for i in order]
front = t0[order]
arrays = DP.truth_arrays(keys_seq)

z = np.load(OUT + '/sweep_full.npz', allow_pickle=True)
sgain = np.asarray(z['gain'], np.float64)
sP = np.asarray(z['P_search'], np.float64)
srot = np.asarray(z['P_rot'], np.float64)
srot_t0 = np.asarray(z['P_rot_t0'], np.float64)
sbad = np.asarray(z['bad'], np.uint8)
seps = np.asarray(z['eps'], np.float64)
i_eps = int(np.argmin(np.abs(seps - 0.01)))
gain_seq = sgain[order]
P_seq = sP[order]
rot_seq = srot[order]
bad_seq = sbad[order]
gate = (gain_seq > 0.01) & (bad_seq == 0)


def ev(poses):
    return DP.evaluate(keys_seq, poses, arrays=arrays)


print('\n=== reference rows (sequence order) ===')
print(DP.HEADER)
rows = {}
rows['front end (t0)'] = ev(front)
print(DP.fmt_row('front end (t0)', rows['front end (t0)']))
rows['rotation at t0 (eps 1e-6)'] = ev(srot_t0[order])
print(DP.fmt_row('rotation at t0 (eps 1e-6)', rows['rotation at t0 (eps 1e-6)']))
rows['per-frame search all'] = ev(P_seq)
print(DP.fmt_row('per-frame search all', rows['per-frame search all']))
gated = np.where(gate[:, None, None], P_seq, front)
rows['per-frame gate 0.01'] = ev(gated)
print(DP.fmt_row('per-frame gate 0.01', rows['per-frame gate 0.01']))
withrot = np.where(gate[:, None, None], rot_seq[:, i_eps], front)
rows['gate 0.01 + rot (best)'] = ev(withrot)
print(DP.fmt_row('gate 0.01 + rot (best)', rows['gate 0.01 + rot (best)']))
print('  (gate 0.01 keeps %d frames = %.1f%%)' % (int(gate.sum()), 100 * gate.mean()))

# --- the sweep, coarse poses
rows_all, choices = [], []
t = time.time()
grid = [('spec', l, e) for l in SPEC_LAM for e in SPEC_ETA] + \
       [('cal', l, e) for l in CAL_LAM for e in CAL_ETA]
for tag, lam, eta in grid:
    ch = DP.viterbi(prof, emis=emis, lam=lam, eta=eta)
    poses = DP.coarse_poses(front, emis, ch)
    m = DP.evaluate(keys_seq, poses)
    nz = int((ch != DP.STATE_ZERO).sum())
    rows_all.append(dict(tag=tag, lam=lam, eta=eta, nonzero=nz, **{
        k: m[k] for k in ('n', 'mpe', 'moe', 'median', 'ok', 'gt1', 'gt5', 'gt10')}))
    choices.append(ch)
    print('[sweep %-4s lam=%-7g eta=%-6g] %s  nz=%6d (%.1f%%)'
          % (tag, lam, eta, DP.fmt_row('', m).strip(), nz, 100.0 * nz / len(ch)), flush=True)
print('[sweep] %d configs in %.1f s' % (len(grid), time.time() - t))

np.savez(OUT + '/dp_sweep_coarse.npz',
         choices=np.stack([c.astype(np.int8) for c in choices]),
         lam=np.array([g[1] for g in grid]), eta=np.array([g[2] for g in grid]),
         tag=np.array([g[0] for g in grid]),
         mpe=np.array([r['mpe'] for r in rows_all]),
         moe=np.array([r['moe'] for r in rows_all]),
         ok=np.array([r['ok'] for r in rows_all]),
         median=np.array([r['median'] for r in rows_all]),
         gt1=np.array([r['gt1'] for r in rows_all]),
         gt5=np.array([r['gt5'] for r in rows_all]),
         nonzero=np.array([r['nonzero'] for r in rows_all]))
print('[saved] %s/dp_sweep_coarse.npz' % OUT)

with open(OUT + '/dp_baselines.json', 'w') as fh:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'per_date'}
               for k, v in rows.items()}, fh, indent=1)
print('[saved] baselines')
