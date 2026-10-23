#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pass 3 + final scoring: refine the DP's novel cells, score every config exactly.

Pose rule per frame:
  chosen == 0        -> the front-end pose (no correction)
  chosen == the frame's own per-frame argmin cell -> the project's own refined pose
  otherwise          -> refined here, in that cell's 5x9 fine window
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
PROC = int(os.environ.get('DP_PROC', '40'))
prof_path = sys.argv[1] if len(sys.argv) > 1 else P + '/results/dp-profiles.npz'

t_all = time.time()
keys, t0, poses_table = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
keys_seq = [keys[int(i)] for i in order]
arrays = DP.truth_arrays(keys_seq)
prof = DP.attach_motion(DP.load_profiles(prof_path), keys, t0, order)
emis = DP.emissions(prof)
front = np.array(t0[order], np.float64)
n = len(front)
argmin_state = (np.argmin(emis['cost'], axis=1)).astype(np.int16)

z = np.load(OUT + '/sweep_full.npz', allow_pickle=True)
sgain = np.asarray(z['gain'], np.float64)
sP = np.asarray(z['P_search'], np.float64)
srot = np.asarray(z['P_rot'], np.float64)
sbad = np.asarray(z['bad'], np.uint8)
seps = np.asarray(z['eps'], np.float64)
i_eps = int(np.argmin(np.abs(seps - 0.01)))
gain_seq, P_seq, rot_seq, bad_seq = sgain[order], sP[order], srot[order], sbad[order]
gate = (gain_seq > 0.01) & (bad_seq == 0)

sw = np.load(OUT + '/dp_sweep_coarse.npz', allow_pickle=True)
choices = np.asarray(sw['choices']).astype(np.int16)     # (ncfg, n)
lam_g, eta_g, tag_g = np.asarray(sw['lam']), np.asarray(sw['eta']), np.asarray(sw['tag'])
print('[loaded] %d configs, %d frames, %.1f s' % (len(lam_g), n, time.time() - t_all), flush=True)


def ev(poses):
    return DP.evaluate(keys_seq, poses, arrays=arrays)


# --------------------------------------------------------------------------- #
# 1. how far is "refined in the frame's own argmin cell" from the project's pose?
# --------------------------------------------------------------------------- #
rng = np.random.RandomState(7)
same = np.nonzero((argmin_state != DP.STATE_ZERO))[0]
sample = rng.choice(same, size=min(120, len(same)), replace=False)
print('[check] refining %d frames in their own argmin cell, vs the project pose'
      % len(sample), flush=True)
s_v, d_v, c_v, t_v = DP.refine_cells(keys, t0, np.stack([sample, argmin_state[sample]], 1),
                                     order, PROC)
mine_sample = DP.shift_poses(front[sample], s_v, d_v)
dev = np.linalg.norm(mine_sample[:, :3, 3] - P_seq[sample][:, :3, 3], axis=1)
print('  deviation: median %.4f  p90 %.4f  max %.4f  exact(<1e-9) %.1f%%'
      % (np.median(dev), np.percentile(dev, 90), dev.max(), 100 * np.mean(dev < 1e-9)),
      flush=True)
print('  refine cost %.3f s/frame (core)' % float(np.mean(t_v)), flush=True)

# --------------------------------------------------------------------------- #
# 2. refine only the cells the DP actually moves away to
# --------------------------------------------------------------------------- #
need = {}
for k in range(len(lam_g)):
    if lam_g[k] == 0:
        continue                      # lam=0 is the per-frame branch: pose already known
    ch = choices[k]
    novel = np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]
    for r in novel:
        need.setdefault(int(r), set()).add(int(ch[r]))
pairs = [(r, a) for r, S in need.items() for a in sorted(S)]
print('\n[pass3] %d (frame, cell) refinements needed by the lam>0 grid (%.1f%% of frames touched)'
      % (len(pairs), 100.0 * len(need) / n), flush=True)
t = time.time()
if pairs:
    s_v, d_v, c_v, t_v = DP.refine_cells(keys, t0, pairs, order, PROC)
    S_TAB = np.zeros((n, DP.N_STATE))
    D_TAB = np.zeros((n, DP.N_STATE))
    S_TAB[np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])] = s_v
    D_TAB[np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])] = d_v
else:
    S_TAB = D_TAB = None
    t_v = np.zeros(1)
print('[pass3] %.1f s wall, %.2f s/frame/core' % (time.time() - t, float(np.mean(t_v))), flush=True)


def poses_for(ch):
    """Exact pose per frame under the documented rule."""
    out = np.array(front, np.float64, copy=True)
    # zero -> front (already), argmin cell -> the project's own refined pose
    use_proj = (ch != DP.STATE_ZERO) & (ch == argmin_state)
    out[use_proj] = P_seq[use_proj]
    novel = np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]
    if len(novel):
        if S_TAB is None:
            raise RuntimeError('novel cells but no refinement')
        out[novel] = DP.shift_poses(front[novel], S_TAB[novel, ch[novel]],
                                    D_TAB[novel, ch[novel]])
    return out


# --------------------------------------------------------------------------- #
# 3. reference rows and the exact sweep table
# --------------------------------------------------------------------------- #
print('\n=== reference rows ===')
print(DP.HEADER)
ref = {}
for tag, poses in (('front end (t0)', front),
                   ('per-frame search all', P_seq),
                   ('per-frame gate 0.01', np.where(gate[:, None, None], P_seq, front)),
                   ('gate 0.01 + rotation', np.where(gate[:, None, None],
                                                     rot_seq[:, i_eps], front))):
    ref[tag] = ev(poses)
    print(DP.fmt_row(tag, ref[tag]))

print('\n=== DP sweep, exact (fine refinement where the DP moves the cell) ===')
print('%-30s %6s  %6s  %6s  %6s  %6s  %5s  %5s  %6s'
      % ('config', 'n', 'MPE', 'MOE', 'median', 'ok', '>1m', '>5m', 'fine%'))
rows = []
t = time.time()
for k in range(len(lam_g)):
    poses = poses_for(choices[k])
    m = ev(poses)
    nz = float(np.mean(choices[k] != DP.STATE_ZERO))
    rows.append(dict(k=k, tag=str(tag_g[k]), lam=float(lam_g[k]), eta=float(eta_g[k]),
                     nz=nz, **{kk: m[kk] for kk in
                               ('n', 'mpe', 'moe', 'median', 'ok', 'gt1', 'gt5', 'gt10')}))
    print('%-30s %6d  %.4f  %.4f  %.4f  %5.1f%%  %5d  %5d  %5.1f%%'
          % ('%s lam=%g eta=%g' % (tag_g[k], lam_g[k], eta_g[k]), m['n'], m['mpe'],
             m['moe'], m['median'], 100 * m['ok'], m['gt1'], m['gt5'], 100 * nz))
print('[sweep] %.1f s' % (time.time() - t))

json.dump(dict(rows=rows, ref={k: {kk: vv for kk, vv in v.items() if kk not in
                                   ('per_date', 'err', 'rot', 'date')}
                               for k, v in ref.items()},
               dev=dict(median=float(np.median(dev)), p90=float(np.percentile(dev, 90)),
                        max=float(dev.max()), exact=float(np.mean(dev < 1e-9))),
               pairs=len(pairs), touched=len(need)),
          open(OUT + '/dp_final_table.json', 'w'), indent=1,
          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

np.savez(OUT + '/dp_sweep_exact.npz', choices=choices.astype(np.int8), lam=lam_g,
         eta=eta_g, tag=tag_g)
print('[saved] dp_final_table.json  dp_sweep_exact.npz  total %.1f min'
      % ((time.time() - t_all) / 60.0))
