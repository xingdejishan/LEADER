#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second DP grid: where the hand-set gate is actually optimal (eta 0.015-0.05).

The first grid showed the sequence effect only appears near eta = 0.01, but the gate's
own MPE minimum sits at 0.02-0.03, so the comparison has to be made there too.
"""
import json
import os
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch'
LAM = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
ETA = (0.015, 0.02, 0.03, 0.05)

keys, t0, _ = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
keys_seq = [keys[int(i)] for i in order]
arrays = DP.truth_arrays(keys_seq)
prof = DP.attach_motion(DP.load_profiles(P + '/results/dp-profiles.npz'), keys, t0, order)
emis = DP.emissions(prof)
front = np.array(t0[order], np.float64)
n = len(front)
argmin_state = np.argmin(emis['cost'], axis=1).astype(np.int16)

z = np.load(OUT + '/sweep_full.npz', allow_pickle=True)
sP = np.asarray(z['P_search'], np.float64)
sbad = np.asarray(z['bad'], np.uint8)
gain_seq = np.asarray(z['gain'], np.float64)[order]
P_seq = sP[order]
bad_seq = sbad[order]

choices, rows = [], []
need = {}
for lam in LAM:
    for eta in ETA:
        ch = DP.viterbi(prof, emis=emis, lam=lam, eta=eta)
        choices.append(ch)
        for r in np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]:
            need.setdefault(int(r), set()).add(int(ch[r]))

pairs = [(r, a) for r, S in need.items() for a in sorted(S)]
print('[extend] %d configs, %d novel (frame, cell) refinements' % (len(choices), len(pairs)), flush=True)
S_TAB = np.zeros((n, DP.N_STATE))
D_TAB = np.zeros((n, DP.N_STATE))
if pairs:
    s_v, d_v, _c, t_v = DP.refine_cells(keys, t0, pairs, order, 40)
    S_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = s_v
    D_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = d_v
    print('[extend] refined in %.0f s of core time' % float(np.sum(t_v)), flush=True)


def poses_for(ch):
    out = np.array(front, np.float64, copy=True)
    use = (ch != DP.STATE_ZERO) & (ch == argmin_state)
    out[use] = P_seq[use]
    novel = np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]
    if len(novel):
        out[novel] = DP.shift_poses(front[novel], S_TAB[novel, ch[novel]],
                                    D_TAB[novel, ch[novel]])
    return out


print('%-8s %-7s %6s %7s %7s %7s %8s %6s %6s'
      % ('lam', 'eta', 'fine%', 'MPE', 'MOE', 'median', 'ok', '>1m', '>5m'))
for k, ch in enumerate(choices):
    m = DP.evaluate(keys_seq, poses_for(ch), arrays=arrays)
    lam, eta = LAM[k // len(ETA)], ETA[k % len(ETA)]
    nz = float(np.mean(ch != DP.STATE_ZERO))
    rows.append(dict(tag='ext', lam=lam, eta=eta, nz=nz,
                     **{kk: m[kk] for kk in ('n', 'mpe', 'moe', 'median', 'ok',
                                             'gt1', 'gt5', 'gt10')}))
    print('%-8g %-7g %5.1f%% %7.4f %7.4f %7.4f %7.1f%% %6d %6d'
          % (lam, eta, 100 * nz, m['mpe'], m['moe'], m['median'], 100 * m['ok'],
             m['gt1'], m['gt5']), flush=True)

np.savez(OUT + '/dp_sweep_extend.npz', choices=np.stack([c.astype(np.int8) for c in choices]),
         lam=np.array([LAM[k // len(ETA)] for k in range(len(choices))]),
         eta=np.array([ETA[k % len(ETA)] for k in range(len(choices))]),
         tag=np.array(['ext'] * len(choices)), pairs=len(pairs))
json.dump(rows, open(OUT + '/dp_extend_table.json', 'w'), indent=1,
          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('[saved] dp_sweep_extend.npz / dp_extend_table.json')
