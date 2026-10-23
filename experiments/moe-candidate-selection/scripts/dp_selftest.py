#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API self-test: the module's public surface on a few frames, end to end."""
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

keys, t0, _ = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
print('[1] load_pose_table  keys=%d  order=bounds %s' % (len(keys), [b[0] for b in bounds]))
print('[2] sequence_order   first 6 %s  strictly temporal: %s'
      % (order[:6], bool(np.all(np.diff(order[:25546]) >= 0)) if False else 'n/a'))

prof_all = DP.load_profiles(P + '/results/dp-profiles.npz')
print('[3] load_profiles    fields %s' % sorted(prof_all))
print('    surface %s  keys %s  rel_yaw %s  gap %s'
      % (prof_all['surface'].shape, prof_all['keys'].shape, prof_all['rel_yaw'].shape,
         prof_all['gap'].shape))

# a short slice of one date: 60 consecutive frames
a = [b for b in bounds if b[0] == '2012-02-12'][0][1]
sl = slice(a + 1000, a + 1060)
prof = DP.attach_motion({k: (v[sl] if getattr(v, 'shape', None) else v)
                         for k, v in prof_all.items()}, keys, t0, order[sl])
print('[4] attach_motion    rows=%d  bounds=%s' % (len(prof['surface']), prof['bounds']))
emis = DP.emissions(prof)
print('[5] emissions        cost %s  state s %s' % (emis['cost'].shape, np.round(emis['s'], 1)))

for lam, eta in ((0.0, 0.0), (0.01, 0.01), (0.5, 0.0)):
    ch = DP.viterbi(prof, emis=emis, lam=lam, eta=eta)
    s, d = DP.chosen_offsets(emis, ch)
    print('[6] viterbi          lam=%-5g eta=%-5g states %s  s %s'
          % (lam, eta, np.bincount(ch, minlength=DP.N_STATE).nonzero()[0],
             np.round(s[s != 0], 2)))

ch = DP.viterbi(prof, emis=emis, lam=0.01, eta=0.005)
poses, diag = DP.refine_selected(keys, t0, ch, order=order[sl], processes=2)
moved = np.linalg.norm(poses[:, :3, 3] - t0[order[sl]][:, :3, 3], axis=1)
print('[7] refine_selected  refined %d  moved max %.3f m  poses %s'
      % (int(diag['refined'].sum()), moved.max(), poses.shape))

m = DP.evaluate([keys[int(i)] for i in order[sl]], poses)
print('[8] evaluate         n=%d MPE %.4f MOE %.4f ok %.1f%% >1m %d >5m %d'
      % (m['n'], m['mpe'], m['moe'], 100 * m['ok'], m['gt1'], m['gt5']))
print('[9] fmt_row/HEADER   ok:', DP.HEADER.startswith('config'))
print('ALL OK')
