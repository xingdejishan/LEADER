#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analysis of the exact DP sweep: full table, per date, temporal stats, bursts."""
import json
import os
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch'
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
sgain = np.asarray(z['gain'], np.float64)
sP = np.asarray(z['P_search'], np.float64)
srot = np.asarray(z['P_rot'], np.float64)
sbad = np.asarray(z['bad'], np.uint8)
seps = np.asarray(z['eps'], np.float64)
i_eps = int(np.argmin(np.abs(seps - 0.01)))
gain_seq, P_seq, rot_seq, bad_seq = sgain[order], sP[order], srot[order], sbad[order]
gate = (gain_seq > 0.01) & (bad_seq == 0)

sw = np.load(OUT + '/dp_sweep_exact.npz', allow_pickle=True)
choices = np.asarray(sw['choices']).astype(np.int16)
lam_g, eta_g, tag_g = np.asarray(sw['lam']), np.asarray(sw['eta']), np.asarray(sw['tag'])

# refine the cells the DP moves away to (208 pairs, 5 s)
need = {}
for k in range(len(lam_g)):
    if lam_g[k] == 0:
        continue
    ch = choices[k]
    for r in np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]:
        need.setdefault(int(r), set()).add(int(ch[r]))
pairs = [(r, a) for r, S in need.items() for a in sorted(S)]
S_TAB = np.zeros((n, DP.N_STATE))
D_TAB = np.zeros((n, DP.N_STATE))
if pairs:
    s_v, d_v, _c, _t = DP.refine_cells(keys, t0, pairs, order, 40)
    S_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = s_v
    D_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = d_v


def poses_for(ch):
    out = np.array(front, np.float64, copy=True)
    use = (ch != DP.STATE_ZERO) & (ch == argmin_state)
    out[use] = P_seq[use]
    novel = np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]
    if len(novel):
        out[novel] = DP.shift_poses(front[novel], S_TAB[novel, ch[novel]],
                                    D_TAB[novel, ch[novel]])
    return out


def ev(poses):
    return DP.evaluate(keys_seq, poses, arrays=arrays)


refs = [('front end (t0)', front),
        ('per-frame search, all frames', P_seq),
        ('per-frame gate 0.01 (current)', np.where(gate[:, None, None], P_seq, front)),
        ('per-frame gate 0.01 + rotation', np.where(gate[:, None, None], rot_seq[:, i_eps], front))]
ref_rows = []
for tag, poses in refs:
    m = ev(poses)
    ref_rows.append((tag, m, float(np.mean(poses is not None and True))))

rows = []
for k in range(len(lam_g)):
    m = ev(poses_for(choices[k]))
    rows.append(dict(k=k, tag=str(tag_g[k]), lam=float(lam_g[k]), eta=float(eta_g[k]),
                     nz=float(np.mean(choices[k] != DP.STATE_ZERO)), **{
                         kk: m[kk] for kk in ('n', 'mpe', 'moe', 'median', 'ok',
                                              'gt1', 'gt5', 'gt10')}))

hdr = ('| lambda | eta | fine% | MPE | MOE | median | ok(1m,2deg) | >1m | >5m |')
print('%-8s %-8s %6s %7s %7s %7s %8s %6s %6s' % ('lam', 'eta', 'fine%', 'MPE', 'MOE',
                                                 'median', 'ok', '>1m', '>5m'))
print('--- references')
for tag, m, _ in ref_rows:
    print('%-28s %6s %7.4f %7.4f %7.4f %7.1f%% %6d %6d'
          % (tag, '-', m['mpe'], m['moe'], m['median'], 100 * m['ok'], m['gt1'], m['gt5']))
for grid in ('spec', 'cal'):
    print('--- %s grid' % grid)
    for r in rows:
        if r['tag'] != grid:
            continue
        print('lam=%-7g eta=%-6g %5.1f%% %7.4f %7.4f %7.4f %7.1f%% %6d %6d'
              % (r['lam'], r['eta'], 100 * r['nz'], r['mpe'], r['moe'], r['median'],
                 100 * r['ok'], r['gt1'], r['gt5']))

best = min(rows, key=lambda r: (r['mpe'], r['gt1']))
print('\n[best by MPE] %s lam=%g eta=%g  MPE %.4f MOE %.4f ok %.1f%% >1m %d >5m %d fine %.1f%%'
      % (best['tag'], best['lam'], best['eta'], best['mpe'], best['moe'], 100 * best['ok'],
         best['gt1'], best['gt5'], 100 * best['nz']))

# --------------------------------------------------------------------------- #
# temporal behaviour of the chosen correction
# --------------------------------------------------------------------------- #
def temporal(ch, tag):
    s, d = DP.chosen_offsets(emis, ch)
    nz = ch != DP.STATE_ZERO
    ds = np.zeros(n); dd = np.zeros(n)
    ds[1:] = np.abs(np.diff(s)); dd[1:] = np.abs(np.diff(d))
    same_date = np.array([i > 0 and keys_seq[i].split('/')[0] == keys_seq[i - 1].split('/')[0]
                          for i in range(n)])
    jump = (ds + dd)[same_date]
    out = dict(tag=tag, nonzero=float(nz.mean()),
               mean_abs=float(np.mean(np.abs(s[nz]) + np.abs(d[nz]))) if nz.any() else 0.0,
               mean_step=float(jump.mean()), median_step=float(np.median(jump)),
               p99_step=float(np.percentile(jump, 99)), var_step=float(jump.var()),
               step_gt1=float(np.mean(jump > 1.0)))
    print('  %-22s nonzero %5.1f%%  mean|c| %.3f  mean|dc| %.4f  median %.4f  p99 %.3f  '
          'var %.4f  share|dc|>1m %.2f%%'
          % (tag, 100 * out['nonzero'], out['mean_abs'], out['mean_step'],
             out['median_step'], out['p99_step'], out['var_step'], 100 * out['step_gt1']))
    return out


print('\n[temporal] correction and its frame-to-frame change (metres)')
t_arg = temporal(argmin_state, 'per-frame argmin')
gate_choice = np.where(gate, argmin_state, DP.STATE_ZERO).astype(np.int16)
t_gate = temporal(gate_choice, 'per-frame gate 0.01')
kbest = best['k']
t_best = temporal(choices[kbest], 'DP lam=%g eta=%g' % (best['lam'], best['eta']))

# --------------------------------------------------------------------------- #
# bursts of front-end error > 2 m
# --------------------------------------------------------------------------- #
e_front = arrays[0]
err_front = np.linalg.norm(front[:, :3, 3] - arrays[0], axis=1)
runs = []
start = None
for i in range(n + 1):
    bad_i = i < n and np.isfinite(err_front[i]) and err_front[i] > 2.0
    if bad_i and start is None:
        start = i
    elif not bad_i and start is not None:
        runs.append((start, i))
        start = None
runs.sort(key=lambda r: -(r[1] - r[0]))
print('\n[bursts] %d runs of front-end error > 2 m; longest %s' % (
    len(runs), [r[1] - r[0] for r in runs[:8]]))
best_poses = poses_for(choices[kbest])
e_best = np.linalg.norm(best_poses[:, :3, 3] - arrays[0], axis=1)
e_gate = np.linalg.norm(np.where(gate[:, None, None], P_seq, front)[:, :3, 3] - arrays[0], axis=1)
print('  %-6s %-12s %-6s %6s %8s %8s %8s %7s %7s'
      % ('burst', 'date', 'n', '>2m', 'front', 'gate', 'DP', 'gate%', 'DP%'))
burst_rows = []
for bi, (a, b) in enumerate(runs[:5]):
    date = keys_seq[a].split('/')[0]
    burst_rows.append(dict(
        burst=bi, date=date, n=b - a, over2=int((err_front[a:b] > 2).sum()),
        front=float(err_front[a:b].mean()), gate=float(e_gate[a:b].mean()),
        dp=float(e_best[a:b].mean()),
        gate_share=float(gate[a:b].mean()),
        dp_share=float((choices[kbest][a:b] != DP.STATE_ZERO).mean())))
    print('  %-6d %-12s %6d %6d %8.2f %8.2f %8.2f %6.0f%% %6.0f%%'
          % (bi, date, b - a, int((err_front[a:b] > 2).sum()), burst_rows[-1]['front'],
             burst_rows[-1]['gate'], burst_rows[-1]['dp'],
             100 * burst_rows[-1]['gate_share'], 100 * burst_rows[-1]['dp_share']))

# --------------------------------------------------------------------------- #
# per date, best config
# --------------------------------------------------------------------------- #
print('\n[per date] best config vs the gate')
for date in [b[0] for b in bounds]:
    sel = np.array([k.split('/')[0] == date for k in keys_seq])
    mg = DP.evaluate([k for k, s in zip(keys_seq, sel) if s], np.where(gate[:, None, None], P_seq, front)[sel], arrays=(
        arrays[0][sel], arrays[1][sel], arrays[2][sel]))
    mb = DP.evaluate([k for k, s in zip(keys_seq, sel) if s], best_poses[sel], arrays=(
        arrays[0][sel], arrays[1][sel], arrays[2][sel]))
    mf = DP.evaluate([k for k, s in zip(keys_seq, sel) if s], front[sel], arrays=(
        arrays[0][sel], arrays[1][sel], arrays[2][sel]))
    print('  %s n=%d  front %.4f  gate %.4f (%d >1m)  DP %.4f (%d >1m)  DP fine %.1f%%'
          % (date, int(sel.sum()), mf['mpe'], mg['mpe'], mg['gt1'], mb['mpe'], mb['gt1'],
             100 * best['nz']))

json.dump(dict(best=best, rows=rows, temporal=[t_arg, t_gate, t_best], bursts=burst_rows,
               refs=[(tag, {k: v for k, v in m.items()
                            if k not in ('per_date', 'err', 'rot', 'date')})
                     for tag, m, _ in ref_rows]),
          open(OUT + '/dp_analysis.json', 'w'), indent=1,
          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

np.savez(P + '/results/dp-offsets.npz', keys=np.array(keys_seq),
         states=choices[best['k']].astype(np.int8), argmin_state=argmin_state.astype(np.int8),
         s=DP.chosen_offsets(emis, choices[best['k']])[0],
         d=DP.chosen_offsets(emis, choices[best['k']])[1],
         poses=best_poses, front=front,
         lam=best['lam'], eta=best['eta'],
         all_states=choices.astype(np.int8), all_lam=lam_g, all_eta=eta_g, all_tag=tag_g)
print('\n[saved] %s/results/dp-offsets.npz and dp_analysis.json' % P)
