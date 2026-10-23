"""Assemble the recommended configuration from the sweep and score it on the full set.

recommended: gate2 = 0.01 (run the search only where descent depth exceeds it),
rotation accepted when the held-out cost improves by more than eps = 0.01,
and rotation applied only to the frames that actually ran the search.
"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
GATE = 0.01
EPS = 0.01

z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
gain = np.asarray(z['gain'], float)
t0 = np.asarray(z['t0'], float)
P_search = np.asarray(z['P_search'], float)
P_rot = np.asarray(z['P_rot'], float)
eps = np.asarray(z['eps'], float)
bad = np.asarray(z['bad'], bool)
idx = int(np.argmin(np.abs(eps - EPS)))
print('eps grid %s -> using %s (index %d)' % (list(eps), eps[idx], idx))

run = (gain > GATE) & (~bad)
final = t0.copy()
final[run] = P_rot[run, idx]
print('frames: %d   search ran: %d (%.1f%%)   skipped(bad): %d'
      % (len(keys), int(run.sum()), 100.0 * run.mean(), int(bad.sum())))

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


rows = []
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth.get((d, ts))
    if g is None:
        continue
    rows.append((d, i, np.linalg.norm(t0[i][:3, 3] - g[:3, 3]),
                 np.linalg.norm(final[i][:3, 3] - g[:3, 3]),
                 rot_deg(t0[i][:3, :3], g[:3, :3]), rot_deg(final[i][:3, :3], g[:3, :3])))
print('scored frames: %d' % len(rows))


def rep(tag, sel):
    e0 = np.array([r[2] for r in rows if sel(r)])
    e1 = np.array([r[3] for r in rows if sel(r)])
    f0 = np.array([r[4] for r in rows if sel(r)])
    f1 = np.array([r[5] for r in rows if sel(r)])
    print('%-18s n=%6d  MPE %.4f -> %.4f   MOE %.4f -> %.4f   ok %.1f%% -> %.1f%%   '
          '>1m %d -> %d   >5m %d -> %d'
          % (tag, len(e0), e0.mean(), e1.mean(), f0.mean(), f1.mean(),
             100 * ((e0 <= 1) & (f0 <= 2)).mean(), 100 * ((e1 <= 1) & (f1 <= 2)).mean(),
             int((e0 > 1).sum()), int((e1 > 1).sum()), int((e0 > 5).sum()), int((e1 > 5).sum())))


rep('all', lambda r: True)
for d in sorted(set(r[0] for r in rows)):
    rep(d, lambda r, d=d: r[0] == d)

np.savez(P + 'full-online-final.npz', keys=np.array(keys), poses=final, t0=t0,
         gain=gain, searched=run)
print('wrote %sfull-online-final.npz' % P)
