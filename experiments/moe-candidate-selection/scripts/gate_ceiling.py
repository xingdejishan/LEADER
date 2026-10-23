"""How many frames did the geometric branch bring inside 1 m, and what could a
suitable gate achieve? (read-only; gate simulation assumes skipped frames keep t0)"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']

z = np.load(P + 'full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)
gain = {}
for name in FILES:
    with open(P + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('key') and r.get('seconds'):
                gain[r['key']] = r.get('gain')

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


n = len(keys)
e0 = np.empty(n); e1 = np.empty(n); r0 = np.empty(n); r1 = np.empty(n)
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth[(d, ts)]
    e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
    e1[i] = np.linalg.norm(fin[i][:3, 3] - g[:3, 3])
    r0[i] = rot_deg(t0[i][:3, :3], g[:3, :3])
    r1[i] = rot_deg(fin[i][:3, :3], g[:3, :3])

G = np.array([gain.get(k) if gain.get(k) is not None else np.nan for k in keys], float)

print('=== 1. effect on the 1 m criterion ===')
print('position: inside 1 m  front end %d -> MoE %d'
      % (int((e0 <= 1).sum()), int((e1 <= 1).sum())))
print('  brought in  (was >1 m, now <=1 m): %d' % int(((e0 > 1) & (e1 <= 1)).sum()))
print('  pushed out  (was <=1 m, now >1 m): %d' % int(((e0 <= 1) & (e1 > 1)).sum()))
ok0 = (e0 <= 1) & (r0 <= 2)
ok1 = (e1 <= 1) & (r1 <= 2)
print('composite (1 m and 2 deg): %d -> %d' % (int(ok0.sum()), int(ok1.sum())))
print('  brought in: %d   pushed out: %d'
      % (int((~ok0 & ok1).sum()), int((ok0 & ~ok1).sum())))

print('\n=== 2. ceilings with a gate (binary choice between the two stored poses) ===')
print('no gate (all search)   MPE=%.4f MOE=%.4f  ok=%d' % (e1.mean(), r1.mean(), int(ok1.sum())))
print('no search at all       MPE=%.4f MOE=%.4f  ok=%d' % (e0.mean(), r0.mean(), int(ok0.sum())))
orc_e = np.minimum(e0, e1)
orc_r = np.minimum(r0, r1)
print('oracle per-frame (MPE) MPE=%.4f  ok=%d (choosing by position)'
      % (orc_e.mean(), int(((orc_e <= 1) & (r1 <= 2)).sum())))
print('oracle per-frame (MOE) MOE=%.4f' % orc_r.mean())

print('\n--- gate by descent depth (skipped frames keep t0) ---')
best = None
for thr in (0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05):
    m = np.isfinite(G) & (G > thr)
    E = np.where(m, e1, e0)
    R = np.where(m, r1, r0)
    ok = int(((E <= 1) & (R <= 2)).sum())
    print('  thr=%.3f  runs %6d (%.1f%%)  MPE=%.4f  MOE=%.4f  ok=%d'
          % (thr, int(m.sum()), 100.0 * m.mean(), E.mean(), R.mean(), ok))
    if best is None or E.mean() < best[1]:
        best = (thr, E.mean(), R.mean(), ok)
print('best threshold by MPE: %.3f -> MPE=%.4f MOE=%.4f ok=%d' % best)
