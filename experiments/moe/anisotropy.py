"""Per-direction residual structure of the front end, and whether the correction helps there.

Supports (or refutes) the anisotropic-prior framing:
  - how the position error splits into along-track / lateral / vertical, per date
  - whether the along-track direction is also the one the map can correct
  - whether local observability (the along-track curvature of the cost surface) is
    present where the residual is large
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/v7-alternating-e7-full-20261018-v6')
from nclt_reference_policy import apply_reference_policy

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
T = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
POLICY = '/home/zhangcy/projects/v7-alternating-e7-full-20261018-v6/nclt-reference-policy.json'

truth = {}
for f in os.listdir(T):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(T, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], float)
Ps = np.asarray(z['P_search'], float)
gain = np.asarray(z['gain'], float)
final = np.load(os.path.join(ROOT, 'results/full-online-final.npz'), allow_pickle=True)
fk = [str(k) for k in final['keys']]
fp = np.asarray(final['poses'], float)
fi = {k: j for j, k in enumerate(fk)}
Pf = np.stack([fp[fi[k]] for k in keys])

dates = np.array([k.split('/')[0] for k in keys])
tus = np.array([int(k.split('/')[1]) for k in keys])
gt = np.stack([truth[(k.split('/')[0], k.split('/')[1])] for k in keys])
ref, _, _ = apply_reference_policy(dates, tus, gt, POLICY)


def decompose(pose, target):
    """Position error split along the pose's own axes, in metres."""
    e = target[:3, 3] - pose[:3, 3]
    R = pose[:3, :3]
    return np.array([e @ R[:, 0], e @ R[:, 1], e @ R[:, 2]])


def rot_deg(a, b):
    c = np.clip((np.einsum('ij,ij->', a, b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


print('=== per-direction residual of the front end (replaced truth) ===')
print('%-12s %6s %8s %8s %8s %8s %8s' %
      ('date', 'n', '|along|', '|lat|', '|vert|', 'rot deg', 'along share'))
for day in list(np.unique(dates)) + ['ALL']:
    m = np.ones(len(keys), bool) if day == 'ALL' else (dates == day)
    comp = np.array([decompose(t0[i], ref[i]) for i in np.flatnonzero(m)])
    a = np.abs(comp[:, 0]).mean(); l = np.abs(comp[:, 1]).mean(); v = np.abs(comp[:, 2]).mean()
    r = np.mean([rot_deg(t0[i][:3, :3], ref[i][:3, :3]) for i in np.flatnonzero(m)])
    share = a / (a + l + v)
    print('%-12s %6d %8.3f %8.3f %8.3f %8.3f %8.1f%%' % (day, m.sum(), a, l, v, r, 100 * share))

print('\n=== after the geometric correction (final online run) ===')
for day in list(np.unique(dates)) + ['ALL']:
    m = np.ones(len(keys), bool) if day == 'ALL' else (dates == day)
    comp = np.array([decompose(Pf[i], ref[i]) for i in np.flatnonzero(m)])
    a = np.abs(comp[:, 0]).mean(); l = np.abs(comp[:, 1]).mean(); v = np.abs(comp[:, 2]).mean()
    r = np.mean([rot_deg(Pf[i][:3, :3], ref[i][:3, :3]) for i in np.flatnonzero(m)])
    print('%-12s %6d %8.3f %8.3f %8.3f %8.3f' % (day, m.sum(), a, l, v, r))

# does the search help exactly where the along-track residual is large?
trig = gain > 0.01
comp0 = np.array([decompose(t0[i], ref[i]) for i in range(len(keys))])
along = np.abs(comp0[:, 0])
e0 = np.linalg.norm(t0[:, :3, 3] - ref[:, :3, 3], axis=1)
e1 = np.linalg.norm(Ps[:, :3, 3] - ref[:, :3, 3], axis=1)
print('\n=== where does the search help? (triggered frames, %d) ===' % trig.sum())
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 1e9)):
    m = trig & (along >= lo) & (along < hi)
    if m.sum() < 20:
        continue
    d = e1[m] - e0[m]
    print('  along %.1f-%.1f m : %5d frames, helped %4.1f%%, mean change %+0.3f m'
          % (lo, hi, m.sum(), 100 * (d < -0.01).mean(), d.mean()))

# local observability: the along-track curvature of the coarse surface
rows = []
for p in glob.glob('/home/zhangcy/.dsh_scratch/obs/*.jsonl'):
    for line in open(p):
        rows.append(json.loads(line))
idx = {k: i for i, k in enumerate(keys)}
curv = []
for r in rows:
    i = idx.get(r['key'])
    if i is None or not trig[i]:
        continue
    curv.append((float(r.get('surface_curv', np.nan)), e1[i] - e0[i], along[i]))
curv = np.array(curv)
ok = np.isfinite(curv[:, 0])
print('\n=== observability (along-track curvature) vs usefulness ===')
for lo, hi in ((-1e9, 0.0), (0.0, 0.002), (0.002, 0.01), (0.01, 1e9)):
    m = ok & (curv[:, 0] >= lo) & (curv[:, 0] < hi)
    if m.sum() < 20:
        continue
    print('  curvature %8.4f..%8.4f : %5d frames, helped %4.1f%%, mean along residual %.2f m'
          % (lo, hi, m.sum(), 100 * (curv[m, 1] < -0.01).mean(), curv[m, 2].mean()))
