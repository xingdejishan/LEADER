"""Why don't the 1-8 m frames all land inside 1 m? (read-only)"""
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

sd = {}
for name in FILES:
    with open(P + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('key') and r.get('seconds'):
                sd[r['key']] = (r.get('s'), r.get('d'), r.get('expanded'))

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
S = np.full(n, np.nan); D = np.full(n, np.nan)
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth[(d, ts)]
    e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
    e1[i] = np.linalg.norm(fin[i][:3, 3] - g[:3, 3])
    r0[i] = rot_deg(t0[i][:3, :3], g[:3, :3])
    r1[i] = rot_deg(fin[i][:3, :3], g[:3, :3])
    s, d, _ = sd.get(k, (None, None, None))
    if s is not None:
        S[i] = s
    if d is not None:
        D[i] = d

for lo, hi in ((1, 2), (2, 5), (5, 8)):
    m = (e0 >= lo) & (e0 < hi)
    if not m.sum():
        continue
    inside = int((e1[m] <= 1).sum())
    print('\nfront-end error %.0f-%.0f m: n=%d' % (lo, hi, int(m.sum())))
    print('  after: inside 1 m %d (%.0f%%), median residual %.3f m, p90 %.3f m'
          % (inside, 100.0 * inside / m.sum(), np.median(e1[m]), np.percentile(e1[m], 90)))
    print('  front-end rotation error in this band: median %.2f deg, p90 %.2f deg'
          % (np.median(r0[m]), np.percentile(r0[m], 90)))
    fix = e1[m] <= 1
    if (~fix).any():
        print('  fixed vs not: rotation error %.2f vs %.2f deg (median)'
              % (np.median(r0[m][fix]) if fix.any() else np.nan, np.median(r0[m][~fix])))
    print('  |s| at the +-8 m edge: %d, |d| at the +-2 m edge: %d, mean windows %.2f'
          % (int((np.abs(S[m]) >= 7.99).sum()), int((np.abs(D[m]) >= 1.99).sum()),
             float(np.nanmean([sd.get(k, (0, 0, 0))[2] or 0 for k, mm in zip(keys, m) if mm]))))

print('\ncorrelation between front-end rotation error and residual position error, '
      'frames with t0 error > 1 m:')
m = e0 > 1
print('  corr(r0, e1) = %.3f   corr(e0, e1) = %.3f'
      % (np.corrcoef(r0[m], e1[m])[0, 1], np.corrcoef(e0[m], e1[m])[0, 1]))
