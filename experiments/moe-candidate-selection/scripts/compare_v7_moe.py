"""Compare the online front end (V7 stage output, t0) against the geometric branch
output (final pose) on the frames this shard produced.

Both are in the driver's jsonl. Ground truth is the evaluator-only truth (scoring
only; it never enters the decision path). Read-only.
"""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SHARD = 'full-shard0.jsonl'

rows = []
with open(P + SHARD) as fh:
    for line in fh:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('skipped') or not r.get('seconds'):
            continue
        rows.append(r)
print('frames with a produced pose: %d' % len(rows))

truth = {}
for f in os.listdir(TRUTH):
    date = f.replace('.json', '')
    blob = json.load(open(os.path.join(TRUTH, f)))
    for ts, v in blob.items():
        truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
print('truth entries: %d' % len(truth))


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = (np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(np.arccos(c))


def collect(which):
    errs, rots, dates, miss = [], [], [], 0
    for r in rows:
        key = r['key']
        date, ts = key.split('/')
        gt = truth.get((date, ts))
        if gt is None:
            miss += 1
            continue
        T = np.asarray(r[which], np.float64).reshape(4, 4)
        errs.append(float(np.linalg.norm(T[:3, 3] - gt[:3, 3])))
        rots.append(float(rot_deg(T[:3, :3], gt[:3, :3])))
        dates.append(date)
    return np.array(errs), np.array(rots), np.array(dates), miss


def rep(tag, e, r):
    print('%-22s n=%d  MPE=%.4f  MOE=%.4f  median=%.4f  ok(1m,2deg)=%d (%.1f%%)  >5m=%d  >10m=%d'
          % (tag, len(e), e.mean(), r.mean(), np.median(e),
             int(((e <= 1) & (r <= 2)).sum()), 100.0 * ((e <= 1) & (r <= 2)).mean(),
             int((e > 5).sum()), int((e > 10).sum())))


e0, r0, d0, miss0 = collect('t0')
e1, r1, d1, miss1 = collect('pose')
print('truth misses: t0=%d pose=%d' % (miss0, miss1))
rep('V7 front end (t0)', e0, r0)
rep('MoE (final pose)', e1, r1)

print('\nper date:')
for date in sorted(set(d0)):
    m = d0 == date
    rep('  V7  ' + date, e0[m], r0[m])
    rep('  MoE ' + date, e1[m], r1[m])

d = e1 - e0
print('\nchange: better on %d (%.1f%%), worse on %d (%.1f%%), |diff|<=0.2 m on %d'
      % (int((d < -0.2).sum()), 100.0 * (d < -0.2).mean(),
         int((d > 0.2).sum()), 100.0 * (d > 0.2).mean(), int((np.abs(d) <= 0.2).sum())))
print('worst regressions: %d frames worse by >2 m, %d worse by >10 m'
      % (int((d > 2).sum()), int((d > 10).sum())))
