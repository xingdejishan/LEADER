"""Probe 1 (no scan loading): frame table, reference library, deployed-pose neighbour spacing."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

HARD = Path('/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1')
z = np.load(HARD / 'scored.npz', allow_pickle=True)
print('scored fields:', {k: (z[k].shape, z[k].dtype) for k in z.files})
keys = [str(k) for k in z['keys']]
dates = np.asarray(z['dates']).astype(str)
repl = np.asarray(z['replaced']).astype(bool)
print('n=%d dates=%d replaced=%d' % (len(keys), len(set(dates)), repl.sum()))
print('per date:', sorted(Counter(dates.tolist()).items()))
print('first keys:', keys[:3], 'last:', keys[-1])

q = json.load(open(HARD / 'selection.json'))
qs = q['queries']
print('queries n=%d, keys sample=%s' % (len(qs), list(qs[0].keys())))
qmap = {}
for r in qs:
    k = r.get('key')
    qmap[str(k)] = r
print('all hard keys in queries:', all(k in qmap for k in keys))
r0 = qmap[keys[0]]
print('query[0]:', {k: v for k, v in r0.items() if k != 'scan_sha256'}, r0['scan_sha256'][:12])

refs = q['references']
print('refs n=%d, fields=%s' % (len(refs), list(refs[0].keys())))
rdates = Counter(str(r['key']).split('/')[0] for r in refs)
print('ref dates:', sorted(rdates.items()))
R = np.stack([np.asarray(r['world_from_body'], float) for r in refs])
print('ref positions bbox:', R[:, :3, 3].min(0).round(1), R[:, :3, 3].max(0).round(1))

# deployed baseline
dep = np.load('/home/zhangcy/projects/r-unified-eval-20261018-v1/deployed_baseline.npz',
              allow_pickle=True)
print('deployed fields:', dep.files, {k: dep[k].shape for k in dep.files})
dkeys = [str(k) for k in dep['keys']]
print('deployed keys == hard keys:', dkeys == keys)
P = np.asarray(dep['refined_V7'], float)

gt = np.asarray(z['reference'], float)
err = np.linalg.norm(P[:, :3, 3] - gt[:, :3, 3], axis=1)
print('deployed pos err: mean=%.4f median=%.4f  >10m=%d  >100m=%d'
      % (err.mean(), np.median(err), (err > 10).sum(), (err > 100).sum()))
c = (P[:, :3, :3].transpose(0, 2, 1) @ gt[:, :3, :3])
ang = np.degrees(np.arccos(np.clip((np.trace(c, axis1=1, axis2=2) - 1) / 2, -1, 1)))
print('deployed rot err: mean=%.3f' % ang.mean())
print('success 1m2deg=%d, 3m8deg=%d' % (((err <= 1) & (ang <= 2)).sum(),
                                        ((err <= 3) & (ang <= 8)).sum()))
print('excl-suspect 1m2deg=%d n=%d' % (((err <= 1) & (ang <= 2) & ~repl).sum(), (~repl).sum()))
print('damage reference: err>10m frames:', np.flatnonzero(err > 10)[:10])

# neighbour spacing of deployed poses inside each date
print('\n--- deployed-pose neighbour spacing (sorted by timestamp within date) ---')
spans = []
for d in sorted(set(dates)):
    idx = np.flatnonzero(dates == d)
    order = idx[np.argsort([int(keys[i].split('/')[1]) for i in idx])]
    ts = np.array([int(keys[i].split('/')[1]) for i in order])
    pos = P[order, :3, 3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(ts) * 1e-6
    print('%s n=%4d  |step| med=%.2f p90=%.2f max=%.2f   dt med=%.2fs max=%.1fs  max gap flag=%d'
          % (d, len(order), np.median(step), np.percentile(step, 90), step.max(),
             np.median(dt), dt.max(), int((step > 15).sum())))
    spans.append(step)
step = np.concatenate(spans)
print('overall step: med=%.2f p90=%.2f p99=%.2f max=%.2f  >15m=%d'
      % (np.median(step), np.percentile(step, 90), np.percentile(step, 99), step.max(),
         int((step > 15).sum())))
print('gt per-frame speed (m between adjacent truth poses):')
gtstep = []
for d in sorted(set(dates)):
    idx = np.flatnonzero(dates == d)
    order = idx[np.argsort([int(keys[i].split('/')[1]) for i in idx])]
    gtstep.append(np.linalg.norm(np.diff(gt[order, :3, 3], axis=0), axis=1))
g = np.concatenate(gtstep)
print('gt step med=%.2f p90=%.2f max=%.2f' % (np.median(g), np.percentile(g, 90), g.max()))
