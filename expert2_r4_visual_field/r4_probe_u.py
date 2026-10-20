"""Probe 2 (no cost grid): u sources, u vs truth-tangent angle, window spans."""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

w = L.World()
n = len(w.keys)
src = Counter()
angs = []
spans = []
short = 0
rows = []
for i in range(n):
    u, meta = w.u_deployable(i)
    src[meta['source']] += 1
    if meta['source'] == 'deploy_neighbour':
        spans.append(meta['span'])
    ut, mt = w.u_truth(i)
    ang = float(np.degrees(np.arccos(np.clip(abs(float(u @ ut)), -1, 1))))
    angs.append(ang)
    if meta['source'] != 'deploy_neighbour' and meta['source'] != 'ref_tangent':
        short += 1
    rows.append((i, w.keys[i], meta['source'], meta.get('k', 0),
                 round(meta.get('span', 0.0), 2), round(ang, 2)))
angs = np.asarray(angs)
spans = np.asarray(spans)
print('u source:', dict(src))
print('window span (deploy): med=%.2f p10=%.2f p90=%.2f max=%.2f'
      % (np.median(spans), np.percentile(spans, 10), np.percentile(spans, 90), spans.max()))
print('angle(deploy u, truth tangent u) deg: med=%.2f p75=%.2f p90=%.2f p99=%.2f max=%.2f'
      % (np.median(angs), np.percentile(angs, 75), np.percentile(angs, 90),
         np.percentile(angs, 99), angs.max()))
for thr in (1, 2, 5, 10, 20, 45):
    print('  <=%2d deg: %d (%.1f%%)' % (thr, int((angs <= thr).sum()), 100.0 * (angs <= thr).mean()))
print('worst 12:')
for r in sorted(rows, key=lambda x: -x[5])[:12]:
    print('   idx=%4d %s src=%s k=%s span=%s ang=%.1f' % r)
print('ref_tangent frames:', [r[:6] for r in rows if r[2] == 'ref_tangent'][:10])
