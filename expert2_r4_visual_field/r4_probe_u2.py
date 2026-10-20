"""Probe 2 (no cost grid): u source mix and angle against the truth-trajectory tangent."""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

w = L.World()
n = len(w.keys)
src = Counter()
angs, chords = [], []
rows = []
for i in range(n):
    u, meta = w.u_deployable(i)
    src[meta['source']] += 1
    if meta['source'] == 'deploy_quad':
        chords.append(meta['chord'])
    ut, mt = w.u_truth(i)
    ang = float(np.degrees(np.arccos(np.clip(abs(float(u @ ut)), -1, 1))))
    angs.append(ang)
    rows.append((i, w.keys[i], meta['source'], mt['source'],
                 round(meta.get('chord', 0.0), 2), round(ang, 2)))
angs = np.asarray(angs)
chords = np.asarray(chords)
print('u source (deployable):', dict(src))
print('truth arm source:', dict(Counter(r[3] for r in rows)))
print('window chord: med=%.1f p10=%.1f p90=%.1f' % (np.median(chords),
                                                   np.percentile(chords, 10),
                                                   np.percentile(chords, 90)))
print('angle(deploy u, truth-tangent u) deg: med=%.2f p75=%.2f p90=%.2f p99=%.2f max=%.2f'
      % (np.median(angs), np.percentile(angs, 75), np.percentile(angs, 90),
         np.percentile(angs, 99), angs.max()))
for thr in (1, 2, 5, 10, 20, 45):
    print('  <=%2d deg: %d (%.1f%%)' % (thr, int((angs <= thr).sum()), 100.0 * (angs <= thr).mean()))
print('worst 10:')
for r in sorted(rows, key=lambda x: -x[5])[:10]:
    print('   idx=%4d %s dep=%s truth=%s chord=%s ang=%.1f' % r)
