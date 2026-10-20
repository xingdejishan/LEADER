"""Probe 6: is the deployed HEADING a better along-track direction than a position fit?
No cost is evaluated here; only angles against the truth-trajectory tangent."""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

w = L.World()
n = len(w.keys)
src = Counter()
ang_pos, ang_head, ang_gt_head = [], [], []
mask = []
for i in range(n):
    ut, mt = w.u_truth(i)
    src[mt['source']] += 1
    up, mp = w.u_deployable(i)
    uh = w.deploy[i, :3, :3] @ np.array([1.0, 0.0, 0.0])
    uh = w._orient(uh / np.linalg.norm(uh), i)
    ug = w.gt[i, :3, :3] @ np.array([1.0, 0.0, 0.0])
    ug = w._orient(ug / np.linalg.norm(ug), i)

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1, 1))))
    ang_pos.append(ang(up, ut))
    ang_head.append(ang(uh, ut))
    ang_gt_head.append(ang(ug, ut))
    mask.append(mt['source'] == 'truth_quad')
ang_pos = np.asarray(ang_pos)
ang_head = np.asarray(ang_head)
ang_gt_head = np.asarray(ang_gt_head)
mask = np.asarray(mask)
print('truth arm sources:', dict(src))


def show(name, a, m):
    a, m = np.asarray(a), np.asarray(m)
    print('%-22s all: med=%5.2f p75=%5.2f p90=%5.2f | on truth_quad frames (n=%d): med=%5.2f p75=%5.2f p90=%5.2f'
          % (name, np.median(a), np.percentile(a, 75), np.percentile(a, 90),
             m.sum(), np.median(a[m]), np.percentile(a[m], 75), np.percentile(a[m], 90)))
    for thr in (2, 5, 10, 20):
        print('      <=%2d deg on truth_quad: %.1f%%' % (thr, 100.0 * (a[m] <= thr).mean()))


show('position quad', ang_pos, mask)
show('deployed heading', ang_head, mask)
show('truth heading (ref)', ang_gt_head, mask)
print()
print('deployed heading vs truth heading: med=%.2f p90=%.2f'
      % (np.median(ang_head - ang_head), 0))
corr = np.corrcoef(ang_pos[mask], ang_head[mask])[0, 1]
print('corr(position-quad err, heading err) = %.3f' % corr)
# how often does heading beat the position fit?
print('heading closer than position fit: %d/%d' % (int((ang_head[mask] < ang_pos[mask]).sum()), int(mask.sum())))
