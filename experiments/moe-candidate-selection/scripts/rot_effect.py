"""Does the module degrade orientation? Distribution of the per-frame change (read-only)."""
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

z = np.load(P + 'full-online.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items() if False else []:
        pass
import json
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


n = len(keys)
e0 = np.empty(n); r0 = np.empty(n); e1 = np.empty(n); r1 = np.empty(n)
for i, k in enumerate(keys):
    d, ts = k.split('/')
    g = truth[(d, ts)]
    e0[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
    e1[i] = np.linalg.norm(fin[i][:3, 3] - g[:3, 3])
    r0[i] = rot_deg(t0[i][:3, :3], g[:3, :3])
    r1[i] = rot_deg(fin[i][:3, :3], g[:3, :3])

dr = r1 - r0
print('MOE: front end %.4f -> final %.4f  (mean change %+.4f deg)'
      % (r0.mean(), r1.mean(), dr.mean()))
print('rotation improved (>0.1 deg): %d (%.1f%%), worsened: %d (%.1f%%), flat: %d'
      % (int((dr < -0.1).sum()), 100.0 * (dr < -0.1).mean(),
         int((dr > 0.1).sum()), 100.0 * (dr > 0.1).mean(), int((np.abs(dr) <= 0.1).sum())))
print('median change %+.3f deg, p90 of |change| %.3f deg'
      % (np.median(dr), np.percentile(np.abs(dr), 90)))

print('\nby front-end orientation error:')
for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 90)):
    m = (r0 >= lo) & (r0 < hi)
    if m.sum() == 0:
        continue
    print('  rot %.0f-%.0f deg: n=%6d  MOE %.3f -> %.3f  (%+.3f)  helped %d  hurt %d'
          % (lo, hi, int(m.sum()), r0[m].mean(), r1[m].mean(), dr[m].mean(),
             int((dr[m] < -0.1).sum()), int((dr[m] > 0.1).sum())))

print('\nby front-end position error:')
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 2), (2, 1e9)):
    m = (e0 >= lo) & (e0 < hi)
    if m.sum() == 0:
        continue
    print('  pos %.1f-%.1f m: n=%6d  MOE %.3f -> %.3f  (%+.3f)'
          % (lo, min(hi, 99), int(m.sum()), r0[m].mean(), r1[m].mean(), dr[m].mean()))
