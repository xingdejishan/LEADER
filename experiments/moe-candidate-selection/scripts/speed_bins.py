import json, os
import numpy as np
from scipy.spatial.transform import Rotation

spec_path = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
import importlib.util
spec = importlib.util.spec_from_file_location('su', spec_path)
su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)
w = su.load_world()
gt = np.asarray(w['hard_gt'], float); pred = np.asarray(w['hard_base'], float)
dates = np.asarray(w['dates_hard'])
Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
err = np.linalg.norm(rv, axis=1)

stamps = np.array([int(str(k).split('/')[1]) for k in w['hard_keys']], dtype=np.int64)
order = np.lexsort((stamps, dates))
speed = np.full(len(err), np.nan)
for a, b in zip(order[:-1], order[1:]):
    if dates[a] != dates[b]:
        continue
    dt = (stamps[b] - stamps[a]) / 1e6
    if 0 < dt <= 2.0:
        speed[b] = float(np.linalg.norm(pred[b, :3, 3] - pred[a, :3, 3]) / dt)

ok = np.isfinite(speed)
e, s = err[ok], speed[ok]
q = np.quantile(s, [0.25, 0.5, 0.75])
edges = [-1, q[0], q[1], q[2], 1e9]
names = ['q1 slowest', 'q2', 'q3', 'q4 fastest']
print('%-12s %6s %9s %9s %9s %9s' % ('speed bin', 'n', 'speed lo', 'speed hi', 'mean err', 'median err'))
for k in range(4):
    m = (s > edges[k]) & (s <= edges[k + 1])
    if m.sum() == 0:
        continue
    print('%-12s %6d %9.2f %9.2f %9.3f %9.3f'
          % (names[k], int(m.sum()), s[m].min(), s[m].max(), e[m].mean(), np.median(e[m])))
print()
print('overall n=%d  mean %.3f' % (len(e), e.mean()))
