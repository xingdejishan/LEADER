"""Read-only: score one or more pose tables with the frozen unified scorer."""
import importlib.util
import sys

import numpy as np

SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
hard_keys, gt = w['hard_keys'], w['hard_gt']

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    P = np.asarray(z['poses'], float)
    order = [keys.index(k) for k in hard_keys]
    A = P[order]
    e = su.pos_err(A, gt)
    r = su.rot_deg(A[:, :3, :3], gt[:, :3, :3])
    name = path.split('results/')[-1]
    print('%-32s MPE=%.4f  MOE=%.4f  ok(1m,2deg)=%4d  >10m=%d  >5m=%d  median=%.3f'
          % (name, e.mean(), r.mean(), int(((e <= 1) & (r <= 2)).sum()),
             int((e > 10).sum()), int((e > 5).sum()), float(np.median(e))))
