"""What each rotation axis is worth: recompute the mean angle with one axis zeroed."""
import importlib.util

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location(
    'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
gt = np.asarray(w['hard_gt'], float)
base = np.asarray(w['hard_base'], float)

Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
full = np.linalg.norm(rv, axis=1)


def moe(drop):
    keep = [i for i in range(3) if i not in drop]
    return np.linalg.norm(rv[:, keep], axis=1).mean()


print('baseline MOE                       : %.4f deg' % full.mean())
print('if the yaw error alone were removed: %.4f deg  (-%.4f)' % (moe({2}), full.mean() - moe({2})))
print('if roll+pitch were removed         : %.4f deg  (-%.4f)' % (moe({0, 1}), full.mean() - moe({0, 1})))
print('if roll alone were removed         : %.4f deg  (-%.4f)' % (moe({0}), full.mean() - moe({0})))
print('if pitch alone were removed        : %.4f deg  (-%.4f)' % (moe({1}), full.mean() - moe({1})))
print()
big = full > 8
print('tail (>8 deg, n=%d): mean %.1f deg, of which yaw %.1f, roll %.1f, pitch %.1f'
      % (big.sum(), full[big].mean(), np.abs(rv[big, 2]).mean(),
         np.abs(rv[big, 0]).mean(), np.abs(rv[big, 1]).mean()))
print('share of the summed angle over all frames contributed by those %d frames: %.1f%%'
      % (big.sum(), 100 * full[big].sum() / full.sum()))
print('share contributed by the 84 gated frames: see next line')
import json
ratio = np.array([json.loads(l)['ratio'] for l in
                  open('/home/zhangcy/projects/e2e-moe-20261020-v1/results/three_route_diag.jsonl')])
g = np.isfinite(ratio) & (ratio < 0.21)
print('  gated frames n=%d  mean angle %.2f deg  share of summed angle %.1f%%'
      % (g.sum(), full[g].mean(), 100 * full[g].sum() / full.sum()))
