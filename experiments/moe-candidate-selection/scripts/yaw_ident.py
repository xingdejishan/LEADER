"""Is the yaw tail a gross-error problem or a missing-constraint problem?

Part 1  cross-tab: yaw error vs translation error over all 1502 frames.
Part 2  for the worst-yaw frames, park the pose at the TRUE translation and sweep
        body yaw over +-15 deg on the reference map.  Three readings:
          identifiable  the map prefers the true yaw and the dip is sharp
          flat          the cost barely changes with yaw -> no rotational evidence there
          inverted      the map prefers a yaw far from the truth -> evidence points the wrong way
"""
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import pipeline as P          # noqa: E402
import fast_lib as F          # noqa: E402

spec = importlib.util.spec_from_file_location(
    'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
gt = np.asarray(w['hard_gt'], float)
base = np.asarray(w['hard_base'], float)

Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
yaw_err = np.abs(rv[:, 2])
full = np.linalg.norm(rv, axis=1)
trans_err = np.linalg.norm(base[:, :3, 3] - gt[:, :3, 3], axis=1)

print('== part 1: joint distribution ==')
print('frames with |yaw| > 8 deg : n=%d' % (yaw_err > 8).sum())
for lo, hi in ((0, 0.5), (0.5, 2), (2, 10), (10, 50), (50, 1e9)):
    m = (yaw_err > 8) & (trans_err >= lo) & (trans_err < hi)
    if m.sum():
        print('   translation error %-8s : %3d frames  (%.0f%% of the yaw tail)'
              % ('%g-%gm' % (lo, hi if hi < 1e8 else 999), m.sum(), 100 * m.sum() / (yaw_err > 8).sum()))
m = (yaw_err > 8)
print('   median translation error in the yaw tail: %.2f m   (all frames: %.2f m)'
      % (np.median(trans_err[m]), np.median(trans_err)))
print('   frames with translation error < 2 m but |yaw| > 8 deg: %d' % int((m & (trans_err < 2)).sum()))

# ---- part 2: yaw sweep at the true translation
frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
cfg = P.Config()
maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                   radius=cfg.crop_radius)

TAIL = list(np.flatnonzero(yaw_err > 8))
rng = np.random.default_rng(7)
CTRL = list(rng.choice(np.flatnonzero(full < 0.7), 15, replace=False))
D = np.arange(-15.0, 15.001, 1.0)


def sweep(i):
    t = gt[i][:3, 3]
    mp = maps.local(t, maps.nearby(t)[:cfg.reference_count])
    if len(mp) < 1000:
        return None
    tree = cKDTree(mp)
    scan = P.prepare_scan(frames[i], cfg)
    q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]
    costs = []
    for d in D:
        R = gt[i][:3, :3] @ Rotation.from_euler('z', d, degrees=True).as_matrix()
        W = q @ R.T + t
        costs.append(float(np.minimum(tree.query(W, k=1, workers=1)[0], 1.0).mean()))
    costs = np.array(costs)
    best = int(np.argmin(costs))
    at0 = float(costs[D == 0][0])
    # sharpness: how much worse 5 deg off the truth is
    sharp = float(np.mean([costs[D == s][0] for s in (-5.0, 5.0)]) - at0)
    pred_yaw = np.degrees(Rotation.from_matrix(gt[i][:3, :3].T @ base[i][:3, :3]).as_rotvec())
    res = -1e9
    score_pred = float(np.interp(pred_yaw[2], D, costs))
    return dict(i=i, best_d=float(D[best]), at0=at0, best=costs[best], sharp=sharp,
                pred_yaw=float(pred_yaw[2]), cost_at_pred=score_pred, cost_min=costs.min(),
                better_than_truth=(costs.min() < at0 - 0.002))


print('\n== part 2: yaw sweep at the true translation ==')
print('%-6s %8s %8s %9s %9s %9s %9s  %s' %
      ('frame', 'predyaw', 'argmin', 'c(truth)', 'c(best)', 'sharp5', 'yrelerr', 'reading'))
rows = []
for tag, group in (('TAIL', TAIL), ('CTRL', CTRL)):
    print('--- %s ---' % tag)
    for i in group:
        r = sweep(i)
        if r is None:
            continue
        if abs(r['best_d']) <= 2 and r['sharp'] > 0.01:
            reading = 'identifiable'
        elif r['sharp'] <= 0.01:
            reading = 'flat'
        else:
            reading = 'inverted'
        rows.append((tag, reading, r['sharp'], r['at0'], r['cost_min']))
        print('%-6d %8.1f %8.1f %9.4f %9.4f %9.4f %9.1f  %s' %
              (i, r['pred_yaw'], r['best_d'], r['at0'], r['cost_min'], r['sharp'], full[i], reading))

print('\n== summary ==')
for tag in ('TAIL', 'CTRL'):
    sub = [x for x in rows if x[0] == tag]
    if not sub:
        continue
    from collections import Counter
    c = Counter(x[1] for x in sub)
    print('%-5s n=%2d  %s   median sharp %.4f' %
          (tag, len(sub), dict(c), float(np.median([x[2] for x in sub]))))
