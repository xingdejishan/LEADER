"""Is the correct yaw basin reachable from where the pipeline actually is?

For every yaw-tail frame (|yaw error| > 8 deg) and a control set:

  A  run the current no-rotation pipeline on that frame -> its real output pose T3
  B  sweep body yaw at T3's translation over +-40 deg (2 deg)   -> can a local sweep
     from the predicted yaw reach the true yaw?
  C  sweep body yaw at the TRUE translation over +-180 deg (4 deg) -> where is the
     global minimum, is the truth the global basin, is there a competing one?

Verdict per frame: in_window (|dyaw| <= 15), improves (cost at truth < cost at pred
at T3's position), identifiable (truth is a local min at T3's position).
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

import run_pipeline as RP      # noqa: E402
import pipeline as P           # noqa: E402

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

RP._OPTS.update(gate2=0.01)
RP.init()

TAIL = list(np.flatnonzero(yaw_err > 8))
rng = np.random.default_rng(7)
CTRL = list(rng.choice(np.flatnonzero(full < 0.7), 12, replace=False))
D_LOCAL = np.arange(-40.0, 40.001, 2.0)
D_GLOBAL = np.arange(-180.0, 180.001, 4.0)


def curve(tree, q, t, R0, grid):
    out = np.empty(len(grid))
    for k, d in enumerate(grid):
        R = R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix()
        W = q @ R.T + t
        out[k] = float(np.minimum(tree.query(W, k=1, workers=1)[0], 1.0).mean())
    return out


def evaluate(i):
    _, T3, diag = RP.one(i)
    t3 = T3[:3, 3]
    tg = gt[i][:3, 3]
    yaw3 = np.degrees(Rotation.from_matrix(gt[i][:3, :3].T @ T3[:3, :3]).as_rotvec())[2]

    scan = P.prepare_scan(RP._G['frames'][i], RP._G['cfg'])
    q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]

    mp3 = RP._G['maps'].local(t3, RP._G['maps'].nearby(t3)[:RP._G['cfg'].reference_count])
    tree3 = cKDTree(mp3)
    local = curve(tree3, q, t3, gt[i][:3, :3], D_LOCAL)      # truth-relative yaw grid
    global_ = curve(tree3, q, t3, gt[i][:3, :3], D_GLOBAL)

    mpg = RP._G['maps'].local(tg, RP._G['maps'].nearby(tg)[:RP._G['cfg'].reference_count])
    treeg = cKDTree(mpg)
    global_gt = curve(treeg, q, tg, gt[i][:3, :3], D_GLOBAL)

    c_truth_at_t3 = float(local[D_LOCAL == 0][0])
    c_pred_at_t3 = float(np.interp(yaw3, D_LOCAL, local))
    best_local_d = float(D_LOCAL[int(np.argmin(local))])
    global_min_d = float(D_GLOBAL[int(np.argmin(global_))])
    global_gt_min_d = float(D_GLOBAL[int(np.argmin(global_gt))])
    return dict(i=int(i), trans_err=float(np.linalg.norm(t3 - tg)), yaw3=float(yaw3),
                dyaw=float(yaw3), in_window=bool(abs(yaw3) <= 15),
                c_truth_at_t3=c_truth_at_t3, c_pred_at_t3=c_pred_at_t3,
                improves=bool(c_truth_at_t3 < c_pred_at_t3 - 0.002),
                best_local_d=best_local_d,
                local_ident=bool(abs(best_local_d) <= 2),
                global_min_at_t3=global_min_d,
                global_min_at_truth=global_gt_min_d,
                ran_fine=bool(diag['ran_fine']))


rows = []
print('%-6s %8s %8s %6s %6s %9s %9s %9s %9s %6s' %
      ('frame', 'transErr', 'predYaw', 'in15', 'improve', 'c(pred)', 'c(truth)', 'localArg', 'globArgT3', 'fine'))
for tag, group in (('TAIL', TAIL), ('CTRL', CTRL)):
    print('--- %s ---' % tag)
    for i in group:
        r = evaluate(i)
        r['tag'] = tag
        rows.append(r)
        print('%-6d %8.2f %8.1f %6s %6s %9.4f %9.4f %9.1f %9.1f %6s' %
              (r['i'], r['trans_err'], r['yaw3'], r['in_window'], r['improves'],
               r['c_pred_at_t3'], r['c_truth_at_t3'], r['best_local_d'],
               r['global_min_at_t3'], r['ran_fine']))

print('\n== verdict ==')
for tag in ('TAIL', 'CTRL'):
    sub = [r for r in rows if r['tag'] == tag]
    n = len(sub)
    inw = sum(r['in_window'] for r in sub)
    imp = sum(r['in_window'] and r['improves'] and r['local_ident'] for r in sub)
    print('%-5s n=%2d  truth yaw inside +-15 of pred: %2d  fully reachable: %2d   '
          'truth is local min at T3: %2d   truth is global min at T3: %2d'
          % (tag, n, inw, imp, sum(r['local_ident'] for r in sub),
             sum(abs(r['global_min_at_t3']) <= 2 for r in sub)))
    print('      truth is global min at the TRUE translation: %d/%d'
          % (sum(abs(r['global_min_at_truth']) <= 2 for r in sub), n))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yaw_reach.json'), 'w'), indent=1)
