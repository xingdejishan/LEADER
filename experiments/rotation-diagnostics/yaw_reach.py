"""Is the correct yaw basin REACHABLE from where the pipeline actually is?

This is the decisive companion to yaw_ident.py.  There the pose was parked at the
ground-truth translation; here it is parked at the translation the current model
actually outputs, which is what a deployment can start from.

For every yaw-tail frame (|yaw error| > 8 deg) and a control set:

  A  run the current model on the frame -> its real output pose T3 (rotation is
     never modified by the current pipeline, so T3 carries the incoming yaw)
  B  sweep body yaw at T3's translation over +-40 deg, 2 deg  -> local reachability
  C  sweep body yaw at the TRUE translation over +-180 deg, 4 deg -> is the truth a
     global basin, or is there a competing one?

Per frame the verdict needs all three of:
  in_window   |predicted yaw error| <= 15 deg          (a +-15 sweep contains the truth)
  improves    cost at the true yaw < cost at the predicted yaw, at T3's position
  identifiable  the true yaw is within 2 deg of a local minimum at T3's position
"""
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import config

sys.path.insert(0, str(config.MODEL / 'stages/fast_search'))
sys.path.insert(0, str(config.MODEL))

import run_pipeline as RP      # noqa: E402
import pipeline as P           # noqa: E402

WINDOW = 15.0
IMPROVE_EPS = 0.002
LOCAL_DEG = 2.0
D_LOCAL = np.arange(-40.0, 40.001, 2.0)
D_GLOBAL = np.arange(-180.0, 180.001, 4.0)


def curve(tree, q, t, R0, grid):
    out = np.empty(len(grid))
    for k, d in enumerate(grid):
        R = R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix()
        out[k] = float(np.minimum(tree.query(q @ R.T + t, k=1, workers=1)[0], 1.0).mean())
    return out


def main():
    spec = importlib.util.spec_from_file_location('score_unified', config.SCORER)
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    base = np.asarray(w['hard_base'], float)

    Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    yaw_err = np.abs(rv[:, 2])
    full = np.linalg.norm(rv, axis=1)

    RP._OPTS.update(gate2=0.01)
    RP.init()

    tail = list(np.flatnonzero(yaw_err > 8))
    rng = np.random.default_rng(7)
    ctrl = list(rng.choice(np.flatnonzero(full < 0.7), min(12, int((full < 0.7).sum())),
                           replace=False))

    def evaluate(i):
        _, T3, _ = RP.one(i)
        t3, tg = T3[:3, 3], gt[i][:3, 3]
        yaw3 = np.degrees(Rotation.from_matrix(gt[i][:3, :3].T @ T3[:3, :3]).as_rotvec())[2]
        scan = P.prepare_scan(RP._G['frames'][i], RP._G['cfg'])
        q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]

        def tree_at(t):
            mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:RP._G['cfg'].reference_count])
            return cKDTree(mp)

        local = curve(tree_at(t3), q, t3, gt[i][:3, :3], D_LOCAL)
        glob3 = curve(tree_at(t3), q, t3, gt[i][:3, :3], D_GLOBAL)
        globg = curve(tree_at(tg), q, tg, gt[i][:3, :3], D_GLOBAL)

        c_truth = float(local[D_LOCAL == 0][0])
        c_pred = float(np.interp(yaw3, D_LOCAL, local))
        best_local = float(D_LOCAL[int(np.argmin(local))])
        return dict(i=int(i), trans_err=float(np.linalg.norm(t3 - tg)), yaw=float(yaw3),
                    in_window=bool(abs(yaw3) <= WINDOW),
                    c_pred=c_pred, c_truth=c_truth,
                    improves=bool(c_truth < c_pred - IMPROVE_EPS),
                    local_ident=bool(abs(best_local) <= LOCAL_DEG),
                    global_at_t3=float(D_GLOBAL[int(np.argmin(glob3))]),
                    global_at_truth=float(D_GLOBAL[int(np.argmin(globg))]))

    rows = []
    print('%-6s %9s %8s %6s %8s %9s %9s %9s' %
          ('frame', 'transErr', 'predYaw', 'in15', 'improve', 'c(pred)', 'c(truth)', 'globArgT3'))
    for tag, group in (('TAIL', tail), ('CTRL', ctrl)):
        print('--- %s ---' % tag)
        for i in group:
            r = evaluate(i)
            r['tag'] = tag
            rows.append(r)
            print('%-6d %9.2f %8.1f %6s %8s %9.4f %9.4f %9.1f' %
                  (r['i'], r['trans_err'], r['yaw'], r['in_window'], r['improves'],
                   r['c_pred'], r['c_truth'], r['global_at_t3']))

    print('\n== verdict ==')
    for tag in ('TAIL', 'CTRL'):
        sub = [r for r in rows if r['tag'] == tag]
        reachable = sum(r['in_window'] and r['improves'] and r['local_ident'] for r in sub)
        print('%-5s n=%2d  truth yaw inside +-15 of pred: %2d   fully reachable: %2d   '
              'truth is local min at T3: %2d   global min at T3: %2d   global min at truth: %2d'
              % (tag, len(sub), sum(r['in_window'] for r in sub), reachable,
                 sum(r['local_ident'] for r in sub),
                 sum(abs(r['global_at_t3']) <= LOCAL_DEG for r in sub),
                 sum(abs(r['global_at_truth']) <= LOCAL_DEG for r in sub)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yaw_reach.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
