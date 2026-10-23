"""Is the base estimator's rotation error structured or white?

Correlates the per-frame rotation error of V7 with quantities the estimator could in
principle be made aware of: heading, local reference density, travel speed, date.
A structured error is fixable inside the estimator; an unstructured one is the floor.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    pred = np.asarray(w['hard_base'], float)
    dates = np.asarray(w['dates_hard'])

    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    err = np.linalg.norm(rv, axis=1)

    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    rpos = np.asarray([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs])
    rtree = cKDTree(rpos)

    fwd = pred[:, :3, 0]
    heading = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]))
    dens = np.array([len(rtree.query_ball_point(p, 40.0)) for p in pred[:, :3, 3]])

    stamps = np.array([int(str(k).split('/')[1]) for k in w['hard_keys']], dtype=np.int64)
    order = np.lexsort((stamps, dates))
    speed = np.full(len(err), np.nan)
    for a, b in zip(order[:-1], order[1:]):
        if dates[a] != dates[b]:
            continue
        dt = (stamps[b] - stamps[a]) / 1e6
        if 0 < dt <= 2.0:
            speed[b] = np.linalg.norm(pred[b, :3, 3] - pred[a, :3, 3]) / dt

    def rep(name, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20:
            print('%-24s  n=%d  (too few)' % (name, int(m.sum())))
            return
        r = np.corrcoef(x[m], y[m])[0, 1]
        lo = np.percentile(x[m], [10, 90])
        print('%-24s  r = %+.3f   n=%d   x range %.2f .. %.2f' % (name, r, int(m.sum()), lo[0], lo[1]))

    print('rotation error: mean %.3f  median %.3f  std %.3f' % (err.mean(), np.median(err), err.std()))
    print()
    rep('vs heading (deg)', heading, err)
    rep('vs local ref density', dens.astype(float), err)
    rep('vs speed (m/s)', speed, err)
    rep('vs |yaw error|', np.abs(rv[:, 2]), err)
    rep('vs |roll error|', np.abs(rv[:, 0]), err)
    print()
    print('%-12s %5s %8s %8s %8s' % ('date', 'n', 'mean', 'median', 'p90'))
    for d in sorted(set(dates)):
        m = dates == d
        print('%-12s %5d %8.3f %8.3f %8.3f' % (d, int(m.sum()), err[m].mean(),
                                               np.median(err[m]), np.percentile(err[m], 90)))
    print()
    print('rotation error by local reference density tertile:')
    q = np.quantile(dens, [1 / 3, 2 / 3])
    for lo, hi, tag in ((-1, q[0], 'sparse'), (q[0], q[1], 'medium'), (q[1], 1e9, 'dense')):
        m = (dens > lo) & (dens <= hi)
        print('  %-7s n=%4d  mean %.3f  median %.3f' % (tag, int(m.sum()), err[m].mean(), np.median(err[m])))


if __name__ == '__main__':
    main()
