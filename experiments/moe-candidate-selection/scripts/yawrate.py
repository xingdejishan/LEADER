"""Radical hypothesis: the shared rotation floor is motion distortion of the scan.

A 10 Hz sweep lasts 100 ms.  If the vehicle is yawing during it, the points are smeared
by an angle the estimator cannot recover from the scan alone, and the effect would be
shared by every estimator - which is exactly what we observe.

Test: correlate the rotation error with the vehicle's own yaw rate, taken from the
ground-truth trajectory at the timestamps adjacent to each frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    keys = [str(k) for k in w['hard_keys']]
    dates = np.asarray(w['dates_hard'])
    stamps = np.array([int(k.split('/')[1]) for k in keys], dtype=np.int64)

    Rd = np.swapaxes(v7[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    err = np.linalg.norm(rv, axis=1)

    order = np.lexsort((stamps, dates))
    yawrate = np.full(len(err), np.nan)
    speed = np.full(len(err), np.nan)
    for a, b in zip(order[:-1], order[1:]):
        if dates[a] != dates[b]:
            continue
        dt = (stamps[b] - stamps[a]) / 1e6
        if not (0 < dt <= 2.0):
            continue
        Ra = gt[a][:3, :3]
        Rb = gt[b][:3, :3]
        dR = Rotation.from_matrix(Ra.T @ Rb).as_rotvec()
        yawrate[b] = abs(np.degrees(dR[2])) / dt
        speed[b] = np.linalg.norm(gt[b][:3, 3] - gt[a][:3, 3]) / dt

    ok = np.isfinite(yawrate)
    print('frames with a usable neighbour: %d' % int(ok.sum()))
    print('yaw rate: median %.1f deg/s   p90 %.1f' % (np.median(yawrate[ok]), np.percentile(yawrate[ok], 90)))
    print('correlation of rotation error with yaw rate : r = %+.3f'
          % np.corrcoef(err[ok], yawrate[ok])[0, 1])
    print('correlation of rotation error with speed    : r = %+.3f'
          % np.corrcoef(err[ok], speed[ok])[0, 1])
    print()
    q = np.quantile(yawrate[ok], [0.25, 0.5, 0.75])
    edges = [-1, q[0], q[1], q[2], 1e9]
    names = ['q1 lowest', 'q2', 'q3', 'q4 highest']
    print('%-12s %6s %12s %10s %10s' % ('yaw rate', 'n', 'range deg/s', 'mean err', 'median err'))
    for k in range(4):
        m = (yawrate > edges[k]) & (yawrate <= edges[k + 1]) & ok
        if m.sum() == 0:
            continue
        print('%-12s %6d %12s %10.3f %10.3f'
              % (names[k], int(m.sum()), '%.1f-%.1f' % (yawrate[m].min(), yawrate[m].max()),
                 err[m].mean(), np.median(err[m])))


if __name__ == '__main__':
    main()
