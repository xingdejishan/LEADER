"""How much of the V7 population is above 2 deg / 5 deg of rotation error?

Reports the hard 1502, the ordinary 1000, and the two pooled, since the two sets have
very different distributions and pooling hides that.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def stats(name, gt, pred):
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    e = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
    n = len(e)
    print('%-16s n=%5d   mean %6.3f   median %6.3f   p90 %6.3f' % (name, n, e.mean(), np.median(e), np.percentile(e, 90)))
    for t in (1, 2, 3, 5, 10):
        m = e > t
        print('    > %2d deg : %5d  (%5.1f%%)   they carry %5.1f%% of the summed angle'
              % (t, int(m.sum()), 100 * m.mean(), 100 * e[m].sum() / e.sum()))
    return e


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()

    eh = stats('hard 1502', np.asarray(w['hard_gt'], float), np.asarray(w['hard_base'], float))
    eo = stats('ordinary 1000', np.asarray(w['ord_gt'], float), np.asarray(w['ord_base'], float))
    print()
    stats('pooled 2502', np.concatenate([eh, eo]), np.concatenate([np.asarray(w['hard_gt'], float),
                                                                  np.asarray(w['ord_gt'], float)]))
    print()
    print('share of the pooled mean contributed by the hard set: %.1f%%'
          % (100 * eh.sum() / (eh.sum() + eo.sum())))


if __name__ == '__main__':
    main()
