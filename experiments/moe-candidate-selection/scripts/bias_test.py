"""Is the floor a common rotation bias, or random error?

Per-frame offset in the world frame is E_i = R_gt R_pred^T; if the mean of E_i is a
stable rotation, that is a calibration/convention bias and can be removed once.
Tests three ways of removing it:
  global      one constant offset fitted on all frames
  per date    one offset per recording date (different sessions, possibly different extrinsics)
  body        the same but applied on the right (a body-frame mounting offset)

Each is scored twice: in-sample (the offset is fitted and evaluated on the same frames)
and split-half (fit on one half, evaluate on the other) so a gain cannot come from
overfitting a per-frame quantity.
"""
import collections

import numpy as np
from scipy.spatial.transform import Rotation


def err_deg(pred, gt):
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    return np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))


def mean_rot(mats):
    """project the mean rotation matrix back onto SO(3)"""
    M = mats.mean(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()

    for tag, gt, pred, dates in (
            ('hard 1502', np.asarray(w['hard_gt'], float), np.asarray(w['hard_base'], float),
             np.asarray(w['dates_hard'])),
            ('ordinary 1000', np.asarray(w['ord_gt'], float), np.asarray(w['ord_base'], float),
             np.array([k.split('/')[0] for k in w['ord_keys']]))):
        base = err_deg(pred, gt)
        E = np.einsum('nij,nkj->nik', gt[:, :3, :3], pred[:, :3, :3])   # R_gt R_pred^T
        Ebody = np.einsum('nji,njk->nik', pred[:, :3, :3], gt[:, :3, :3])
        rv = Rotation.from_matrix(E).as_rotvec()
        rvb = Rotation.from_matrix(Ebody).as_rotvec()
        print('== %s ==  n=%d   MOE %.4f   median %.4f' % (tag, len(base), base.mean(), np.median(base)))
        print('  world-frame offset: mean rotvec %s deg   (per-axis std %s)'
              % (np.round(np.degrees(rv.mean(0)), 2), np.round(np.degrees(rv.std(0)), 2)))
        print('  body-frame  offset: mean rotvec %s deg   (per-axis std %s)'
              % (np.round(np.degrees(rvb.mean(0)), 2), np.round(np.degrees(rvb.std(0)), 2)))

        # global world offset, in sample
        Rg = mean_rot(E)
        fixed = np.einsum('ij,njk->nik', Rg, pred[:, :3, :3])
        e = err_deg(np.dstack([fixed, np.zeros((len(fixed), 3))]), gt)
        print('  with one global world offset removed: MOE %.4f  (delta %+.4f)'
              % (e.mean(), e.mean() - base.mean()))

        # per-date, in sample
        per = np.empty(len(base))
        Rdates = {}
        for d in sorted(set(dates)):
            m = dates == d
            Rd_ = mean_rot(E[m])
            Rdates[d] = Rd_
            per[m] = err_deg(np.dstack([np.einsum('ij,njk->nik', Rd_, pred[m][:, :3, :3]),
                                        np.zeros((m.sum(), 3))]), gt[m])
        print('  with one offset per date removed     : MOE %.4f  (delta %+.4f)'
              % (per.mean(), per.mean() - base.mean()))

        # split-half check: fit on even frames, evaluate on odd
        idx = np.arange(len(base))
        fit, ev = idx[0::2], idx[1::2]
        Rf = mean_rot(E[fit])
        e2 = err_deg(np.dstack([np.einsum('ij,njk->nik', Rf, pred[ev][:, :3, :3]),
                                np.zeros((len(ev), 3))]), gt[ev])
        print('  split-half (fit even, eval odd)      : MOE %.4f vs %.4f on the same frames'
              % (e2.mean(), base[ev].mean()))
        print()


if __name__ == '__main__':
    main()
