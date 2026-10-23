"""LEADER as the baseline: what does its own rotation error look like?

Same battery as before, but on the pure LEADER pose rather than the V7-refined one:
axis decomposition, common-offset test, and how it relates to what the refinement does.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def err(pred, gt):
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    return np.linalg.norm(rv, axis=1), rv


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    z = np.load(HIST, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(keys)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in w['hard_keys']])

    e_l, rv_l = err(lead, gt)
    e_v, rv_v = err(v7, gt)
    print('LEADER   MOE %.4f   median %.4f   p90 %.4f' % (e_l.mean(), np.median(e_l), np.percentile(e_l, 90)))
    print('V7       MOE %.4f   median %.4f   p90 %.4f' % (e_v.mean(), np.median(e_v), np.percentile(e_v, 90)))
    print()
    print('LEADER axis decomposition: |roll| %.3f  |pitch| %.3f  |yaw| %.3f  (deg)'
          % (np.abs(rv_l[:, 0]).mean(), np.abs(rv_l[:, 1]).mean(), np.abs(rv_l[:, 2]).mean()))
    tot = (rv_l ** 2).sum(1).mean()
    print('   share of squared error: roll %.1f%%  pitch %.1f%%  yaw %.1f%%'
          % (100 * (rv_l[:, 0] ** 2).mean() / tot, 100 * (rv_l[:, 1] ** 2).mean() / tot,
             100 * (rv_l[:, 2] ** 2).mean() / tot))
    print()
    print('common-offset test on LEADER (world frame): mean rotvec %s deg, per-axis std %s'
          % (np.round(rv_l.mean(0), 2), np.round(rv_l.std(0), 2)))
    # remove the mean rotation
    Rm = np.array(Rotation.from_rotvec(np.radians(rv_l.mean(0))).as_matrix())
    fixed = np.einsum('ij,njk->nik', Rm, lead[:, :3, :3])
    Rd = np.swapaxes(fixed, -1, -2) @ gt[:, :3, :3]
    e = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
    print('   after removing it: MOE %.4f (delta %+.4f)' % (e.mean(), e.mean() - e_l.mean()))
    print()
    print('what the V7 refinement does: rotation improved on %d frames, worsened on %d'
          % (int((e_v < e_l).sum()), int((e_v > e_l).sum())))
    print('   correlation between LEADER and V7 rotation errors: r = %+.3f'
          % np.corrcoef(e_l, e_v)[0, 1])
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 1e9)):
        m = (e_l > lo) & (e_l <= hi)
        if m.sum() < 5:
            continue
        print('   LEADER %-7s n=%4d   V7 on the same frames %.2f -> %.2f'
              % ('<=%g' % hi if hi < 1e8 else '>5', int(m.sum()), e_l[m].mean(), e_v[m].mean()))


if __name__ == '__main__':
    main()
