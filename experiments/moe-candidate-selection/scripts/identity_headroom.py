"""How much is the identity-shortcut property worth?

The refinement behaves like a shrinkage: it helps bad frames and hurts good ones.
The ResNet-style fix would be to parametrise it as a residual with an identity default,
so a frame is only moved when there is evidence.  Before proposing that, measure the
headroom: the oracle MOE if every frame kept whichever of LEADER / V7 was better.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def rerr(pred, gt):
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    return np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))


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

    el = rerr(lead, gt)
    ev = rerr(v7, gt)
    oracle = np.minimum(el, ev)
    print('LEADER MOE                 %.4f' % el.mean())
    print('V7 (current) MOE           %.4f' % ev.mean())
    print('oracle min(LEADER, V7)     %.4f   headroom %.4f' % (oracle.mean(), ev.mean() - oracle.mean()))
    print()
    print('frames where LEADER is better: %d (%.0f%%)  their mean gain %.2f deg'
          % (int((el < ev).sum()), 100 * (el < ev).mean(),
             float((ev[el < ev] - el[el < ev]).mean())))
    print('frames where V7 is better    : %d (%.0f%%)  their mean gain %.2f deg'
          % (int((ev <= el).sum()), 100 * (ev <= el).mean(),
             float((el[ev <= el] - ev[ev <= el]).mean())))
    print()
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 1e9)):
        m = (el > lo) & (el <= hi)
        if m.sum() < 5:
            continue
        print('LEADER %-8s n=%4d  V7 %.2f  oracle %.2f  (V7 helps on %.0f%% of them)'
              % ('<=%g' % hi if hi < 1e8 else '>5', int(m.sum()), ev[m].mean(), oracle[m].mean(),
                 100 * (ev[m] < el[m]).mean()))


if __name__ == '__main__':
    main()
