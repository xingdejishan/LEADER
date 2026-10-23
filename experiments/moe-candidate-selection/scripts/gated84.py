"""The 84 gated frames: how wrong are they, in position as well as rotation?"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def errs(pred, gt):
    t = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    r = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
    return t, r


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

    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    ratio = np.array([json.loads(l).get('ratio', np.nan) for l in
                      open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))])
    g = np.isfinite(ratio) & (ratio < 0.21)

    tl, rl = errs(lead, gt)
    tv, rv = errs(v7, gt)

    print('%-10s %5s %10s %10s %10s %10s' % ('group', 'n', 'LEADER MPE', 'LEADER MOE', 'V7 MPE', 'V7 MOE'))
    for name, m in (('gated', g), ('rest', ~g), ('all', np.ones(len(g), bool))):
        print('%-10s %5d %10.3f %10.3f %10.3f %10.3f'
              % (name, int(m.sum()), tl[m].mean(), rl[m].mean(), tv[m].mean(), rv[m].mean()))
    print()
    print('share of the summed error carried by the %d gated frames:  LEADER MPE %.1f%%   MOE %.1f%%'
          % (int(g.sum()), 100 * tl[g].sum() / tl.sum(), 100 * rl[g].sum() / rl.sum()))
    print()
    print('inside the gated set: correlation(position error, yaw error) = %+.3f'
          % np.corrcoef(tl[g], rl[g])[0, 1])
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 1e9)):
        m = g & (tl > lo) & (tl <= hi)
        if m.sum() < 3:
            continue
        print('   LEADER position %-9s n=%3d   MOE %.2f   V7 MPE %.2f'
              % ('<=%g m' % hi if hi < 1e8 else '>20 m', int(m.sum()), rl[m].mean(), tv[m].mean()))


if __name__ == '__main__':
    main()
