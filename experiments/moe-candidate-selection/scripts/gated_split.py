"""Among the 84 gated frames: which ones took the visual branch, which did not, and why."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


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
    keys = [str(k) for k in w['hard_keys']]

    diag = [json.loads(l) for l in open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))]
    ratio = np.array([d.get('ratio', np.nan) for d in diag])
    visual = np.array([bool(d.get('visual')) for d in diag])
    gated = np.isfinite(ratio) & (ratio < 0.21)

    z = np.load(os.path.join(MOE, 'results', 'three_route.npz'), allow_pickle=True)
    m = {str(k): np.asarray(p, float) for k, p in zip(z['keys'], z['poses'])}
    moe = np.stack([m[str(k)] for k in keys])
    tm, rm = errs(moe, gt)

    # verified candidate availability and the margin test
    have = np.zeros(len(keys), bool)
    margin = np.zeros(len(keys), bool)
    for i in range(len(keys)):
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_') or not row.get('geometry_verified'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        if best is not None:
            have[i] = True
            if base is not None and best[0] > float(base) + 0.10:
                margin[i] = True

    print('gated frames: %d   of which took the visual branch: %d' % (int(gated.sum()), int((gated & visual).sum())))
    for name, m_ in (('took visual', gated & visual), ('did NOT take visual', gated & ~visual)):
        n = int(m_.sum())
        if n == 0:
            continue
        print('  %-20s n=%2d   MPE %7.2f   MOE %6.2f   >10m %d' %
              (name, n, tm[m_].mean(), rm[m_].mean(), int((tm[m_] > 10).sum())))

    print()
    print('of the %d that did NOT take it:' % int((gated & ~visual).sum()))
    print('   had a verified candidate      : %d' % int((gated & ~visual & have).sum()))
    print('   passed the 0.10 score margin  : %d' % int((gated & ~visual & margin).sum()))
    print('   no verified candidate at all  : %d' % int((gated & ~visual & ~have).sum()))
    print()
    print('of the %d that did take it:' % int((gated & visual).sum()))
    print('   MPE median %.2f  max %.1f   MOE median %.2f  max %.1f'
          % (np.median(tm[gated & visual]), tm[gated & visual].max(),
             np.median(rm[gated & visual]), rm[gated & visual].max()))


if __name__ == '__main__':
    main()
