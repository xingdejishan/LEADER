"""Can everything above the floor be pulled down?

Three ceilings for MOE:
  clamp      every frame's error capped at the floor      (needs the truth -> not deployable)
  oracle     per frame, take whichever available estimate is better (needs the truth to pick)
  rule       the deployable rule: candidate replaces V7 when its map score wins by 0.10
Also counts, among frames above the floor, how many have any better estimate available.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
FLOOR = 1.80


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    def rot_err(a, b):
        Rd = np.swapaxes(a, -1, -2) @ b
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec())))

    n = len(gt)
    err = np.array([rot_err(v7[i][:3, :3], gt[i][:3, :3]) for i in range(n)])
    best_alt = np.full(n, np.nan)
    rule = np.zeros(n, bool)
    for i in range(n):
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        bv, bc = None, None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            P = np.asarray(row['pose'], float)
            e = rot_err(P[:3, :3], gt[i][:3, :3])
            verified = bool(row.get('geometry_verified'))
            sc = (row.get('fair') or {}).get('geometry_score')
            if verified and (bv is None or e < bv):
                bv = e
            if sc is not None and (bc is None or sc > bc[0]):
                bc = (float(sc), e)
        if bv is not None:
            best_alt[i] = bv
        if bc is not None and base is not None and bc[0] > float(base) + 0.10:
            rule[i] = True

    have = np.isfinite(best_alt)
    print('frames %d   baseline MOE %.4f   median %.4f' % (n, err.mean(), np.median(err)))
    print('floor (median) %.2f deg' % FLOOR)
    print()
    print('ceiling A  clamp every error to %.2f          : MOE %.4f  (needs truth)' % (FLOOR, np.minimum(err, FLOOR).mean()))
    oracle = np.where(have, np.minimum(err, best_alt), err)
    print('ceiling B  best available estimate per frame   : MOE %.4f  (needs truth to pick)' % oracle.mean())
    print('rule       map-score margin, deployable        : MOE %.4f' % np.where(rule, best_alt, err).mean())
    print()
    above = err > FLOOR
    print('frames above the floor: %d (%.0f%%)   carrying %.0f%% of the summed error'
          % (above.sum(), 100 * above.mean(), 100 * err[above].sum() / err.sum()))
    print('  of those, a verified visual candidate exists : %d (%.0f%%)'
          % (int((above & have).sum()), 100 * (above & have).sum() / above.sum()))
    print('  of those, the candidate is actually better   : %d (%.0f%%)'
          % (int((above & have & (best_alt < err)).sum()),
             100 * (above & have & (best_alt < err)).sum() / above.sum()))
    if (above & have & (best_alt < err)).any():
        sel = above & have & (best_alt < err)
        print('  mean error on those: %.2f -> %.2f deg   (they would give -%.3f MOE)'
              % (err[sel].mean(), best_alt[sel].mean(), (err[sel] - best_alt[sel]).sum() / n))


if __name__ == '__main__':
    main()
