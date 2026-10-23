"""Take the visual branch's ROTATION only, keep the search translation.

Why this is a single-frame idea with a real mechanism: in PnP the rotation is fixed
by the bearing vectors and is usually well conditioned, while the translation is the
weak axis (depth).  The current arbitration takes or rejects the whole pose, so a
candidate with a good orientation and a bad position is thrown away entirely.

Reads the frozen verify tables (no new inference) and asks:
  * when a candidate exists, is its rotation better than the baseline's?
  * what MOE would the hybrid get (candidate rotation + search translation)?
  * does the geometry score that the arbitration uses predict rotation quality?
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = w['hard_keys']
    gt = np.asarray(w['hard_gt'], float)

    search = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    smap = {str(k): p for k, p in zip(search['keys'], search['poses'])}

    def rot_err(R_a, R_b):
        Rd = np.swapaxes(R_a, -1, -2) @ R_b
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec())))

    n = len(keys)
    base_r = np.zeros(n)
    cand_r = np.full(n, np.nan)
    base_score = np.full(n, np.nan)
    cand_score = np.full(n, np.nan)
    for i in range(n):
        P = smap[keys[i]]
        base_r[i] = rot_err(P[:3, :3], gt[i][:3, :3])
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        base_score[i] = np.nan if base is None else float(base)
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        if best is not None:
            cand_score[i] = best[0]
            cand_r[i] = rot_err(best[1][:3, :3], gt[i][:3, :3])

    have = np.isfinite(cand_r)
    print('frames with a candidate: %d / %d' % (have.sum(), n))
    print('rotation error   search %.4f deg   candidate %.4f deg   (on those %d frames)'
          % (base_r[have].mean(), cand_r[have].mean(), have.sum()))
    print('candidate rotation better than search: %d of %d (%.0f%%)'
          % (int((cand_r[have] < base_r[have]).sum()), int(have.sum()),
             100.0 * (cand_r[have] < base_r[have]).mean()))
    print()
    print('search-only MOE over all frames      : %.4f deg' % base_r.mean())
    print('hybrid MOE, candidate rot on the %d   : %.4f deg'
          % (have.sum(), np.where(have, cand_r, base_r).mean()))
    for margin in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
        sel = have & np.isfinite(base_score) & np.isfinite(cand_score) & \
            (cand_score > base_score + margin)
        if sel.sum() == 0:
            continue
        moe = np.where(sel, cand_r, base_r).mean()
        print('  file  geometry-score margin %.2f : %4d frames  MOE %.4f  (candidate rot on those %.3f vs search %.3f)'
              % (margin, int(sel.sum()), moe, cand_r[sel].mean(), base_r[sel].mean()))
    print()
    ok = have & np.isfinite(base_score) & np.isfinite(cand_score)
    if ok.sum() > 10:
        gain = base_r[ok] - cand_r[ok]
        print('correlation of (candidate score - baseline score) with rotation gain: r = %.3f'
              % np.corrcoef((cand_score - base_score)[ok], gain)[0, 1])
        print('correlation of candidate score with candidate rotation error: r = %.3f'
              % np.corrcoef(cand_score[ok], cand_r[ok])[0, 1])


if __name__ == '__main__':
    main()
