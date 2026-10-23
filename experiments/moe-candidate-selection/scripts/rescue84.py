"""Can the visual T branch rescue the 84 gated frames (and the 26 that are rotation-only)?

For each gated frame: does a LiDAR-verified visual candidate exist, how wrong is it in
position and rotation, and what would replacing the pose do to MPE and MOE.
"""
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

    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    ratio = np.array([json.loads(l).get('ratio', np.nan) for l in
                      open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))])
    gated = np.isfinite(ratio) & (ratio < 0.21)

    tl, rl = errs(lead, gt)
    cand = np.full((len(keys), 4, 4), np.nan)
    have = np.zeros(len(keys), bool)
    for i in range(len(keys)):
        rec = json.load(open(VERIFY % i))['poses']
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
            cand[i] = best[1]
            have[i] = True

    tc, rc = errs(np.where(have[:, None, None], cand, lead), gt)

    for name, m in (('all 1502', np.ones(len(keys), bool)),
                    ('gated 84', gated),
                    ('gated & LEADER pos<=1m', gated & (tl <= 1.0))):
        n = int(m.sum())
        h = m & have
        print('== %s ==  n=%d' % (name, n))
        print('   have a verified visual candidate: %d (%.0f%%)' % (h.sum(), 100 * h.sum() / max(n, 1)))
        if h.sum() == 0:
            print()
            continue
        print('   LEADER   MPE %7.2f   MOE %6.2f' % (tl[h].mean(), rl[h].mean()))
        print('   candidate MPE %7.2f   MOE %6.2f' % (tc[h].mean(), rc[h].mean()))
        print('   candidate better in position on %.0f%%, in rotation on %.0f%%'
              % (100 * (tc[h] < tl[h]).mean(), 100 * (rc[h] < rl[h]).mean()))
        # replace everywhere we have a candidate
        mix_t = np.where(h, tc, tl)
        mix_r = np.where(h, rc, rl)
        print('   if replaced: whole-set MPE %.3f -> %.3f   MOE %.3f -> %.3f'
              % (tl.mean(), mix_t.mean(), rl.mean(), mix_r.mean()))
        print()


if __name__ == '__main__':
    main()
