"""Does the selection need LiDAR verification at all?

The match stage already stores visual-only quality per hypothesis: unique query
inliers, inlier ratio, median reprojection, coverage cells, and a 'qualified' flag.
Compare the rotation accuracy of five selection rules on the same frozen data.

  geo        highest LiDAR geometry score (what the shipped rule uses)
  verified   highest geometry score among LiDAR-verified candidates
  inliers    most unique query inliers among qualified hypotheses
  reproj     smallest median reprojection among qualified
  combo      inliers x coverage among qualified
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def rerr(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)

    rules = ('geo', 'verified', 'inliers', 'reproj', 'combo')
    errs = {k: [] for k in rules}
    miss = {k: 0 for k in rules}

    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        if not hyps:
            for k in rules:
                miss[k] += 1
            continue
        rows = []
        for k, h in enumerate(hyps):
            cand = v.get('candidate_%02d' % k)
            if cand is None:
                continue
            P = np.asarray(h['pose'], float)
            rows.append(dict(pose=P,
                             geo=(cand.get('fair') or {}).get('geometry_score'),
                             ver=bool(cand.get('geometry_verified')),
                             inl=int(h.get('unique_query_inliers') or 0),
                             rep=float(h.get('median_reprojection_px') or 1e9),
                             cov=int(h.get('coverage_cells') or 0),
                             qual=bool(h.get('qualified'))))
        if not rows:
            for k in rules:
                miss[k] += 1
            continue

        def pick(key, best=True, pool=None):
            cand = [r for r in rows if (pool is None or pool(r))]
            if not cand:
                return None
            return max(cand, key=key) if best else min(cand, key=key)

        picks = {
            'geo': pick(lambda r: r['geo'] if r['geo'] is not None else -9),
            'verified': pick(lambda r: r['geo'] if r['geo'] is not None else -9,
                             pool=lambda r: r['ver']),
            'inliers': pick(lambda r: r['inl'], pool=lambda r: r['qual']),
            'reproj': pick(lambda r: r['rep'], best=False, pool=lambda r: r['qual']),
            'combo': pick(lambda r: r['inl'] * max(1, r['cov']), pool=lambda r: r['qual']),
        }
        for k in rules:
            p = picks[k]
            if p is None:
                miss[k] += 1
            else:
                errs[k].append(rerr(p['pose'][:3, :3], gt[i][:3, :3]))

    print('%-10s %7s %7s %9s %9s %9s' % ('rule', 'n', 'miss', 'median', 'mean', 'p75'))
    for k in rules:
        e = np.array(errs[k])
        if len(e) == 0:
            continue
        print('%-10s %7d %7d %9.2f %9.2f %9.2f'
              % (k, len(e), miss[k], np.median(e), e.mean(), np.percentile(e, 75)))


if __name__ == '__main__':
    main()
