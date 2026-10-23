"""Nonlinear selector over all per-candidate features, leave-one-date-out.

Fixes the two flaws of the previous attempt:
  * the absolute geometry score is kept as a feature (and so is the gap to the baseline
    score, its within-frame rank and z-score) - the previous version destroyed it;
  * the model is a gradient-boosted tree ensemble, not a linear logistic, so
    interactions between the map-side and image-side features can be represented.

The rule keeps the baseline pose unless the top candidate's probability clears a
threshold, so a frame can always fall back.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.ensemble import GradientBoostingClassifier

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
TAU_T, TAU_R = 3.0, 5.0
GOOD_T, GOOD_R = 2.0, 2.0


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    dates = np.asarray(w['dates_hard'])
    keys = [str(k) for k in w['hard_keys']]

    zz = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    base = {str(k): np.asarray(p, float) for k, p in zip(zz['keys'], zz['poses'])}

    rows, frames = [], {}
    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        if not hyps:
            continue
        cands = []
        for k, h in enumerate(hyps):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cands.append(dict(
                pose=np.asarray(h['pose'], float),
                refs=set(int(x) for x in (h.get('reference_ids') or [])),
                score=float(f.get('geometry_score') if f.get('geometry_score') is not None else -9.0),
                ver=float(bool(c.get('geometry_verified'))),
                inl=float(h.get('unique_query_inliers') or 0),
                irat=float(h.get('inlier_ratio') or 0),
                rep=float(h.get('median_reprojection_px') or 1e9),
                cov=float(h.get('coverage_cells') or 0),
                resm=float(f.get('residual_median_m') or 1e9),
                resp90=float(f.get('residual_p90_m') or 1e9),
                bins=float(f.get('supported_bins') or 0)))
        n = len(cands)
        if n == 0:
            continue
        cons = np.zeros(n)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if cands[a]['refs'] and cands[b]['refs'] and (cands[a]['refs'] & cands[b]['refs']):
                    continue
                if (np.linalg.norm(cands[a]['pose'][:3, 3] - cands[b]['pose'][:3, 3]) <= TAU_T
                        and angle(cands[a]['pose'][:3, :3], cands[b]['pose'][:3, :3]) <= TAU_R):
                    cons[a] += 1
        base_s = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        base_s = -9.0 if base_s is None else float(base_s)
        sc = np.array([c['score'] for c in cands])
        rk = sc.argsort().argsort() / max(1, n - 1)
        zs = (sc - sc.mean()) / (sc.std() + 1e-9)
        frames[i] = cands
        for k, c in enumerate(cands):
            good = (np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= GOOD_T
                    and angle(c['pose'][:3, :3], gt[i][:3, :3]) <= GOOD_R)
            rows.append(dict(i=i, y=int(good),
                             f=[c['score'], c['score'] - base_s, rk[k], zs[k],
                                c['inl'], c['irat'], -c['rep'], c['cov'],
                                -c['resm'], -c['resp90'], c['bins'], cons[k],
                                c['ver'], float(np.linalg.norm(c['pose'][:3, 3] - v7[i][:3, 3])),
                                angle(c['pose'][:3, :3], v7[i][:3, :3])]))

    X = np.array([r['f'] for r in rows]); y = np.array([r['y'] for r in rows])
    owner = np.array([r['i'] for r in rows])
    print('candidates %d   correct %.1f%%' % (len(y), 100 * y.mean()))

    prob = np.zeros(len(y))
    for d in sorted(set(dates)):
        te = dates[owner] == d
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08,
                                         subsample=0.8, random_state=0)
        clf.fit(X[~te], y[~te])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    def auc(x, yy):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        n1, n0 = yy.sum(), (~yy).sum()
        return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print('GBM LODO AUC             %.3f   (geometry score alone %.3f)'
          % (auc(prob, y), auc(X[:, 0], y)))

    by = {}
    for j, i in enumerate(owner):
        by.setdefault(int(i), []).append(j)

    for thr in (0.3, 0.5, 0.7, 0.85):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[k].copy()
            if i in by:
                js = by[i]
                j = max(js, key=lambda j: prob[j])
                if prob[j] >= thr:
                    P = frames[i][js.index(j)]['pose'].copy()
                    used += 1
            poses.append(P)
        r = su.evaluate(np.stack(poses), None, w, 'gbm %.2f' % thr, verbose=False)['hard:all1502']
        print('threshold %.2f -> frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (thr, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
