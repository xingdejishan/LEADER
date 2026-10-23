"""GBM as a second opinion on top of the score-margin rule (and a correct AUC)."""
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
    gt = np.asarray(w['hard_gt'], float); v7 = np.asarray(w['hard_base'], float)
    dates = np.asarray(w['dates_hard']); keys = [str(k) for k in w['hard_keys']]
    zz = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    base = {str(k): np.asarray(p, float) for k, p in zip(zz['keys'], zz['poses'])}

    rows, frames, base_s = [], {}, {}
    for i in range(len(gt)):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        if not hyps:
            continue
        cands = []
        for k, h in enumerate(hyps):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cands.append(dict(pose=np.asarray(h['pose'], float),
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
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        b = -9.0 if b is None else float(b)
        base_s[i] = b
        frames[i] = cands
        sc = np.array([c['score'] for c in cands])
        rk = sc.argsort().argsort() / max(1, n - 1)
        zs = (sc - sc.mean()) / (sc.std() + 1e-9)
        cons = np.zeros(n)
        for a in range(n):
            for c2 in range(n):
                if a == c2:
                    continue
                if cands[a]['refs'] and cands[c2]['refs'] and (cands[a]['refs'] & cands[c2]['refs']):
                    continue
                if (np.linalg.norm(cands[a]['pose'][:3, 3] - cands[c2]['pose'][:3, 3]) <= TAU_T
                        and angle(cands[a]['pose'][:3, :3], cands[c2]['pose'][:3, :3]) <= TAU_R):
                    cons[a] += 1
        for k, c in enumerate(cands):
            good = bool(np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= GOOD_T
                        and angle(c['pose'][:3, :3], gt[i][:3, :3]) <= GOOD_R)
            rows.append(dict(i=i, y=good,
                             f=[c['score'], c['score'] - b, rk[k], zs[k], c['inl'], c['irat'],
                                -c['rep'], c['cov'], -c['resm'], -c['resp90'], c['bins'],
                                cons[k], c['ver'],
                                float(np.linalg.norm(c['pose'][:3, 3] - v7[i][:3, 3])),
                                angle(c['pose'][:3, :3], v7[i][:3, :3])]))

    X = np.array([r['f'] for r in rows], float)
    y = np.array([r['y'] for r in rows], bool)
    owner = np.array([r['i'] for r in rows])
    prob = np.zeros(len(y))
    for d in sorted(set(dates)):
        te = dates[owner] == d
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08,
                                         subsample=0.8, random_state=0)
        clf.fit(X[~te], y[~te])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    def auc(x, yy):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        n1, n0 = int(yy.sum()), int((~yy).sum())
        return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print('GBM LODO AUC %.3f   score alone %.3f' % (auc(prob, y), auc(X[:, 0], y)))

    by = {}
    for j, i in enumerate(owner):
        by.setdefault(int(i), []).append(j)

    def run(tag, choose):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[k].copy()
            if i in by:
                j = choose(i, by[i])
                if j is not None:
                    P = frames[i][by[i].index(j)]['pose'].copy(); used += 1
            poses.append(P)
        r = su.evaluate(np.stack(poses), None, w, tag, verbose=False)['hard:all1502']
        print('%-30s frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (tag, used, r['MPE'], r['MOE'], r['success_1m_2deg']))

    def margin(i, js, mg=0.10):
        ok = [j for j in js if X[j, 0] > base_s[i] + mg]
        return max(ok, key=lambda j: X[j, 0]) if ok else None

    run('margin only (control)', lambda i, js: margin(i, js))
    for t in (0.5, 0.7):
        def pick(i, js, t=t):
            m = margin(i, js)
            if m is not None:
                return m
            ok = [j for j in js if prob[j] >= t and X[j, 0] > base_s[i]]
            return max(ok, key=lambda j: prob[j]) if ok else None
        run('margin + gbm>=%.1f fallback' % t, pick)
    for mg in (0.0, 0.05):
        def pick2(i, js, mg=mg):
            ok = [j for j in js if X[j, 0] > base_s[i] + mg and prob[j] >= 0.7]
            if ok:
                return max(ok, key=lambda j: (prob[j], X[j, 0]))
            return margin(i, js)
        run('margin %.2f AND gbm>=0.7' % mg, pick2)


if __name__ == '__main__':
    main()
