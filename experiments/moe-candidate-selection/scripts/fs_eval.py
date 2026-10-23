"""Stage 2: does the free-space feature add anything?"""
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


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float); v7 = np.asarray(w['hard_base'], float)
    dates = np.asarray(w['dates_hard']); keys = [str(k) for k in w['hard_keys']]
    fs = np.load('/dev/shm/freespace.npy', allow_pickle=True).item()
    vio = dict(zip(fs['idx'].tolist(), fs['vio']))
    zz = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    base = {str(k): np.asarray(p, float) for k, p in zip(zz['keys'], zz['poses'])}

    X, E, y, owner, frames, bs = [], [], [], [], {}, {}
    for i in range(len(gt)):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        cs = []
        for k, h in enumerate(hyps):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cs.append(dict(pose=np.asarray(h['pose'], float),
                           score=float(f.get('geometry_score') if f.get('geometry_score') is not None else -9.0),
                           ver=float(bool(c.get('geometry_verified'))),
                           inl=float(h.get('unique_query_inliers') or 0),
                           rep=float(h.get('median_reprojection_px') or 1e9),
                           cov=float(h.get('coverage_cells') or 0),
                           resm=float(f.get('residual_median_m') or 1e9),
                           resp90=float(f.get('residual_p90_m') or 1e9)))
        if not cs or i not in vio:
            continue
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        b = -9.0 if b is None else float(b); bs[i] = b; frames[i] = cs
        v_ = np.asarray(vio[i], float)
        cv, bv = v_[:-1], v_[-1]
        for k, c in enumerate(cs):
            good = bool(np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= 2.0
                        and ang(c['pose'][:3, :3], gt[i][:3, :3]) <= 2.0)
            X.append([c['score'], c['score'] - b, c['inl'], -c['rep'], c['cov'],
                      -c['resm'], -c['resp90'], c['ver'],
                      float(np.linalg.norm(c['pose'][:3, 3] - v7[i][:3, 3])),
                      ang(c['pose'][:3, :3], v7[i][:3, :3])])
            fs_feat = [bv - cv[k], float(np.mean(cv < cv[k]))] if np.isfinite(cv[k]) else [0.0, 0.5]
            E.append(fs_feat); y.append(good); owner.append(i)
    X = np.array(X); E = np.array(E); y = np.array(y, bool); owner = np.array(owner)

    def auc(x, yy):
        x = np.nan_to_num(x, nan=-9.0)
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        n1, n0 = int(yy.sum()), int((~yy).sum())
        return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print('AUC free-space (vs baseline) %.3f   AUC score %.3f'
          % (auc(E[:, 0], y), auc(X[:, 0], y)))
    prob = {}
    for tag, M in (('without', X), ('with', np.hstack([X, E]))):
        p = np.zeros(len(y))
        for d in sorted(set(dates)):
            te = dates[owner] == d
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.06,
                                             subsample=0.8, random_state=0)
            clf.fit(M[~te], y[~te])
            p[te] = clf.predict_proba(M[te])[:, 1]
        prob[tag] = p
        print('GBM LODO AUC %-8s %.3f' % (tag, auc(p, y)))

    by = {}
    for j, i in enumerate(owner):
        by.setdefault(int(i), []).append(j)

    for tag in ('without', 'with'):
        p = prob[tag]
        for t in (0.5, 0.7):
            poses, used = [], 0
            for i, k in enumerate(keys):
                P = base[k].copy()
                if i in by:
                    js = by[i]
                    ok = [j for j in js if X[j, 0] > bs[i] and p[j] >= t]
                    if ok:
                        j = max(ok, key=lambda j: p[j])
                        P = frames[i][js.index(j)]['pose'].copy(); used += 1
                    else:
                        ok2 = [j for j in js if X[j, 0] > bs[i] + 0.10]
                        if ok2:
                            j = max(ok2, key=lambda j: X[j, 0])
                            P = frames[i][js.index(j)]['pose'].copy(); used += 1
                poses.append(P)
            r = su.evaluate(np.stack(poses), None, w, '%s %.1f' % (tag, t), verbose=False)['hard:all1502']
            print('  feat %-8s threshold %.1f -> frames %4d  MPE %.4f  MOE %.4f  ok1/2 %d'
                  % (tag, t, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
