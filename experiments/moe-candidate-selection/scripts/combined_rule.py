"""Combined selector: can the per-candidate features together beat the score alone?

Leave-one-date-out logistic over the features the pipeline already computes, then used
as the acceptance rule with the V7 pose as the fallback.  The fallback matters: the rule
must be able to keep the baseline.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
TAU_T, TAU_R = 3.0, 5.0
GOOD_T, GOOD_R = 2.0, 2.0
FEATS = ('score', 'ver', 'inl', 'irat', 'rep', 'cov', 'resm', 'resp90', 'bins', 'cons')


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def build():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    frames = []
    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        if not hyps:
            frames.append(None); continue
        cands = []
        for k, h in enumerate(hyps):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cands.append(dict(pose=np.asarray(h['pose'], float),
                              refs=set(int(x) for x in (h.get('reference_ids') or [])),
                              score=float(f.get('geometry_score') if f.get('geometry_score') is not None else -9),
                              ver=float(bool(c.get('geometry_verified'))),
                              inl=float(h.get('unique_query_inliers') or 0),
                              irat=float(h.get('inlier_ratio') or 0),
                              rep=-float(h.get('median_reprojection_px') or 1e9),
                              cov=float(h.get('coverage_cells') or 0),
                              resm=-float(f.get('residual_median_m') or 1e9),
                              resp90=-float(f.get('residual_p90_m') or 1e9),
                              bins=float(f.get('supported_bins') or 0)))
        n = len(cands)
        if n == 0:
            frames.append(None); continue
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
        for k in range(n):
            cands[k]['cons'] = float(cons[k])
            cands[k]['y'] = bool(np.linalg.norm(cands[k]['pose'][:3, 3] - gt[i][:3, 3]) <= GOOD_T
                                 and angle(cands[k]['pose'][:3, :3], gt[i][:3, :3]) <= GOOD_R)
        base = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        frames.append(dict(cands=cands, base=None if base is None else float(base)))
    return su, w, gt, v7, frames


def logistic(X, y, iters=1500, lr=0.5, l2=1e-3):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) / len(y) + l2 * w
        w -= lr * g / max(1e-9, np.abs(g).max())
    return w


def main():
    su, w, gt, v7, frames = build()
    dates = np.asarray(w['dates_hard'])

    X, y, owner = [], [], []
    for i, fr in enumerate(frames):
        if fr is None:
            continue
        for c in fr['cands']:
            X.append([c[f] for f in FEATS]); y.append(c['y']); owner.append(i)
    X = np.array(X, float); y = np.array(y, bool); owner = np.array(owner)

    # within-frame standardisation of each feature (the rule must compare candidates
    # of the same frame, not absolute values across frames)
    for f_i in range(X.shape[1]):
        for i in np.unique(owner):
            m = owner == i
            s = X[m, f_i].std()
            if s > 1e-9:
                X[m, f_i] = (X[m, f_i] - X[m, f_i].mean()) / s

    pr = np.zeros(len(y))
    for d in sorted(set(dates)):
        te = dates[owner] == d
        tr = ~te
        w_ = logistic(X[tr], y[tr])
        pr[te] = 1.0 / (1.0 + np.exp(-np.clip(np.hstack([np.ones((te.sum(), 1)), X[te]]) @ w_, -30, 30)))

    def auc(x, yy):
        order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
        n1, n0 = yy.sum(), (~yy).sum()
        return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')

    print('combined AUC (LODO)      %.3f' % auc(pr, y))
    print('geometry score alone     %.3f' % auc(X[:, 0], y))

    base = {str(k): np.asarray(p, float) for k, p in
            zip(np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)['keys'],
                np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)['poses'])}
    keys = [str(k) for k in w['hard_keys']]

    idx_by_frame = {}
    for j, i in enumerate(owner):
        idx_by_frame.setdefault(int(i), []).append(j)

    for thr in (0.5, 0.7, 0.85, 0.95):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[k].copy()
            fr = frames[i]
            if fr is not None and i in idx_by_frame:
                js = idx_by_frame[i]
                best = max(js, key=lambda j: pr[j])
                if pr[best] >= thr:
                    P = fr['cands'][js.index(best)]['pose'].copy()
                    used += 1
            poses.append(P)
        rows = su.evaluate(np.stack(poses), None, w, 'thr %.2f' % thr, verbose=False)
        r = rows['hard:all1502']
        print('threshold %.2f -> frames %3d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (thr, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
