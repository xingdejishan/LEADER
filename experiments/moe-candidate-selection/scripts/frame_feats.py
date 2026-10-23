"""Among the 55 dead frames, which ones hold a usable candidate?  Frame-level features only.

By construction none of these frames has a candidate that passes the 0.10 margin, so the
score gap cannot be the answer.  Everything here is a frame-level statistic that is
available at inference.
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


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def auc(x, y):
    x = np.nan_to_num(np.asarray(x, float), nan=-1e9)
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    diag = [json.loads(l) for l in open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))]
    ratio = np.array([d.get('ratio', np.nan) for d in diag])
    visual = np.array([bool(d.get('visual')) for d in diag])
    dead = np.isfinite(ratio) & (ratio < 0.21) & ~visual

    names, feats, labels, frames = [], [], [], []
    for i in np.flatnonzero(dead):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        cs = []
        for k, h in enumerate(hyps):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cs.append(dict(pose=np.asarray(h['pose'], float),
                           refs=set(int(x) for x in (h.get('reference_ids') or [])),
                           score=float(f.get('geometry_score') if f.get('geometry_score') is not None else -9.0),
                           ver=bool(c.get('geometry_verified')),
                           inl=float(h.get('unique_query_inliers') or 0),
                           irat=float(h.get('inlier_ratio') or 0),
                           rep=float(h.get('median_reprojection_px') or 1e9),
                           cov=float(h.get('coverage_cells') or 0),
                           resm=float(f.get('residual_median_m') or 1e9),
                           resp90=float(f.get('residual_p90_m') or 1e9)))
        if not cs:
            continue
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        b = -9.0 if b is None else float(b)
        good = np.array([np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= 2.0
                         and ang(c['pose'][:3, :3], gt[i][:3, :3]) <= 2.0 for c in cs])
        n = len(cs)
        sc = np.array([c['score'] for c in cs])
        cons = np.zeros(n)
        for a in range(n):
            for c2 in range(n):
                if a == c2:
                    continue
                if cs[a]['refs'] and cs[c2]['refs'] and (cs[a]['refs'] & cs[c2]['refs']):
                    continue
                if (np.linalg.norm(cs[a]['pose'][:3, 3] - cs[c2]['pose'][:3, 3]) <= TAU_T
                        and ang(cs[a]['pose'][:3, :3], cs[c2]['pose'][:3, :3]) <= TAU_R):
                    cons[a] += 1
        # frame-level statistics
        pts = np.array([c['pose'][:3, 3] for c in cs])
        spread = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        feats.append(dict(
            best_score=sc.max(), gap=b - sc.max(), mean_score=sc.mean(), std_score=sc.std(),
            n=n, n_ver=int(sum(c['ver'] for c in cs)),
            max_cons=float(cons.max()), mean_cons=float(cons.mean()),
            frac_cons=float((cons >= 1).mean()),
            max_inl=max(c['inl'] for c in cs), sum_inl=sum(c['inl'] for c in cs),
            min_rep=min(c['rep'] for c in cs), max_cov=max(c['cov'] for c in cs),
            min_resm=min(c['resm'] for c in cs), min_resp90=min(c['resp90'] for c in cs),
            spread=spread, base=b, ratio=float(ratio[i])))
        labels.append(bool(good.any()))
        frames.append((int(i), good, cs, cons))

    labels = np.array(labels)
    print('dead frames with a usable candidate: %d / %d' % (int(labels.sum()), len(labels)))
    rows = []
    for key in feats[0]:
        x = np.array([f[key] for f in feats], float)
        a = auc(x, labels)
        rows.append((abs(a - 0.5), a, key))
    rows.sort(reverse=True)
    print('\n%-14s %6s' % ('frame feature', 'AUC'))
    for _, a, k in rows:
        print('%-14s %6.3f' % (k, a))

    best = rows[0][2]
    x = np.array([f[best] for f in feats], float)
    print('\nbest single feature: %s' % best)
    for t in np.percentile(x, [20, 40, 50, 60, 80]):
        sel = x >= t
        if sel.sum() == 0:
            continue
        tp = int((sel & labels).sum()); fp = int((sel & ~labels).sum())
        print('  threshold %.3f -> selects %2d frames, %2d correct, %2d wrong (precision %.0f%%)'
              % (t, int(sel.sum()), tp, fp, 100 * tp / max(1, tp + fp)))


if __name__ == '__main__':
    main()
