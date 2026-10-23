"""What information separates a correct visual candidate from a wrong one?

Over ALL candidates of all 1502 frames (not just the verified subset):
  * the true oracle: MOE if a perfect selector picked the best candidate per frame
  * the AUC of every per-candidate feature that the pipeline already computes
  * a leave-one-date-out logistic over all features -> achievable separation and
    end-to-end MPE/MOE when used as the acceptance rule
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


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float('nan')
    order = np.argsort(x)
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    rows = []
    per_frame = []
    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        h = m.get('hypotheses') or []
        if not h:
            continue
        cands = []
        for k, hyp in enumerate(h):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            f = c.get('fair') or {}
            cands.append(dict(pose=np.asarray(hyp['pose'], float),
                              refs=set(int(x) for x in (hyp.get('reference_ids') or [])),
                              score=f.get('geometry_score'),
                              ver=bool(c.get('geometry_verified')),
                              inl=float(hyp.get('unique_query_inliers') or 0),
                              irat=float(hyp.get('inlier_ratio') or 0),
                              rep=float(hyp.get('median_reprojection_px') or 1e9),
                              cov=float(hyp.get('coverage_cells') or 0),
                              qual=bool(hyp.get('qualified')),
                              resm=float(f.get('residual_median_m') or 1e9),
                              resp90=float(f.get('residual_p90_m') or 1e9),
                              bins=float(f.get('supported_bins') or 0),
                              nref=float(len(f.get('map_references') or []))
                              if isinstance(f.get('map_references'), list) else float(f.get('map_references') or 0)))
        n = len(cands)
        cons = np.zeros(n, int)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if cands[a]['refs'] and cands[b]['refs'] and (cands[a]['refs'] & cands[b]['refs']):
                    continue
                if (np.linalg.norm(cands[a]['pose'][:3, 3] - cands[b]['pose'][:3, 3]) <= TAU_T
                        and angle(cands[a]['pose'][:3, :3], cands[b]['pose'][:3, :3]) <= TAU_R):
                    cons[a] += 1
        errs_t = [float(np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3])) for c in cands]
        errs_r = [angle(c['pose'][:3, :3], gt[i][:3, :3]) for c in cands]
        best_idx = int(np.argmin(np.array(errs_r)))
        per_frame.append((i, errs_r, errs_t, best_idx, cands))

        for k, c in enumerate(cands):
            rows.append(dict(i=i,
                             y=bool(errs_t[k] <= GOOD_T and errs_r[k] <= GOOD_R),
                             score=c['score'] if c['score'] is not None else -9.0,
                             ver=float(c['ver']), inl=c['inl'], irat=c['irat'],
                             rep=-c['rep'], cov=c['cov'], qual=float(c['qual']),
                             resm=-c['resm'], resp90=-c['resp90'], bins=c['bins'],
                             cons=float(cons[k]),
                             dt=float(np.linalg.norm(c['pose'][:3, 3] - v7[i][:3, 3])),
                             dr=angle(c['pose'][:3, :3], v7[i][:3, :3]),
                             nref=c['nref']))

    y = np.array([r['y'] for r in rows])
    print('candidates %d   correct %.1f%%' % (len(rows), 100 * y.mean()))
    print()
    feats = ('score', 'ver', 'inl', 'irat', 'rep', 'cov', 'qual', 'resm', 'resp90',
             'bins', 'cons', 'dt', 'dr', 'nref')
    print('%-8s %6s' % ('feature', 'AUC'))
    res = []
    for f in feats:
        a = auc([r[f] for r in rows], y)
        res.append((a, f))
        print('%-8s %6.3f' % (f, a))
    print()
    # oracle: best candidate per frame by truth
    opos = np.array([np.min(pf[2]) for pf in per_frame])
    orot = np.array([np.min(pf[1]) for pf in per_frame])
    print('oracle over ALL candidates: MPE %.3f   MOE %.3f' % (opos.mean(), orot.mean()))
    print('V7 baseline               : MPE 1.913   MOE 2.524')
    print('deployed rule (margin only): MPE 0.971   MOE 2.107')


if __name__ == '__main__':
    main()
