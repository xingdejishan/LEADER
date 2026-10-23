"""The decisive experiment: does cross-candidate consensus separate right from wrong?

For every hypothesis of the dead frames, compute m_k = number of OTHER hypotheses that
agree in pose AND come from a disjoint reference set.  Then ask whether correct ones
have higher m_k than wrong ones, and whether a consensus rule would rescue frames.
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

TAU_T = 3.0
TAU_R = 5.0
GOOD_T = 2.0
GOOD_R = 2.0


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def load(i):
    m = json.load(open(MATCH % i))
    v = json.load(open(VERIFY % i))['poses']
    hyps = m.get('hypotheses') or []
    out = []
    for k, h in enumerate(hyps):
        cand = v.get('candidate_%02d' % k)
        if cand is None:
            continue
        sc = (cand.get('fair') or {}).get('geometry_score')
        out.append(dict(pose=np.asarray(h['pose'], float),
                        refs=set(int(x) for x in (h.get('reference_ids') or [])),
                        score=None if sc is None else float(sc),
                        ver=bool(cand.get('geometry_verified'))))
    base = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
    return out, (None if base is None else float(base))


def consensus(cands):
    n = len(cands)
    m = np.zeros(n, int)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if cands[a]['refs'] and cands[b]['refs'] and (cands[a]['refs'] & cands[b]['refs']):
                continue
            dt = np.linalg.norm(cands[a]['pose'][:3, 3] - cands[b]['pose'][:3, 3])
            if dt <= TAU_T and angle(cands[a]['pose'][:3, :3], cands[b]['pose'][:3, :3]) <= TAU_R:
                m[a] += 1
    return m


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)

    diag = [json.loads(l) for l in open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))]
    ratio = np.array([d.get('ratio', np.nan) for d in diag])
    visual = np.array([bool(d.get('visual')) for d in diag])
    gated = np.isfinite(ratio) & (ratio < 0.21)
    dead = gated & ~visual

    good_m, bad_m = [], []
    per_frame = []
    for i in np.flatnonzero(dead):
        cands, base = load(i)
        if not cands:
            continue
        m = consensus(cands)
        is_good = np.array([np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= GOOD_T
                            and angle(c['pose'][:3, :3], gt[i][:3, :3]) <= GOOD_R
                            for c in cands])
        good_m.extend(m[is_good].tolist())
        bad_m.extend(m[~is_good].tolist())
        per_frame.append((int(i), int(is_good.sum()), m, is_good, cands, base))

    gm, bm = np.array(good_m), np.array(bad_m)
    print('candidates on the dead frames: correct %d, incorrect %d' % (len(gm), len(bm)))
    print('consensus m  correct: mean %.2f  median %.1f  share m>=2 %.0f%%'
          % (gm.mean(), np.median(gm), 100 * (gm >= 2).mean()))
    print('consensus m  wrong  : mean %.2f  median %.1f  share m>=2 %.0f%%'
          % (bm.mean(), np.median(bm), 100 * (bm >= 2).mean()))
    # rank-based AUC
    allm = np.concatenate([gm, bm])
    lab = np.concatenate([np.ones(len(gm)), np.zeros(len(bm))])
    order = np.argsort(allm)
    ranks = np.empty(len(allm)); ranks[order] = np.arange(1, len(allm) + 1)
    n1, n0 = len(gm), len(bm)
    auc = (ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')
    print('AUC of m for separating correct from wrong: %.3f' % auc)

    print()
    print('per frame: would a consensus rule (m>=2 and score>baseline) accept a correct one?')
    rescued = 0
    wrong_accept = 0
    for i, ngood, m, is_good, cands, base in per_frame:
        if ngood == 0:
            continue
        ok = [k for k in range(len(cands))
              if m[k] >= 2 and cands[k]['score'] is not None and base is not None
              and cands[k]['score'] > base]
        if not ok:
            continue
        k = max(ok, key=lambda k: (m[k], cands[k]['score']))
        if is_good[k]:
            rescued += 1
        else:
            wrong_accept += 1
    print('   frames with at least one correct candidate: %d' % sum(1 for r in per_frame if r[1] > 0))
    print('   rescued correctly: %d    accepted a wrong one: %d' % (rescued, wrong_accept))


if __name__ == '__main__':
    main()
