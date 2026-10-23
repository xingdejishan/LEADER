"""Operating points for the consensus rule, and the end-to-end effect of the best one."""
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


def load(i):
    m = json.load(open(MATCH % i))
    v = json.load(open(VERIFY % i))['poses']
    out = []
    for k, h in enumerate(m.get('hypotheses') or []):
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
            if (np.linalg.norm(cands[a]['pose'][:3, 3] - cands[b]['pose'][:3, 3]) <= TAU_T
                    and angle(cands[a]['pose'][:3, :3], cands[b]['pose'][:3, :3]) <= TAU_R):
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
    dead = np.isfinite(ratio) & (ratio < 0.21) & ~visual

    frames = []
    for i in np.flatnonzero(dead):
        cands, base = load(i)
        if not cands or base is None:
            continue
        m = consensus(cands)
        good = np.array([np.linalg.norm(c['pose'][:3, 3] - gt[i][:3, 3]) <= GOOD_T
                         and angle(c['pose'][:3, :3], gt[i][:3, :3]) <= GOOD_R for c in cands])
        frames.append((int(i), m, good, cands, base))

    print('%-28s %8s %8s %8s' % ('rule', 'rescued', 'wrong', 'precision'))
    best = None
    for mt in (1, 2, 3):
        for mg in (0.0, 0.05, 0.10):
            for need_ver in (False, True):
                res = wrong = 0
                for i, m, good, cands, base in frames:
                    ok = [k for k in range(len(cands))
                          if m[k] >= mt and cands[k]['score'] is not None
                          and cands[k]['score'] > base + mg
                          and (cands[k]['ver'] or not need_ver)]
                    if not ok:
                        continue
                    k = max(ok, key=lambda k: (m[k], cands[k]['score']))
                    if good[k]:
                        res += 1
                    else:
                        wrong += 1
                if res + wrong == 0:
                    continue
                prec = res / (res + wrong)
                tag = 'm>=%d margin %.2f %s' % (mt, mg, 'verified' if need_ver else '')
                print('%-28s %8d %8d %8.0f%%' % (tag, res, wrong, 100 * prec))
                if best is None or (res - 2 * wrong) > best[0]:
                    best = (res - 2 * wrong, mt, mg, need_ver, res, wrong)
    print()
    print('chosen operating point:', best)


if __name__ == '__main__':
    main()
