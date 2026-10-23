"""Apply the consensus rule on ALL 1502 frames and score end to end.

On the dead frames the base rate of correct candidates is ~10%, so a ranker cannot be
turned into a decision.  Across the whole set the base rate is much better; if the rule
still separates there, it is deployable.
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


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


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
    keys = [str(k) for k in w['hard_keys']]

    z = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    base = {str(k): np.asarray(p, float) for k, p in zip(z['keys'], z['poses'])}

    store = []
    for i in range(len(keys)):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        cands = []
        for k, h in enumerate(m.get('hypotheses') or []):
            c = v.get('candidate_%02d' % k)
            if c is None:
                continue
            sc = (c.get('fair') or {}).get('geometry_score')
            cands.append(dict(pose=np.asarray(h['pose'], float),
                              refs=set(int(x) for x in (h.get('reference_ids') or [])),
                              score=None if sc is None else float(sc)))
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        store.append((cands, None if b is None else float(b),
                      consensus(cands) if cands else np.zeros(0, int)))

    for mt, mg in ((2, 0.0), (2, 0.05), (3, 0.0), (1, 0.10)):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[str(k)].copy()
            cands, b, m = store[i]
            if cands and b is not None:
                ok = [j for j in range(len(cands))
                      if m[j] >= mt and cands[j]['score'] is not None
                      and cands[j]['score'] > b + mg]
                if ok:
                    j = max(ok, key=lambda j: (m[j], cands[j]['score']))
                    P = cands[j]['pose'].copy()
                    used += 1
            poses.append(P)
        rows = su.evaluate(np.stack(poses), None, w, 'cca m%d mg%.2f' % (mt, mg), verbose=False)
        r = rows['hard:all1502']
        print('rule m>=%d margin %.2f -> frames %3d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (mt, mg, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
