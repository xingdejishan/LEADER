"""The additive form: keep the current rule, and add consensus as a second acceptance path
only on frames the current rule rejects."""
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

    for mt in (99, 2, 3):
        poses, cur, add = [], 0, 0
        for i, k in enumerate(keys):
            P = base[str(k)].copy()
            m = json.load(open(MATCH % i))
            v = json.load(open(VERIFY % i))['poses']
            hyps = m.get('hypotheses') or []
            cands = []
            for j, h in enumerate(hyps):
                c = v.get('candidate_%02d' % j)
                if c is None:
                    continue
                sc = (c.get('fair') or {}).get('geometry_score')
                cands.append(dict(pose=np.asarray(h['pose'], float),
                                  refs=set(int(x) for x in (h.get('reference_ids') or [])),
                                  score=None if sc is None else float(sc)))
            b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
            if not cands or b is None:
                poses.append(P); continue
            n = len(cands)
            cons = np.zeros(n, int)
            for a in range(n):
                for c2 in range(n):
                    if a == c2:
                        continue
                    if cands[a]['refs'] and cands[c2]['refs'] and (cands[a]['refs'] & cands[c2]['refs']):
                        continue
                    if (np.linalg.norm(cands[a]['pose'][:3, 3] - cands[c2]['pose'][:3, 3]) <= TAU_T
                            and angle(cands[a]['pose'][:3, :3], cands[c2]['pose'][:3, :3]) <= TAU_R):
                        cons[a] += 1
            main_ok = [j for j in range(n)
                       if cands[j]['score'] is not None and cands[j]['score'] > b + 0.10]
            if main_ok:
                j = max(main_ok, key=lambda j: cands[j]['score'])
                P = cands[j]['pose'].copy(); cur += 1
            else:
                ok = [j for j in range(n) if cons[j] >= mt
                      and cands[j]['score'] is not None and cands[j]['score'] > b]
                if ok:
                    j = max(ok, key=lambda j: (cons[j], cands[j]['score']))
                    P = cands[j]['pose'].copy(); add += 1
            poses.append(P)
        rows = su.evaluate(np.stack(poses), None, w, 'cca-add m%d' % mt, verbose=False)
        r = rows['hard:all1502']
        print('current rule %3d frames + consensus %3d frames (m>=%d)  -> MPE %.4f  MOE %.4f  ok1/2 %d'
              % (cur, add, mt, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
