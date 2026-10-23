"""Use consensus as a FRAME gate and the score as the candidate selector.

Previous failure: consensus was used to pick which candidate.  The frame-level result
says its real signal is "this frame holds a corroborated candidate at all".  So:
  accept = (current margin rule)  OR  (frame has max_cons >= 1, then take the best score)
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


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = [str(k) for k in w['hard_keys']]
    zz = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    base = {str(k): np.asarray(p, float) for k, p in zip(zz['keys'], zz['poses'])}

    per_frame = []
    for i in range(len(keys)):
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
                           score=float(f.get('geometry_score') if f.get('geometry_score') is not None else -9.0)))
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        b = -9.0 if b is None else float(b)
        cons = np.zeros(len(cs))
        for a in range(len(cs)):
            for c2 in range(len(cs)):
                if a == c2:
                    continue
                if cs[a]['refs'] and cs[c2]['refs'] and (cs[a]['refs'] & cs[c2]['refs']):
                    continue
                if (np.linalg.norm(cs[a]['pose'][:3, 3] - cs[c2]['pose'][:3, 3]) <= TAU_T
                        and ang(cs[a]['pose'][:3, :3], cs[c2]['pose'][:3, :3]) <= TAU_R):
                    cons[a] += 1
        per_frame.append((cs, b, cons))

    def run(tag, gate_cons, margin, extra_score):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[k].copy()
            cs, b, cons = per_frame[i]
            if cs:
                main_ok = [j for j in range(len(cs)) if cs[j]['score'] > b + margin]
                if main_ok:
                    j = max(main_ok, key=lambda j: cs[j]['score'])
                    P = cs[j]['pose'].copy(); used += 1
                elif cons.max() >= gate_cons and (not extra_score or
                                                 max(c['score'] for c in cs) > b):
                    j = max(range(len(cs)), key=lambda j: cs[j]['score'])
                    P = cs[j]['pose'].copy(); used += 1
            poses.append(P)
        r = su.evaluate(np.stack(poses), None, w, tag, verbose=False)['hard:all1502']
        print('%-34s frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (tag, used, r['MPE'], r['MOE'], r['success_1m_2deg']))

    run('control (margin only)', 99, 0.10, False)
    for g in (1, 2):
        for es in (False, True):
            run('margin + consgate>=%d%s' % (g, ' +score' if es else ''), g, 0.10, es)


if __name__ == '__main__':
    main()
