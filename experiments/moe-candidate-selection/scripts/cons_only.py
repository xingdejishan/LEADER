"""Can consensus replace the LiDAR arbitration?  Use no map score at all.

For each frame: rank the candidates by how many other candidates from disjoint
references agree with them, take the top one, and adopt it if the agreement clears a
level.  No LiDAR scoring is used anywhere.  Compare with the arbitration-based control.
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

    frames = []
    for i in range(len(keys)):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        cs = []
        for j, h in enumerate(m.get('hypotheses') or []):
            c = v.get('candidate_%02d' % j)
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
        frames.append((cs, b, cons))

    def run(tag, choose):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = base[k].copy()
            j = choose(i)
            if j is not None:
                P = frames[i][0][j]['pose'].copy(); used += 1
            poses.append(P)
        r = su.evaluate(np.stack(poses), None, w, tag, verbose=False)['hard:all1502']
        print('%-34s frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (tag, used, r['MPE'], r['MOE'], r['success_1m_2deg']))

    run('control: arbitration (margin 0.10)',
        lambda i: (lambda cs, b, c: (max([j for j in range(len(cs)) if cs[j]['score'] > b + 0.10],
                                         key=lambda j: cs[j]['score'])
                                     if any(cs[j]['score'] > b + 0.10 for j in range(len(cs))) else None))(*frames[i]))
    for t in (1, 2):
        run('pure consensus >= %d (no map score)' % t,
            lambda i, t=t: (lambda cs, b, c: (int(np.argmax(c)) if len(cs) and c.max() >= t else None))(*frames[i]))
    run('pure consensus, any (>0)',
        lambda i: (lambda cs, b, c: (int(np.argmax(c)) if len(cs) and c.max() > 0 else None))(*frames[i]))


if __name__ == '__main__':
    main()
