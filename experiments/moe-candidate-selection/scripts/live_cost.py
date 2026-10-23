"""Fix the comparison: score the DEPLOYED pose (the search output) with the same cost
function used for the candidate, instead of comparing against V7's frozen score.

Rule:
  accept the highest-scoring candidate if
      (its frozen score beats V7's by 0.10)                      [current rule]
   or (the frame holds a corroborated candidate, max_cons>=1,
       and its live map cost beats the deployed pose's cost)     [new path]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F
import pipeline as P

TAU_T, TAU_R = 3.0, 5.0
PTS = 2000
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg,
              maps=F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                                 for r in refs]), radius=cfg.crop_radius))


def cost(tree, q, T):
    return float(np.minimum(tree.query(q @ T[:3, :3].T + T[:3, 3], k=1, workers=1)[0], 1.0).mean())


def one(args):
    i, T_search, poses = args
    cfg = _G['cfg']
    t = T_search[:3, 3]
    mp = _G['maps'].local(t, _G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    scan = P.prepare_scan(_G['frames'][i], cfg)
    q = scan if len(scan) <= PTS else scan[::max(1, len(scan) // PTS)][:PTS]
    return i, (cost(tree, q, T_search), np.array([cost(tree, q, p) for p in poses]))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = [str(k) for k in w['hard_keys']]
    zz = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    search = {str(k): np.asarray(p, float) for k, p in zip(zz['keys'], zz['poses'])}

    def ang(a, b):
        from scipy.spatial.transform import Rotation
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))

    frames = []
    for i, k in enumerate(keys):
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
        frames.append(dict(cs=cs, base=b, cons=cons))

    init()
    todo = [(i, search[keys[i]], [c['pose'] for c in frames[i]['cs']])
            for i in range(len(keys)) if frames[i]['cs']]
    with Pool(10, initializer=init) as pool:
        live = dict(pool.map(one, todo, chunksize=8))

    for delta in (None, 0.0, 0.01, 0.02):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P = search[k].copy()
            fr = frames[i]
            cs, b, cons = fr['cs'], fr['base'], fr['cons']
            if cs:
                main_ok = [j for j in range(len(cs)) if cs[j]['score'] > b + 0.10]
                if main_ok:
                    j = max(main_ok, key=lambda j: cs[j]['score'])
                    P = cs[j]['pose'].copy(); used += 1
                elif delta is not None and i in live and cons.max() >= 1:
                    c_s, c_c = live[i]
                    ok = [j for j in range(len(cs)) if c_c[j] < c_s - delta]
                    if ok:
                        j = max(ok, key=lambda j: (c_s - c_c[j], cs[j]['score']))
                        P = cs[j]['pose'].copy(); used += 1
            poses.append(P)
        tag = 'control' if delta is None else 'consensus+livecost delta=%.2f' % delta
        r = su.evaluate(np.stack(poses), None, w, tag, verbose=False)['hard:all1502']
        print('%-32s frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (tag, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
