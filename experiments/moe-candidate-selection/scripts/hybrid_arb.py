"""Hybrid arbitration: cheap ranking, full-precision decision.

64 points rank all candidates; only the top one and the deployed pose are then scored at
full precision (4000 points), because the accept/reject decision compares those two and
the difference is small.  Reports time and metrics.
"""
import json
import os
import sys
import time
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

N_COARSE, N_FULL = 64, 4000
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
    if len(scan) < 100:
        return i, None, 0.0
    rng = np.random.default_rng(i)
    perm = rng.permutation(len(scan))
    qc = scan[perm[:N_COARSE]]
    qf = scan[perm[:N_FULL]]
    t0 = time.perf_counter()
    cc = np.array([cost(tree, qc, p) for p in poses])
    j = int(np.argmin(cc))
    s_search = cost(tree, qf, T_search)
    s_cand = cost(tree, qf, poses[j])
    dt = time.perf_counter() - t0
    return i, (j, s_search - s_cand), dt


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

    poses_of = {}
    for i in range(len(keys)):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        cs = [np.asarray(h['pose'], float) for j, h in enumerate(m.get('hypotheses') or [])
              if v.get('candidate_%02d' % j) is not None]
        if cs:
            poses_of[i] = cs

    init()
    todo = [(i, search[keys[i]], poses_of[i]) for i in poses_of]
    with Pool(10, initializer=init) as pool:
        res = {r[0]: r[1:] for r in pool.map(one, todo, chunksize=8) if r[1] is not None}

    dt = [r[1] for r in res.values()]
    print('hybrid scoring time per frame: median %.1f ms   (coarse %d pts x K, full %d pts x 2)'
          % (1000 * float(np.median(dt)), N_COARSE, N_FULL))

    for delta in (0.0, 0.01, 0.02):
        poses, used = [], 0
        for i, k in enumerate(keys):
            P0 = search[k].copy()
            r = res.get(i)
            if r is not None:
                j, gain = r[0]
                if gain > delta:
                    P0 = poses_of[i][j].copy(); used += 1
            poses.append(P0)
        r = su.evaluate(np.stack(poses), None, w, 'hybrid d=%.2f' % delta, verbose=False)['hard:all1502']
        print('delta %.2f -> frames %4d   MPE %.4f   MOE %.4f   ok1/2 %d'
              % (delta, used, r['MPE'], r['MOE'], r['success_1m_2deg']))


if __name__ == '__main__':
    main()
