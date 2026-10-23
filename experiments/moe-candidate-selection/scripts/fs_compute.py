"""Relative free-space consistency as a per-candidate feature (stage 1: compute).

For a candidate pose, march along the scan's beams and ask whether the map has a surface
where the beam says empty.  Used only relative to the frame's other candidates, so the
shared bias cancels - the absolute version of this (the historical beam term) saturated.
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

N_Q = 250
FRACS = (0.35, 0.55, 0.75)
HIT_R = 0.35
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg,
              maps=F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                                 for r in refs]), radius=cfg.crop_radius))


def violation(tree, q, T):
    if len(q) < 10:
        return float('nan')
    t = T[:3, 3]
    W = q @ T[:3, :3].T + t
    P0 = np.vstack([t + f * (W - t) for f in FRACS])
    d, _ = tree.query(P0, k=1, workers=1)
    return float((d < HIT_R).mean())


def one(args):
    i, poses = args
    try:
        return _one(i, poses)
    except Exception:
        return i, np.full(len(poses), np.nan)


def _one(i, poses):
    cfg = _G['cfg']
    t = poses[0][:3, 3]
    mp = _G['maps'].local(t, _G['maps'].nearby(t)[:cfg.reference_count])
    if len(mp) < 500:
        return i, np.zeros(len(poses))
    tree = cKDTree(mp)
    scan = P.prepare_scan(_G['frames'][i], cfg)
    scan = scan[np.linalg.norm(scan, axis=1) > 2.0]
    if len(scan) < 50:
        return i, np.zeros(len(poses))
    q = scan if len(scan) <= N_Q else scan[::max(1, len(scan) // N_Q)][:N_Q]
    return i, np.array([violation(tree, q, p) for p in poses])


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    v7 = np.asarray(w['hard_base'], float)

    cands = {}
    for i in range(len(w['hard_keys'])):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        ps = []
        for k, h in enumerate(m.get('hypotheses') or []):
            if v.get('candidate_%02d' % k) is None:
                continue
            ps.append(np.asarray(h['pose'], float))
        if ps:
            cands[i] = ps

    init()
    todo = [(i, cands[i] + [v7[i]]) for i in cands]
    with Pool(10, initializer=init) as pool:
        res = dict(pool.map(one, todo, chunksize=8))

    idx, vio = [], []
    for i in cands:
        if i in res:
            idx.append(i)
            vio.append(res[i])
    np.save('/dev/shm/freespace.npy', dict(idx=np.array(idx, int), vio=vio), allow_pickle=True)
    print('frames %d   candidates %d   mean violation rate %.3f'
          % (len(idx), sum(len(r) - 1 for r in vio), float(np.mean([np.mean(r[:-1]) for r in vio]))))


if __name__ == '__main__':
    main()
