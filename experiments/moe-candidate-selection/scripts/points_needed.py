"""How many query points does the LiDAR score actually need?

The score is a mean of truncated nearest-neighbour distances.  Its reliability depends on
the number of points.  For a fixed random permutation of the query cloud, score every
candidate with the first N points and compare the induced ranking with the 4000-point
ranking (Spearman), then run the acceptance rule with the cheap score end to end.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F
import pipeline as P

NS = (1, 4, 16, 64, 256, 1024, 4000)
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg,
              maps=F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                                 for r in refs]), radius=cfg.crop_radius))


def one(args):
    i, poses = args
    cfg = _G['cfg']
    t = poses[0][:3, 3]
    mp = _G['maps'].local(t, _G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    scan = P.prepare_scan(_G['frames'][i], cfg)
    if len(scan) < 50:
        return i, None
    rng = np.random.default_rng(i)
    perm = rng.permutation(len(scan))
    out = {}
    for n in NS:
        q = scan[perm[:min(n, len(scan))]]
        out[n] = [float(np.minimum(tree.query(q @ p[:3, :3].T + p[:3, 3], k=1, workers=1)[0], 1.0).mean())
                  for p in poses]
    return i, out


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

    frames = {}
    for i, k in enumerate(keys):
        m = json.load(open(MATCH % i)); v = json.load(open(VERIFY % i))['poses']
        cs = []
        for j, h in enumerate(m.get('hypotheses') or []):
            if v.get('candidate_%02d' % j) is None:
                continue
            cs.append(dict(pose=np.asarray(h['pose'], float),
                           score=float((v['candidate_%02d' % j].get('fair') or {}).get('geometry_score') or -9)))
        b = ((v.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        if cs:
            frames[i] = (cs, -9.0 if b is None else float(b))

    rng = np.random.default_rng(7)
    subset = sorted(rng.choice(list(frames), 120, replace=False).tolist())
    todo = [(i, [c['pose'] for c in frames[i][0]] + [search[keys[i]]]) for i in subset]
    init()
    with Pool(10, initializer=init) as pool:
        res = dict(pool.map(one, todo, chunksize=4))

    print('%-6s %10s %10s' % ('points', 'Spearman', 'top1 agree'))
    for n in NS[:-1]:
        rhos, agree = [], []
        for i in subset:
            o = res.get(i)
            if not o:
                continue
            a = np.array(o[n][:-1]); ref = np.array(o[4000][:-1])
            if len(a) < 3:
                continue
            rhos.append(spearmanr(a, ref).correlation)
            agree.append(int(np.argmax(a) == np.argmax(ref)))
        print('%-6d %10.3f %9.0f%%' % (n, np.nanmean(rhos), 100 * np.mean(agree)))

    np.save('/dev/shm/score_ns.npy', dict(res=res, subset=subset,
                                          keys=keys, search=search), allow_pickle=True)


if __name__ == '__main__':
    main()
