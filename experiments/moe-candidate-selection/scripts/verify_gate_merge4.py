"""Read-only, round 4: the merge forces the coarse stage from 1500 to 4000 points.
Does gate 2 survive that? (its threshold 0.01 was tuned on 1500-point gains)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402

_G = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    mp = _G['maps'].local(T0[:3, 3], _G['maps'].nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    u = P.along_track_direction(T0)
    t = {}
    for tag, npts, strided in (('1500_prefix', 1500, False), ('4000_strided', 4000, True)):
        n = min(npts, len(scan))
        q = scan[np.linspace(0, len(scan) - 1, n).astype(int)] if strided else scan[:n]
        s0 = time.perf_counter()
        tree = cKDTree(mp)
        rot, origin = T0[:3, :3], T0[:3, 3]
        lat = np.cross(u, [0.0, 0.0, 1.0]); lat /= np.linalg.norm(lat)
        base = q @ rot.T
        c_zero = float(np.minimum(tree.query(base + origin, k=1)[0], 1.0).mean())
        home = np.array([np.minimum(tree.query(base + origin + d * lat, k=1)[0], 1.0).mean()
                         for d in F.D_HOME])
        dv = np.clip(F.D_HOME[int(np.argmin(home))] + F.D_COARSE, -2.0, 2.0)
        best = np.inf
        for s in F.S_COARSE:
            for d in dv:
                c = float(np.minimum(tree.query(base + origin + s * u + d * lat, k=1)[0], 1.0).mean())
                best = min(best, c)
        t[tag] = (c_zero - best, time.perf_counter() - s0)
    return i, t['1500_prefix'], t['4000_strided']


def main():
    init()
    n = len(_G['frames'])
    with Pool(8, initializer=init) as p:
        res = p.map(one, range(n), chunksize=8)
    g1 = np.empty(n); g4 = np.empty(n)
    t1 = np.empty(n); t4 = np.empty(n)
    for i, a, b in res:
        g1[i], t1[i] = a
        g4[i], t4[i] = b
    print('median gain  1500 pts: %.4f   4000 pts: %.4f' % (np.median(g1), np.median(g4)))
    print('correlation of the two gains: %.3f' % np.corrcoef(g1, g4)[0, 1])
    for thr in (0.005, 0.01, 0.02):
        p1 = int((g1 > thr).sum()); p4 = int((g4 > thr).sum())
        flip = int(((g1 > thr) != (g4 > thr)).sum())
        print('  threshold %.3f -> pass 1500: %4d  pass 4000: %4d  decisions flipped: %4d'
              % (thr, p1, p4, flip))
    print('\nmedian cost of the coarse stage: %.0f ms (1500) vs %.0f ms (4000)'
          % (np.median(t1) * 1000, np.median(t4) * 1000))


if __name__ == '__main__':
    main()
