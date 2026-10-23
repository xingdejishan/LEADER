"""Split the resolution effect: is it the point budget or the along-track step?

Reuses variant A (frozen gate) ratios from merge_probe.jsonl; recomputes only the
route-3-map variants:
  C2  4000 pts, 1.00 m
  C3  1500 pts, 0.50 m
  C4  2000 pts, 0.50 m
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import pipeline as P          # noqa: E402
import fast_lib as F          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'merge_probe2.jsonl')
S1 = np.round(np.arange(-10.0, 10.0001, 1.0), 3)
S5 = np.round(np.arange(-10.0, 10.0001, 0.5), 3)
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    z = np.load(os.path.join(MOE, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))


def ratio_from(costs, s):
    costs = np.asarray(costs, float)
    far = np.abs(s) > 5.0
    c0 = float(costs.min())
    cf = float(costs[far].min())
    return (cf - c0) / c0 if c0 > 0 else float('inf')


def var(tree, q, base, u, s):
    t0 = time.perf_counter()
    costs = [float(np.minimum(tree.query(base + si * u, k=1, workers=1)[0], 1.0).mean()) for si in s]
    return ratio_from(costs, s), (time.perf_counter() - t0) * 1000


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    tree = cKDTree(mp)
    scan = P.prepare_scan(_G['frames'][i], cfg)
    u = P.along_track_direction(T0)
    rot = T0[:3, :3].T
    origin = T0[:3, 3]
    q15 = scan[:min(1500, len(scan))]
    q20 = scan[:min(2000, len(scan))]
    q40 = scan[:min(4000, len(scan))]
    c2, t2 = var(tree, q40, q40 @ rot + origin, u, S1)
    c3, t3 = var(tree, q15, q15 @ rot + origin, u, S5)
    c4, t4 = var(tree, q20, q20 @ rot + origin, u, S5)
    return dict(i=i, c2=c2, c3=c3, c4=c4, t2=t2, t3=t3, t4=t4)


def _w(i):
    return one(i)


def main():
    init()
    print('crop_radius %.1f  reference_count %d  coarse_points %d'
          % (_G['cfg'].crop_radius, _G['cfg'].reference_count, _G['cfg'].coarse_points))
    n = len(_G['frames'])
    with Pool(12, initializer=init) as p:
        rows = p.map(_w, range(n), chunksize=8)
    with open(OUT, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    a = np.array([json.loads(l)['a'] for l in open(os.path.join(HERE, 'merge_probe.jsonl'))])
    TH = 0.21
    fa = a < TH
    print('A flagged %d' % fa.sum())
    for nm, key, tk in (('C2 4k/1.0m', 'c2', 't2'), ('C3 1.5k/0.5m', 'c3', 't3'), ('C4 2k/0.5m', 'c4', 't4')):
        x = np.array([r[key] for r in rows])
        t = np.array([r[tk] for r in rows])
        fx = x < TH
        print('%-12s flagged %3d  agree %5.1f%%  recallA %.2f  extra %3d  missed %2d  corr %.3f  ms %.0f'
              % (nm, fx.sum(), 100 * (fa == fx).mean(), (fx & fa).sum() / max(fa.sum(), 1),
                 int((fx & ~fa).sum()), int((~fx & fa).sum()), np.corrcoef(a, x)[0, 1], np.median(t)))


if __name__ == '__main__':
    main()
