"""Even-strided point subsets + does the flagged set still cover the 29 adopted frames?

C5  4000 pts even-stride, 1.00 m
C6  2000 pts even-stride, 1.00 m
C7  4000 pts even-stride, 0.50 m   (stride-free resolution check on the good point budget)

Compares each against A (frozen gate) and against the 29 frames the visual route
actually won in the shipped run.
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
OUT = os.path.join(HERE, 'merge_probe3.jsonl')
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


def stride(points, n):
    if len(points) <= n:
        return points
    step = max(1, len(points) // n)
    return points[::step][:n]


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
    q4 = stride(scan, 4000)
    q2 = stride(scan, 2000)
    c5, t5 = var(tree, q4, q4 @ rot + origin, u, S1)
    c6, t6 = var(tree, q2, q2 @ rot + origin, u, S1)
    c7, t7 = var(tree, q4, q4 @ rot + origin, u, S5)
    return dict(i=i, c5=c5, c6=c6, c7=c7, t5=t5, t6=t6, t7=t7)


def _w(i):
    return one(i)


def main():
    init()
    n = len(_G['frames'])
    with Pool(12, initializer=init) as p:
        rows = p.map(_w, range(n), chunksize=8)
    with open(OUT, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')

    a = np.array([json.loads(l)['a'] for l in open(os.path.join(HERE, 'merge_probe.jsonl'))])
    adopted = [i for i, l in enumerate(open(os.path.join(MOE, 'results/three_route_diag.jsonl')))
               if json.loads(l).get('visual')]
    fa = a < 0.21
    print('A flagged %d   adopted-by-visual %d   (all adopted inside A: %s)'
          % (fa.sum(), len(adopted), all(fa[i] for i in adopted)))
    for nm, key, tk in (('C5 4k even/1.0m', 'c5', 't5'), ('C6 2k even/1.0m', 'c6', 't6'), ('C7 4k even/0.5m', 'c7', 't7')):
        x = np.array([r[key] for r in rows])
        t = np.array([r[tk] for r in rows])
        fx = x < 0.21
        kept = int(sum(fx[i] for i in adopted))
        print('%-16s flagged %3d  agree %5.1f%%  recallA %.2f  extra %3d  missed %2d  adopted kept %d/%d  ms %.0f'
              % (nm, fx.sum(), 100 * (fa == fx).mean(), (fx & fa).sum() / max(fa.sum(), 1),
                 int((fx & ~fa).sum()), int((~fx & fa).sum()), kept, len(adopted), np.median(t)))


if __name__ == '__main__':
    main()
