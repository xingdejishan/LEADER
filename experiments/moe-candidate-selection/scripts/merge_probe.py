"""Can the along-track ambiguity ratio be read off route 3's own coarse surface?

Three variants per frame, all on the 1502 hard frames:
  A  frozen gate        : gate's own map (3 refs, 60 m), 4000 pts, +-10 m at 0.50 m, lateral 0
  B  route3 map, fine   : route 3's map, 4000 pts, +-10 m at 0.50 m, lateral 0
  C  route3 map, coarse : route 3's map, 1500 pts, +-10 m at 1.00 m, lateral 0

A vs B isolates the map/point-count change; B vs C isolates the resolution change.
Writes results as JSONL so the comparison can be redone without recomputation.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
GATE = '/home/zhangcy/projects/e2e-three-route-gate-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
sys.path.insert(0, GATE)

import pipeline as P          # noqa: E402
import fast_lib as F          # noqa: E402
import gate as G1             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merge_probe.jsonl')
S_FINE = np.round(np.arange(-10.0, 10.0001, 0.5), 3)
S_COARSE = np.round(np.arange(-10.0, 10.0001, 1.0), 3)
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    z = np.load(os.path.join(MOE, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius),
              lib=G1.ReferenceLibrary())


def ratio_from(costs, s):
    costs = np.asarray(costs, float)
    far = np.abs(s) > 5.0
    c0 = float(costs.min())
    cf = float(costs[far].min())
    return (cf - c0) / c0 if c0 > 0 else float('inf')


def surface(tree, q, base, u, s):
    return [float(np.minimum(tree.query(base + si * u, k=1, workers=1)[0], 1.0).mean()) for si in s]


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

    t0 = time.perf_counter()
    a = float(G1.ambiguity_ratio(_G['lib'], _G['frames'][i]['scan_path'], T0)['ratio'])
    ta = time.perf_counter() - t0

    q4 = scan[:min(4000, len(scan))]
    t0 = time.perf_counter()
    b = ratio_from(surface(tree, q4, q4 @ rot + origin, u, S_FINE), S_FINE)
    tb = time.perf_counter() - t0

    q15 = scan[:min(1500, len(scan))]
    t0 = time.perf_counter()
    c = ratio_from(surface(tree, q15, q15 @ rot + origin, u, S_COARSE), S_COARSE)
    tc = time.perf_counter() - t0

    return dict(i=i, a=a, b=b, c=c, ta=ta, tb=tb, tc=tc,
                map_points=int(len(mp)), scan_points=int(len(scan)))


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
    a = np.array([r['a'] for r in rows])
    b = np.array([r['b'] for r in rows])
    c = np.array([r['c'] for r in rows])
    ta = np.array([r['ta'] for r in rows]) * 1000
    tb = np.array([r['tb'] for r in rows]) * 1000
    tc = np.array([r['tc'] for r in rows]) * 1000
    TH = 0.21
    print('frames %d' % n)
    print('flagged  A(frozen) %d   B(route3map,4k,0.5m) %d   C(route3map,1.5k,1m) %d'
          % (int((a < TH).sum()), int((b < TH).sum()), int((c < TH).sum())))
    print('ms   A %.0f   B %.0f   C %.0f' % (np.median(ta), np.median(tb), np.median(tc)))
    for nm, x in (('B', b), ('C', c)):
        fa, fx = a < TH, x < TH
        print('%s: agreement %.1f%%   recall of A %.2f   extra flags %d   missed %d'
              % (nm, 100.0 * (fa == fx).mean(), (fx & fa).sum() / max(fa.sum(), 1),
                 int((fx & ~fa).sum()), int((~fx & fa).sum())))
        print('   corr %.3f   ratio B/A median %.3f' % (np.corrcoef(a, x)[0, 1], np.median(x / np.maximum(a, 1e-9))))
    # what threshold on C reproduces A's flag count?
    order = np.argsort(c)
    k = int((a < TH).sum())
    thr = float(np.sort(c)[k - 1])
    fc = c <= thr
    fa = a < TH
    print('C at matched count %d (thr %.3f): overlap with A = %d, A missed = %d'
          % (fc.sum(), thr, int((fc & fa).sum()), int((fa & ~fc).sum())))


if __name__ == '__main__':
    main()
