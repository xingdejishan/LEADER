"""Gate 2, deployable form: the coarse grid's own cost drop decides the fine stage.

Replicates exactly the coarse stage the fast solver runs (anchor over D_HOME at s=0,
then 17 along-track x 3 lateral at 1500 points), so the gate needs no extra computation
beyond what the search already does.

For each frame it records:
    c_zero        cost at the incoming pose (s=0, d=0)
    c_coarse      best cost on the coarse grid, and the (s, d) that achieves it
    gain          c_zero - c_coarse

then writes two gated pose tables:
    skip_all   gated-out frames keep the incoming pose          (no search at all)
    skip_fine  gated-out frames take the coarse pose            (coarse only)

    python gate2b.py
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402

ROOT = os.path.dirname(HERE)
TRUNC = 1.0
_G = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))


def cost(tree, pts):
    return float(np.minimum(tree.query(pts, k=1, workers=1)[0], TRUNC).mean())


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    rot = T0[:3, :3]
    origin = T0[:3, 3]
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0]); lat = lat / np.linalg.norm(lat)
    q = scan[:min(cfg.coarse_points, len(scan))] @ rot.T
    tree = cKDTree(mp)
    c_zero = cost(tree, q + origin)
    home = np.array([cost(tree, q + origin + d * lat) for d in F.D_HOME])
    d_home = float(F.D_HOME[int(np.argmin(home))])
    dv = np.clip(d_home + F.D_COARSE, -2.0, 2.0)
    best = (np.inf, 0.0, 0.0)
    for s in F.S_COARSE:
        for d in dv:
            c = cost(tree, q + origin + s * u + d * lat)
            if c < best[0]:
                best = (c, float(s), float(d))
    return i, c_zero, best[0], best[1], best[2]


def _w(i):
    return one(i)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=12)
    a = ap.parse_args()
    init()
    n = len(_G['frames'])
    with Pool(a.nproc, initializer=init) as p:
        res = p.map(_w, range(n), chunksize=4)
    gain = np.empty(n); s_coarse = np.empty(n); d_coarse = np.empty(n)
    c_zero = np.empty(n); c_coarse = np.empty(n)
    for i, z0, zc, s, d in res:
        c_zero[i] = z0; c_coarse[i] = zc; s_coarse[i] = s; d_coarse[i] = d
    gain[:] = c_zero - c_coarse
    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']]
    np.savez(os.path.join(ROOT, 'results/gate2_coarse.npz'), keys=np.array(keys),
             gain=gain, c_zero=c_zero, c_coarse=c_coarse, s=s_coarse, d=d_coarse)
    print('frames %d  gain median %.4f p75 %.4f p90 %.4f  frac>0 %.1f%%'
          % (n, np.median(gain), np.percentile(gain, 75), np.percentile(gain, 90),
             100 * (gain > 0).mean()))
