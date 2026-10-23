"""End-to-end driver for the current model (single frame in -> corrected pose out).

Stages, in order:
    1. gate 2      coarse grid cost gain decides whether the fine search runs
    2. search      fast coarse->fine along-track x lateral search   (rotation frozen)

Stage 0 (V7 appearance field) and the visual-candidate branch are upstream /
conditional and are not run here: they need the GPU model and the frozen candidate
tables respectively.

    python run_pipeline.py --frames 1502 --workers 12 --gate2 0.01
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
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402

ROOT = HERE
TRUNC = 1.0
_G = {}
_OPTS = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))
    _OPTS.update(_OPTS)


def cost(tree, pts):
    return float(np.minimum(tree.query(pts, k=1, workers=1)[0], TRUNC).mean())


def coarse_gain(tree, q, origin, u, lat):
    c_zero = cost(tree, q + origin)
    home = [cost(tree, q + origin + d * lat) for d in F.D_HOME]
    d_home = float(F.D_HOME[int(np.argmin(home))])
    best = np.inf
    for s in F.S_COARSE:
        for d in np.clip(d_home + F.D_COARSE, -2.0, 2.0):
            c = cost(tree, q + origin + s * u + d * lat)
            if c < best:
                best = c
    return c_zero - best




def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0]); lat = lat / np.linalg.norm(lat)
    qc = scan[:min(cfg.coarse_points, len(scan))]
    tree = cKDTree(mp)
    gain = coarse_gain(tree, qc @ T0[:3, :3].T, T0[:3, 3], u, lat)
    diag = dict(key=_G['frames'][i]['key'], gain=float(gain), ran_fine=False, rotated=False)
    T = np.array(T0, float)
    if gain > _OPTS['gate2']:
        pose, d = P.localize(_G['frames'][i], T0, maps, cfg)
        T = np.asarray(pose, float)
        diag.update(ran_fine=True, s=float(d['s']), d=float(d['d']))
    return i, T, diag


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=0, help='0 = all')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--gate2', type=float, default=0.01)
    ap.add_argument('--out', default='results/pipeline_no_rotation.npz')
    a = ap.parse_args()
    _OPTS.update(gate2=a.gate2)
    init()
    n = len(_G['frames'])
    idx = list(range(n if a.frames <= 0 else min(a.frames, n)))
    with Pool(a.workers, initializer=init) as p:
        res = p.map(_w, idx, chunksize=4)
    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']][:len(idx)]
    poses = np.stack([r[1] for r in res])
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    np.savez(os.path.join(ROOT, a.out), keys=np.array(keys), poses=poses)
    ran = sum(1 for r in res if r[2]['ran_fine'])
    print('frames %d  fine ran %d (%.1f%%)  gate2=%.3f rotation_refinement=removed'
          % (len(idx), ran, 100 * ran / len(idx), a.gate2))
    with open(os.path.join(ROOT, 'results/pipeline_no_rotation_diag.jsonl'), 'w') as fh:
        for _, _, d in res:
            fh.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    main()
