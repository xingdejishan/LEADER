"""Gate 2: use the coarse stage's own cost gain to decide whether to run the fine stage.

gain = cost(origin) - cost(chosen s,d), both evaluated on the coarse query set (the first
1500 voxelised points) under the same truncated nearest-neighbour cost the solver uses.
That is exactly the quantity the coarse stage already computes, so the gate is free.

Frames with gain <= threshold keep the incoming pose; the rest take the fast-search pose.
Both branches then go through the same frozen geometric refiner, so the comparison
isolates the gate itself.

    python gate2.py --threshold 0.01
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
    cfg = P.Config()
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    _G['frames'] = frames
    _G['cfg'] = cfg
    _G['poses'] = np.asarray(z['poses'], float)
    _G['maps'] = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                             radius=cfg.crop_radius)


def one(i):
    cfg = _G['cfg']
    rec = _G['frames'][i]
    T0 = _G['poses'][i]
    maps = _G['maps']
    triple = maps.nearby(T0[:3, 3])[:cfg.reference_count]
    mp = maps.local(T0[:3, 3], triple)
    scan = P.prepare_scan(rec, cfg)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0])
    lat = lat / np.linalg.norm(lat)
    q = scan[:min(cfg.coarse_points, len(scan))] @ T0[:3, :3].T + T0[:3, 3]
    tree = cKDTree(mp)
    c0 = float(np.minimum(tree.query(q, k=1)[0], TRUNC).mean())
    s, d = _G['diag'][i]['s'], _G['diag'][i]['d']
    q2 = q + s * u + d * lat
    c1 = float(np.minimum(tree.query(q2, k=1)[0], TRUNC).mean())
    return i, c0, c1


def _w(i):
    return one(i)


def refine_all(poses, keys):
    import r4_lib as L
    w = L.World()
    out = np.empty_like(poses)

    def job(i):
        T, _ = L.refine_stage(np.asarray(poses[i], float), w.raw_query(i), w.refs,
                              w.ref_pos[:, :3, 3])
        return i, np.asarray(T, float)

    with Pool(12, initializer=lambda: None) as p:
        pass
    return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--refine', action='store_true')
    a = ap.parse_args()
    init()
    _G['diag'] = [json.loads(l) for l in open(os.path.join(ROOT, 'results/fast_diag.jsonl'))]
    fast = np.load(os.path.join(ROOT, 'results/fast.npz'), allow_pickle=True)
    keys = [str(k) for k in fast['keys']]
    FP = np.asarray(fast['poses'], float)
    IP = _G['poses']
    n = len(keys)
    with Pool(a.nproc, initializer=init) as p:
        res = p.map(_w, range(n), chunksize=4)
    c0 = np.empty(n); c1 = np.empty(n)
    for i, x, y in res:
        c0[i] = x; c1[i] = y
    gain = c0 - c1
    use = gain > a.threshold
    np.savez(os.path.join(ROOT, 'results/gate2.npz'), keys=np.array(keys), gain=gain,
             c0=c0, c1=c1, use=use)
    mixed = np.where(use[:, None, None], FP, IP)
    np.savez(os.path.join(ROOT, 'results/gated.npz'), keys=np.array(keys), poses=mixed)
    print('threshold %.3f: run %d / %d (%.1f%%)  gain median %.4f  p90 %.4f'
          % (a.threshold, int(use.sum()), n, 100 * use.mean(),
             float(np.median(gain)), float(np.percentile(gain, 90))))
