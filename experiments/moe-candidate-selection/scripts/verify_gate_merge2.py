"""Read-only, round 2: which reading of the merged gate reproduces ~85-84 flags at
~99% per-frame agreement? Variants differ in the along-track range/step, whether the
lateral offsets are searched, the point count, and prefix vs strided selection.
"""
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(ROOT, 'stages/gate1'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402

TRUNC, FAR, THR = 1.0, 5.0, 0.21
_G = {}
S8_1 = np.round(np.arange(-8.0, 8.0001, 1.0), 3)
S10_1 = np.round(np.arange(-10.0, 10.0001, 1.0), 3)
LAT = np.array([-0.5, 0.0, 0.5])


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))


def cost(tree, pts):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=TRUNC)[0]
    return float(np.where(np.isfinite(d), d, TRUNC).mean())


def ratio(tree, Q, t0, u, lat, s_grid, use_lat):
    ds = LAT if use_lat else np.array([0.0])
    costs = np.empty((len(s_grid), len(ds)))
    for a, s in enumerate(s_grid):
        for b, dd in enumerate(ds):
            costs[a, b] = cost(tree, Q + (t0 + s * u + dd * lat))
    c0 = float(costs.min())
    cfar = float(costs[np.abs(s_grid) > FAR].min())
    return (cfar - c0) / max(c0, 1e-6)


def pts(scan, n, strided):
    n = min(n, len(scan))
    if strided:
        return scan[np.linspace(0, len(scan) - 1, n).astype(int)]
    return scan[:n]


VARIANTS = [
    ('s8/1.0/no-lat/4000/prefix', S8_1, False, 4000, False),
    ('s10/1.0/no-lat/4000/prefix', S10_1, False, 4000, False),
    ('s8/1.0/no-lat/4000/strided', S8_1, False, 4000, True),
    ('s8/1.0/lat/4000/strided', S8_1, True, 4000, True),
    ('s8/1.0/no-lat/1500/prefix', S8_1, False, 1500, False),
    ('s8/1.0/no-lat/2000/prefix', S8_1, False, 2000, False),
]


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    mp = _G['maps'].local(T0[:3, 3], _G['maps'].nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0])
    lat /= np.linalg.norm(lat)
    t0 = T0[:3, 3]
    out = []
    for _name, grid, use_lat, npts, strided in VARIANTS:
        Q = pts(scan, npts, strided) @ T0[:3, :3].T
        out.append(ratio(tree, Q, t0, u, lat, grid, use_lat))
    return i, out


def main():
    init()
    n = len(_G['frames'])
    with Pool(8, initializer=init) as p:
        res = p.map(one, range(n), chunksize=8)
    R = np.empty((len(VARIANTS), n))
    for i, vals in res:
        for k, v in enumerate(vals):
            R[k, i] = v

    diag = [json.loads(l) for l in open(
        os.path.join(ROOT, 'results/three_route_unified_diag.jsonl'))]
    old = np.array([d['ratio'] for d in diag])
    fo = old < THR
    print('old flags=%d\n' % int(fo.sum()))
    print('%-30s %5s %8s %7s %7s %8s' % ('variant', 'flags', 'agree', 'lost', 'gained', 'in[.21,.3)'))
    for k, (name, *_rest) in enumerate(VARIANTS):
        fn = R[k] < THR
        print('%-30s %5d %7.1f%% %7d %7d %8d'
              % (name, int(fn.sum()), (fo == fn).mean() * 100,
                 int((fo & ~fn).sum()), int((~fo & fn).sum()),
                 int(((R[k] >= THR) & (R[k] < 0.30)).sum())))

    # end-to-end for the two most promising variants
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    hk, gt = w['hard_keys'], w['hard_gt']
    z3 = np.load(os.path.join(ROOT, 'results/final.npz'), allow_pickle=True)
    r3 = {str(k): np.asarray(v, float) for k, v in zip(z3['keys'], z3['poses'])}
    zu = np.load(os.path.join(ROOT, 'results/three_route_unified.npz'), allow_pickle=True)
    uni = {str(k): np.asarray(v, float) for k, v in zip(zu['keys'], zu['poses'])}
    VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
              'verify-frozen_nopruneoptREF/%06d.json')
    vis_pose = {}
    for i in range(n):
        rec = json.load(open(VERIFY % i))['poses']
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        if best is not None:
            vis_pose[hk[i]] = best[1]

    def cost_at(pose, q):
        c = pose[:3, 3]
        mp = _G['maps'].local(c, _G['maps'].nearby(c)[:_G['cfg'].reference_count])
        return cost(cKDTree(mp), q @ pose[:3, :3].T + c)

    for k, (name, *_rest) in enumerate(VARIANTS[:4]):
        fn = R[k] < THR
        poses = {}
        for i, key in enumerate(hk):
            if fo[i] and fn[i]:
                poses[key] = uni[key]
            elif fo[i] and not fn[i]:
                poses[key] = r3[key]
            elif fn[i] and not fo[i]:
                q = P.prepare_scan(_G['frames'][i], _G['cfg'])[:4000]
                c3 = cost_at(r3[key], q)
                cv = cost_at(vis_pose[key], q) if key in vis_pose else np.inf
                poses[key] = vis_pose[key] if cv < c3 - 0.05 else r3[key]
            else:
                poses[key] = r3[key]
        A = np.stack([poses[x] for x in hk])
        e = su.pos_err(A, gt)
        rot = su.rot_deg(A[:, :3, :3], gt[:, :3, :3])
        print('\n[%s] MPE=%.4f MOE=%.4f ok=%d >10m=%d >5m=%d'
              % (name, e.mean(), rot.mean(), int(((e <= 1) & (rot <= 2)).sum()),
                 int((e > 10).sum()), int((e > 5).sum())))
    print('\n(current unified: MPE=0.9761 MOE=2.1707 ok=727 >10m=3 >5m=10)')


if __name__ == '__main__':
    main()
