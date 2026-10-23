"""Read-only, round 3: close the remaining gaps.
  - timing breakdown of the frozen gate vs building the ratio surface;
  - along-track step 0.5 vs 1.0 m (no lateral, strided, 4000 points);
  - the 2000-point variant: does the accepted count really fall 29 -> 23;
  - is a looser gate (0.30) actually harmful under the cost-based arbiter?
"""
import importlib.util
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
sys.path.insert(0, os.path.join(ROOT, 'stages/gate1'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402
import gate as G1  # noqa: E402

TRUNC, FAR, THR = 1.0, 5.0, 0.21
_G = {}
S1 = np.round(np.arange(-8.0, 8.0001, 1.0), 3)
S05 = np.round(np.arange(-8.0, 8.0001, 0.5), 3)


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius),
              lib=G1.ReferenceLibrary())


def cost(tree, pts):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=TRUNC)[0]
    return float(np.where(np.isfinite(d), d, TRUNC).mean())


def ratio(tree, Q, t0, u, s_grid):
    c = np.array([cost(tree, Q + (t0 + s * u)) for s in s_grid])
    c0, cfar = float(c.min()), float(c[np.abs(s_grid) > FAR].min())
    return (cfar - c0) / max(c0, 1e-6)


def pts(scan, n, strided):
    n = min(n, len(scan))
    return scan[np.linspace(0, len(scan) - 1, n).astype(int)] if strided else scan[:n]


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    mp = _G['maps'].local(T0[:3, 3], _G['maps'].nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    t0 = T0[:3, 3]
    R0 = T0[:3, :3]
    r1 = ratio(tree, pts(scan, 4000, True) @ R0.T, t0, u, S1)
    r05 = ratio(tree, pts(scan, 4000, True) @ R0.T, t0, u, S05)
    r2k = ratio(tree, pts(scan, 2000, False) @ R0.T, t0, u, S1)
    return i, (r1, r05, r2k)


def timing():
    cfg = _G['cfg']
    t_gate, t_tree, t_surf = [], [], []
    for i in range(24):
        rec, T0 = _G['frames'][i], _G['poses'][i]
        s = time.perf_counter()
        G1.ambiguity_ratio(_G['lib'], rec['scan_path'], T0)
        t_gate.append(time.perf_counter() - s)

        mp = _G['maps'].local(T0[:3, 3], _G['maps'].nearby(T0[:3, 3])[:cfg.reference_count])
        scan = P.prepare_scan(rec, cfg)
        u = P.along_track_direction(T0)
        Q = pts(scan, 4000, True) @ T0[:3, :3].T
        s = time.perf_counter()
        tree = cKDTree(mp)
        t_tree.append(time.perf_counter() - s)
        s = time.perf_counter()
        ratio(tree, Q, T0[:3, 3], u, S1)
        t_surf.append(time.perf_counter() - s)
    med = lambda v: np.median(v) * 1000
    print('\n-- timing over 24 frames, medians (ms) --')
    print('  frozen gate, its own map each frame : %.0f' % med(t_gate))
    print('  KD-tree build on the shared map     : %.0f' % med(t_tree))
    print('  17-position ratio sweep, 4000 points: %.0f' % med(t_surf))
    print('  merged gate (tree + sweep)          : %.0f' % (med(t_tree) + med(t_surf)))


def main():
    init()
    timing()
    n = len(_G['frames'])
    with Pool(8, initializer=init) as p:
        res = p.map(one, range(n), chunksize=8)
    r1 = np.empty(n); r05 = np.empty(n); r2k = np.empty(n)
    for i, v in res:
        r1[i], r05[i], r2k[i] = v

    diag = [json.loads(l) for l in open(
        os.path.join(ROOT, 'results/three_route_unified_diag.jsonl'))]
    old = np.array([d['ratio'] for d in diag])
    fo = old < THR
    print('\n-- gate variants (old flags=%d) --' % int(fo.sum()))
    for tag, r in (('1.0 m step', r1), ('0.5 m step', r05), ('2000 pts, prefix', r2k)):
        fn = r < THR
        print('  %-18s flags=%3d  agree=%.1f%%  lost=%d gained=%d  in[.21,.3)=%d'
              % (tag, int(fn.sum()), (fo == fn).mean() * 100,
                 int((fo & ~fn).sum()), int((~fo & fn).sum()),
                 int(((r >= THR) & (r < 0.30)).sum())))

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

    vis_uni = np.array([bool(d['visual']) for d in diag])

    def assemble(fn):
        poses, acc = {}, 0
        for i, key in enumerate(hk):
            if fn[i] and fo[i]:
                poses[key] = uni[key]
                acc += int(vis_uni[i])
            elif fn[i] and not fo[i]:
                q = P.prepare_scan(_G['frames'][i], _G['cfg'])[:4000]
                c3 = cost_at(r3[key], q)
                cv = cost_at(vis_pose[key], q) if key in vis_pose else np.inf
                ok = cv < c3 - 0.05
                poses[key] = vis_pose[key] if ok else r3[key]
                acc += int(ok)
            else:
                poses[key] = r3[key]
        return poses, acc

    print('\n-- end to end --')
    for tag, fn in (('merged gate @0.21', r1 < THR),
                    ('merged gate @0.30', r1 < 0.30),
                    ('2000 pts gate @0.21', r2k < THR)):
        poses, acc = assemble(fn)
        A = np.stack([poses[k] for k in hk])
        e = su.pos_err(A, gt)
        rr = su.rot_deg(A[:, :3, :3], gt[:, :3, :3])
        print('  %-20s eligible=%3d accepted=%2d  MPE=%.4f MOE=%.4f ok=%d >10m=%d >5m=%d'
              % (tag, int(fn.sum()), acc, e.mean(), rr.mean(),
                 int(((e <= 1) & (rr <= 2)).sum()), int((e > 10).sum()), int((e > 5).sum())))
    print('  current unified      accepted=23  MPE=0.9761 MOE=2.1707 ok=727 >10m=3 >5m=10')


if __name__ == '__main__':
    main()
