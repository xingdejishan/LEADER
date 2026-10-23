"""Read-only verification of the proposed gate merge (ambiguity ratio read off the
coarse cost surface).

Checks, in order:
  1. cost breakdown: frozen gate (own map per frame) vs the same statistic on the
     search's cached local map, on a sample of frames;
  2. does the ratio survive the move: flag counts and per-frame agreement against
     the frozen gate, for step 1.0 vs 0.5 m, 4000 vs 2000 points, prefix vs strided;
  3. the one frame claimed to be lost, and the count of frames a threshold of 0.30
     would additionally admit;
  4. end-to-end effect: assemble the merged design's poses from the existing tables
     (+ cost arbitration only where eligibility changes) and score.

Writes nothing.
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

SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
TRUNC = 1.0
FAR = 5.0
THR = 0.21
_G = {}


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


def surface_ratio(tree, Q, t0, u, s_grid, d_off):
    """min over the grid, and the same over |s| > FAR -> (ratio, c0, cfar)."""
    costs = np.empty((len(s_grid), len(d_off)))
    for a, s in enumerate(s_grid):
        for b, dd in enumerate(d_off):
            costs[a, b] = cost(tree, Q + (t0 + s * u + dd))
    c0 = float(costs.min())
    far = costs[np.abs(s_grid) > FAR]
    cfar = float(far.min())
    return (cfar - c0) / max(c0, 1e-6), c0, cfar


def points(scan, n, strided):
    if strided:
        return scan[np.linspace(0, len(scan) - 1, min(n, len(scan))).astype(int)]
    return scan[:min(n, len(scan))]


S17 = np.round(np.arange(-8.0, 8.0001, 1.0), 3)
S33 = np.round(np.arange(-8.0, 8.0001, 0.5), 3)


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0])
    lat /= np.linalg.norm(lat)
    d_off = np.outer([-0.5, 0.0, 0.5], lat)
    t0 = T0[:3, 3]

    Q = points(scan, 4000, False) @ T0[:3, :3].T
    rA = surface_ratio(tree, Q, t0, u, S17, d_off)
    rB = surface_ratio(tree, Q, t0, u, S33, d_off)
    Qc = points(scan, 2000, False) @ T0[:3, :3].T
    rC = surface_ratio(tree, Qc, t0, u, S17, d_off)
    Qd = points(scan, 4000, True) @ T0[:3, :3].T
    rD = surface_ratio(tree, Qd, t0, u, S17, d_off)
    return i, (rA[0], rB[0], rC[0], rD[0])


def timing(nframes=24):
    cfg = _G['cfg']
    t_old, t_new = [], []
    for i in range(nframes):
        rec = _G['frames'][i]
        T0 = _G['poses'][i]
        s = time.perf_counter()
        G1.ambiguity_ratio(_G['lib'], rec['scan_path'], T0)
        t_old.append(time.perf_counter() - s)

        mp = _G['maps'].local(T0[:3, 3], _G['maps'].nearby(T0[:3, 3])[:cfg.reference_count])
        scan = P.prepare_scan(rec, cfg)
        u = P.along_track_direction(T0)
        lat = np.cross(u, [0.0, 0.0, 1.0]); lat /= np.linalg.norm(lat)
        d_off = np.outer([-0.5, 0.0, 0.5], lat)
        s = time.perf_counter()
        tree = cKDTree(mp)
        Q = points(scan, 4000, False) @ T0[:3, :3].T
        surface_ratio(tree, Q, T0[:3, 3], u, S17, d_off)
        t_new.append(time.perf_counter() - s)
    f = lambda v: (np.median(v) * 1000, np.percentile(v, 90) * 1000)
    print('\n-- timing, %d frames, median / p90 (ms) --' % nframes)
    print('  frozen gate (own map per frame) : %.0f / %.0f' % f(t_old))
    print('  ratio on the shared search map   : %.0f / %.0f' % f(t_new))


def main():
    init()
    timing()
    n = len(_G['frames'])
    with Pool(8, initializer=init) as p:
        res = p.map(one, range(n), chunksize=8)
    rA = np.empty(n); rB = np.empty(n); rC = np.empty(n); rD = np.empty(n)
    for i, vals in res:
        rA[i], rB[i], rC[i], rD[i] = vals

    diag = [json.loads(l) for l in open(
        os.path.join(ROOT, 'results/three_route_unified_diag.jsonl'))]
    old = np.array([d['ratio'] for d in diag])

    def cmp(tag, r):
        flat_o = old < THR
        flat_n = r < THR
        agree = (flat_o == flat_n).mean() * 100
        print('  %-26s flags=%3d  agree=%.1f%%  (old flags=%d)' %
              (tag, int(flat_n.sum()), agree, int(flat_o.sum())))
        return flat_o, flat_n

    print('\n-- does the ratio survive the move? --')
    o, a = cmp('17 x 3, 1 m, 4000 prefix', rA)
    cmp('33 x 3, 0.5 m, 4000 prefix', rB)
    cmp('17 x 3, 1 m, 2000 prefix', rC)
    cmp('17 x 3, 1 m, 4000 strided', rD)
    print('  median |new - old| = %.3f   (A)' % np.median(np.abs(rA - old)))

    lost = np.where(o & ~a)[0]
    gained = np.where(~o & a)[0]
    print('\n-- frames the move changes --')
    print('  lost (old flagged, new not): %d -> %s' % (len(lost), lost.tolist()))
    for i in lost:
        print('    frame %d  old=%.3f  new=%.3f  key=%s' % (i, old[i], rA[i], diag[i]['key']))
    print('  gained (new flagged, old not): %d -> %s' % (len(gained), gained.tolist()))
    print('  frames a 0.30 threshold would additionally admit: %d'
          % int(((rA >= THR) & (rA < 0.30)).sum()))

    # ---- end-to-end assembly of the merged design
    spec = importlib.util.spec_from_file_location('su', SCORER)
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    hk, gt = w['hard_keys'], w['hard_gt']
    z3 = np.load(os.path.join(ROOT, 'results/final.npz'), allow_pickle=True)
    r3 = {str(k): np.asarray(v, float) for k, v in zip(z3['keys'], z3['poses'])}
    zu = np.load(os.path.join(ROOT, 'results/three_route_unified.npz'), allow_pickle=True)
    uni = {str(k): np.asarray(v, float) for k, v in zip(zu['keys'], zu['poses'])}

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

    poses = {}
    for i, k in enumerate(hk):
        rec = _G['frames'][i]
        if o[i] and a[i]:
            poses[k] = uni[k]
        elif o[i] and not a[i]:
            poses[k] = r3[k]
        elif a[i] and not o[i]:
            q = P.prepare_scan(rec, _G['cfg'])[:4000]
            c3 = cost_at(r3[k], q)
            cv = cost_at(vis_pose[k], q) if k in vis_pose else np.inf
            poses[k] = vis_pose[k] if cv < c3 - 0.05 else r3[k]
        else:
            poses[k] = r3[k]

    A = np.stack([poses[k] for k in hk])
    e = su.pos_err(A, gt)
    r = su.rot_deg(A[:, :3, :3], gt[:, :3, :3])
    print('\n-- end-to-end, merged gate (delta 0.05 unchanged) --')
    print('  MPE=%.4f  MOE=%.4f  ok(1m,2deg)=%d  >10m=%d  >5m=%d'
          % (e.mean(), r.mean(), int(((e <= 1) & (r <= 2)).sum()),
             int((e > 10).sum()), int((e > 5).sum())))
    print('  (current: MPE=0.9761 MOE=2.1707 ok=727 >10m=3 >5m=10)')


if __name__ == '__main__':
    main()
