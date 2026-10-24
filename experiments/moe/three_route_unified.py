"""One objective, two landscape statistics, one selection rule.

The pose problem is the minimisation of a single robust cost: the mean truncated
nearest-neighbour distance between the transformed scan and the local reference
cloud (truncation 1 m). Everything below reads that one number.

    T_leader 鈹€鈹€鈻?coarse cost surface (17 x 3, 1500 points)          stage A
                    鈹?                    鈹溾攢 descent depth  g = C(T_leader) - min C       -> compute gate
                    鈹溾攢 ambiguity ratio rho = (C_far - C_0) / C_0    -> modality query
                    鈹斺攢 fine grid (5 x 9 per candidate, gradient bound)
                            鈹?                            鈹斺攢 rotation block (translation frozen, held-out accept)
                                    鈹?    visual hypothesis 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈻?pick the cheaper one, minus a switching penalty

The coarse surface is built once per frame and both the compute gate and the fine
stage read it; the fine stage is never charged for it a second time.

Modes (so the ablation is a one-line switch):
    --mode route3    no visual hypothesis: gate -> search -> rotation
    --mode frozen    previous arbitration: rho < 0.21 and an external score margin > 0.10
    --mode unified   arbitration by the same cost: rho < 0.21 and C_vis < C_route3 - delta

The ambiguity ratio has two sources, also one flag:
    --gate merged    measured on the map and KD-tree the search already built (default)
    --gate frozen    the standalone gate, which rebuilds its own map every frame

    python three_route_unified.py --mode unified --gate merged --workers 8 --delta 0.05
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
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
sys.path.insert(0, os.path.join(HERE, 'stages/gate1'))
sys.path.insert(0, os.path.join(HERE, 'stages'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402
import ambiguity as A  # noqa: E402

ROOT = HERE
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
COST_TRUNC = 1.0        # the one cost function, in metres
ARB_POINTS = 4000       # query points used when scoring a hypothesis
AMB_THRESHOLD = A.THRESHOLD  # 0.21: below this the other modality is queried
FROZEN_MARGIN = 0.10    # previous arbitration: margin on the external geometry score
DEFAULT_DELTA = 0.05    # switching penalty, in metres of mean truncated residual
_G = {}
_OPT = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config(solver=_OPT.get('solver', 'adaptive'),
                   coarse_points=_OPT.get('coarse_points', 1000),
                   fine_points=_OPT.get('fine_points', 1500),
                   workers=_OPT.get('threads', 4))
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))
    if _OPT['mode'] != 'route3' and _OPT['gate'] == 'frozen':
        import gate as G1
        _G['gate'] = G1
        _G['lib'] = G1.ReferenceLibrary()

    if _OPT['mode'] == 'route3':
        return
    vis = []
    if _OPT['mode'] == 'frozen':
        for i in range(len(frames)):
            rec = json.load(open(VERIFY % i))['poses']
            base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
            best = None
            for name, row in rec.items():
                if not name.startswith('candidate_'):
                    continue
                sc = (row.get('fair') or {}).get('geometry_score')
                if sc is None:
                    continue
                if best is None or sc > best[0]:
                    best = (float(sc), np.asarray(row['pose'], float))
            vis.append((best, base))
        _G['vis'] = vis
    else:
        _G['vis'] = [loading_hook(i) for i in range(len(frames))]


def loading_hook(i):
    """Pose-only view of the frozen visual table (the unified rule needs no score)."""
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
    return (best, None)


def cost(tree, pts, trunc=COST_TRUNC):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


def cost_at(pose, q, maps, cfg):
    """Score a hypothesis on the map cropped around its own position."""
    centre = pose[:3, 3]
    mp = maps.local(centre, maps.nearby(centre)[:cfg.reference_count])
    return cost(cKDTree(mp), q @ pose[:3, :3].T + centre)


def rot_only(tree, geom, q, T, eps=1e-6):
    """Rotation block: translation frozen, step accepted on a held-out half."""
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        W = val @ M[:3, :3].T + M[:3, 3]
        d = tree.query(W, k=1)[0]
        return float(np.minimum(d, 1.2).mean()), int((d < 0.6).sum())

    t = np.array(T, float)
    centre = t[:3, 3].copy()
    value, hit = obj(t)
    for _ in range(6):
        W = train @ t[:3, :3].T + centre
        dist, idx = tree.query(W, k=1)
        ok = dist < 1.2
        if ok.sum() < 30:
            break
        A, B = W[ok] - centre, geom[idx[ok]] - centre
        a, b = A.mean(0), B.mean(0)
        U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
        if s[1] < 1e-8:
            break
        corr = np.eye(3)
        corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ corr @ U.T
        vec = Rotation.from_matrix(R).as_rotvec()
        scale = min(1.0, np.deg2rad(0.5) / max(np.linalg.norm(vec), 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre
        nv, nh = obj(cand)
        if not (nv < value - eps and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return t


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    rec = _G['frames'][i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(rec, cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0])
    lat = lat / np.linalg.norm(lat)

    # --- stage A: the coarse surface, built once
    cs = F.coarse_stage(mp, scan, T0, u, workers=1,
                        n_coarse=cfg.coarse_points, topk=cfg.topk)
    qc = scan[:min(cfg.coarse_points, len(scan))]
    c_zero = cost(cs['tree'], cs['base_c'] + T0[:3, 3])
    gain = c_zero - float(cs['surface'].min())

    # --- gates: descend only if the surface actually drops
    T3 = np.array(T0, float)
    ran = False
    if gain > _OPT['gate2']:
        pose, _ = P.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
        T3 = np.asarray(pose, float)
        ran = True

    diag = dict(key=rec['key'], gain=float(gain), ran_fine=bool(ran), rotated=False)

    if _OPT['mode'] == 'route3':
        diag.update(visual=False, chosen='route3')
        return i, T3, diag

    # --- ambiguity: is there a competing along-track position that fits as well?
    #     merged: same map and tree as the search, strided 4000-point sweep
    #     frozen: the standalone gate, with its own map built per frame
    if _OPT['gate'] == 'merged':
        _, amb = A.decide(cs['tree'], scan, T0, u)
    else:
        amb = _G['gate'].ambiguity_ratio(_G['lib'], rec['scan_path'], T0)
    ratio = amb['ratio']
    diag['ratio'] = float(ratio)
    diag['gate'] = _OPT['gate']

    best, base = _G['vis'][i]
    accepted = False
    c3 = cv = float('nan')
    if best is not None and np.isfinite(ratio) and ratio < AMB_THRESHOLD:
        if _OPT['mode'] == 'frozen':
            accepted = bool(base is not None and best[0] > float(base) + FROZEN_MARGIN)
        else:
            # the same cost, on each hypothesis' own local map
            q = scan[:min(ARB_POINTS, len(scan))]
            c3 = cost_at(T3, q, maps, cfg)
            cv = cost_at(best[1], q, maps, cfg)
            accepted = bool(cv < c3 - _OPT['delta'])
    diag.update(c_route3=float(c3), c_visual=float(cv),
                dcost=float(cv - c3) if np.isfinite(c3) and np.isfinite(cv) else float('nan'),
                visual=bool(accepted), chosen='visual' if accepted else 'route3')
    return i, (best[1] if accepted else T3), diag


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('route3', 'frozen', 'unified'), default='unified')
    ap.add_argument('--gate', choices=('merged', 'frozen'), default='merged',
                    help='merged: ratio from the search map; frozen: standalone gate')
    ap.add_argument('--solver', choices=('adaptive', 'prune', 'full'), default='adaptive')
    ap.add_argument('--coarse-points', type=int, default=1000)
    ap.add_argument('--fine-points', type=int, default=1500)
    ap.add_argument('--threads', type=int, default=4,
                    help='KD-tree query threads inside one frame')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gate2', type=float, default=-1.0,
                    help='descent threshold; -1 disables the gate (default, see README)')
    ap.add_argument('--delta', type=float, default=DEFAULT_DELTA)
    ap.add_argument('--rot', action='store_true', default=False, help='deprecated: the rotation block was removed (it bought 0.003 deg of MOE)')
    ap.add_argument('--limit', type=int, default=0, help='0 = all frames')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    _OPT.update(mode=a.mode, gate=a.gate, gate2=a.gate2, delta=a.delta, rot=a.rot,
                solver=a.solver, coarse_points=a.coarse_points,
                fine_points=a.fine_points, threads=a.threads)
    if a.mode == 'route3':
        default_out = 'results/three_route_route3.npz'
    elif a.gate == 'merged':
        default_out = 'results/three_route_merged.npz'
    else:
        default_out = 'results/three_route_%s.npz' % a.mode
    out = a.out or default_out

    init()
    n = len(_G['frames'])
    idx = list(range(n if a.limit <= 0 else min(a.limit, n)))
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as p:
            res = p.map(_w, idx, chunksize=4)
    else:
        res = [_w(i) for i in idx]

    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']][:len(idx)]
    poses = np.stack([r[1] for r in res])
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    np.savez(os.path.join(ROOT, out), keys=np.array(keys), poses=poses)
    diag_path = os.path.splitext(out)[0] + '_diag.jsonl'
    diag_path = diag_path if os.path.isabs(diag_path) else os.path.join(ROOT, diag_path)
    with open(diag_path, 'w') as fh:
        for _, _, dg in res:
            fh.write(json.dumps(dg) + '\n')
    vis = sum(1 for r in res if r[2].get('visual'))
    fine = sum(1 for r in res if r[2]['ran_fine'])
    print('mode=%s gate=%s frames=%d visual=%d fine=%d gate2=%.3f delta=%.3f -> %s'
          % (a.mode, a.gate, len(idx), vis, fine, a.gate2, a.delta, out))


if __name__ == '__main__':
    main()

