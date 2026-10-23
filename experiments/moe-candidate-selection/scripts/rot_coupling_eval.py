#!/usr/bin/env python
"""Two candidate fixes for the rotation-coupling problem of the geometric branch.

Baseline (deployed): coarse+fine along-track/lateral search with the front-end
rotation frozen, then one rotation-only block (three_route_unified.rot_only).

A  ALTERNATING : search -> rot_only -> search (from the rotated pose) -> rot_only
B  YAW GRID    : coarse stage repeated for yaw in {-3..3} deg about world z, keep
                 the best cell, fine stage with that yaw fixed, then rot_only

Scans, maps, config and the rotation block all come from the project itself
(fast_lib / adaptive / pipeline / three_route_unified.rot_only from the backup made
when the rotation block was removed from the live tree).

  python rot_coupling_eval.py --check 12
  python rot_coupling_eval.py --n-control 2000 --workers 16 --out results/....
"""
import argparse
import json
import os
import sys
import time
import importlib.util
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
FULL_NPZ = os.path.join(PROJECT, 'results/full-online.npz')
BACKUP = os.path.join(PROJECT, '.backups/remove-rotation-20261022-182520')

sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages/gate1'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
sys.path.insert(0, PROJECT)

import fast_lib as F          # noqa: E402
import pipeline as P          # noqa: E402
import adaptive               # noqa: E402


def _load_rot_only():
    spec = importlib.util.spec_from_file_location(
        'tru_backup', os.path.join(BACKUP, 'three_route_unified.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rot_only


rot_only = _load_rot_only()
YAW_DEG = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
DATASET_ROOTS = (os.path.join(DATA, 'NCLT'), '/data2/dataset/NCLT')
NAMES = ('baseline', 'A', 'B')

_STATE = {}


# --------------------------------------------------------------------------- #
# subset                                                                        #
# --------------------------------------------------------------------------- #
def truth_table():
    out = {}
    for name in sorted(os.listdir(TRUTH_DIR)):
        if not name.endswith('.json'):
            continue
        date = name[:-5]
        for ts, rec in json.load(open(os.path.join(TRUTH_DIR, name))).items():
            out[date + '/' + ts] = np.asarray(rec['T_WB'], float)
    return out


def scan_path(key):
    date, ts = key.split('/')
    for root in DATASET_ROOTS:
        path = os.path.join(root, date, 'velodyne_sync', ts + '.bin')
        if os.path.exists(path):
            return path
    raise IOError('scan not found for ' + key)


def build_subset(n_control, seed):
    z = np.load(FULL_NPZ, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    t0 = np.asarray(z['t0'], float)
    online = np.asarray(z['poses'], float)
    gt = truth_table()
    G = np.stack([gt[k] for k in keys])
    ep0 = np.linalg.norm(t0[:, :3, 3] - G[:, :3, 3], axis=1)

    hard = np.where(ep0 > 0.5)[0]
    rest = np.where(ep0 <= 0.5)[0]
    rng = np.random.RandomState(seed)
    ctrl = np.sort(rng.choice(rest, size=min(n_control, len(rest)), replace=False))
    sel = np.concatenate([hard, ctrl])
    sel = sel[np.argsort([keys[i] for i in sel])]          # map-cache locality
    return dict(keys=[keys[i] for i in sel], t0=t0[sel], gt=G[sel], ep0=ep0[sel],
                online=online[sel], hard=np.isin(sel, hard))


# --------------------------------------------------------------------------- #
# per-frame variants                                                            #
# --------------------------------------------------------------------------- #
def _lateral(u):
    lat = np.cross(u, [0.0, 0.0, 1.0])
    return lat / np.linalg.norm(lat)


def _search(mp, scan, T0, cfg, coarse=None):
    """The project's search block: coarse grid + adaptive fine, rotation frozen."""
    u = P.along_track_direction(T0)
    if coarse is None:
        coarse = F.coarse_stage(mp, scan, T0, u, workers=cfg.workers,
                                n_coarse=cfg.coarse_points, topk=cfg.topk)
    s, d, cost, evals, expanded = adaptive.solve_adaptive(
        mp, scan, T0, u, n_coarse=cfg.coarse_points, n_fine=cfg.fine_points,
        workers=cfg.workers, topk=cfg.topk, coarse=coarse)
    pose = np.array(T0, float)
    pose[:3, 3] = T0[:3, 3] + s * u + d * _lateral(u)
    return pose, coarse, u, s, d, cost


def variant_baseline(mp, scan, T0, q, cfg):
    pose, cs, u, s, d, cost = _search(mp, scan, T0, cfg)
    out = rot_only(cs['tree'], mp, q, pose)
    return out, dict(s=s, d=d, cost=cost)


def variant_alt(mp, scan, T0, q, cfg):
    pose, cs, u, s, d, cost = _search(mp, scan, T0, cfg)
    t1 = rot_only(cs['tree'], mp, q, pose)
    pose2, cs2, u2, s2, d2, cost2 = _search(mp, scan, t1, cfg)
    out = rot_only(cs2['tree'], mp, q, pose2)
    return out, dict(s=s, d=d, s2=s2, d2=d2, cost=cost, cost2=cost2)


def variant_yaw(mp, scan, T0, q, cfg):
    best = None
    surfaces = []
    for deg in YAW_DEG:
        Y = np.array(T0, float)
        Y[:3, :3] = Rotation.from_rotvec(
            np.deg2rad(deg) * np.array([0.0, 0.0, 1.0])).as_matrix() @ T0[:3, :3]
        uy = P.along_track_direction(Y)
        cs = F.coarse_stage(mp, scan, Y, uy, workers=cfg.workers,
                            n_coarse=cfg.coarse_points, topk=cfg.topk)
        c = float(cs['surface'].min())
        surfaces.append(c)
        if best is None or c < best[0]:
            best = (c, Y, uy, cs)
    c, Y, uy, cs = best
    s, d, cost, evals, expanded = adaptive.solve_adaptive(
        mp, scan, Y, uy, n_coarse=cfg.coarse_points, n_fine=cfg.fine_points,
        workers=cfg.workers, topk=cfg.topk, coarse=cs)
    pose = np.array(Y, float)
    pose[:3, 3] = Y[:3, 3] + s * uy + d * _lateral(uy)
    out = rot_only(cs['tree'], mp, q, pose)
    return out, dict(yaw=float(YAW_DEG[int(np.argmin(surfaces))]), s=s, d=d,
                     cost=cost, surfaces=surfaces)


# --------------------------------------------------------------------------- #
# workers                                                                       #
# --------------------------------------------------------------------------- #
def _init(refs, positions, cfg):
    _STATE['maps'] = F.MapSource(refs, positions, radius=cfg.crop_radius,
                                 max_refs=150, max_maps=40)
    _STATE['cfg'] = cfg


def one(item):
    i, key, T0 = item
    cfg = _STATE['cfg']
    maps = _STATE['maps']
    rec = {'key': key, 'scan_path': scan_path(key)}
    t_prep = time.perf_counter()
    scan = P.prepare_scan(rec, cfg)
    centre = T0[:3, 3]
    triple = maps.nearby(centre)[:cfg.reference_count]
    mp = maps.local(centre, triple)
    prep = time.perf_counter() - t_prep
    if len(mp) < 500 or len(triple) < 2:
        return dict(i=i, skipped='thin_map', n_map=len(mp), n_scan=len(scan),
                    prep=prep, poses=None, secs={}, diag={})
    q = scan[np.linalg.norm(scan, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]

    res = dict(i=i, skipped=None, n_map=len(mp), n_scan=len(scan), prep=prep,
               poses={}, secs={}, diag={})
    for name, fn in (('baseline', variant_baseline), ('A', variant_alt),
                     ('B', variant_yaw)):
        t = time.perf_counter()
        pose, diag = fn(mp, scan, T0, q, cfg)
        res['secs'][name] = time.perf_counter() - t
        res['poses'][name] = pose
        res['diag'][name] = diag
    return res


# --------------------------------------------------------------------------- #
# metrics                                                                       #
# --------------------------------------------------------------------------- #
def ang_err(A, G):
    R = A[:3, :3] @ G[:3, :3].T
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def score(rows, subset, keep_fn, label):
    """Aggregate every variant over the frames of one band."""
    out = []
    ok = [r for r in rows if r['skipped'] is None]
    flags = [keep_fn(subset['ep0'][r['i']]) for r in ok]
    gt = {r['i']: subset['gt'][r['i']] for r in ok}
    for name in NAMES:
        ep = np.array([np.linalg.norm(r['poses'][name][:3, 3] - gt[r['i']][:3, 3])
                       for r, f in zip(ok, flags) if f])
        er = np.array([ang_err(r['poses'][name], gt[r['i']])
                       for r, f in zip(ok, flags) if f])
        sec = np.array([r['secs'][name] for r, f in zip(ok, flags) if f])
        out.append(dict(band=label, variant=name, n=len(ep), mpe=ep.mean(),
                        moe=er.mean(), ok=float(((ep <= 1.0) & (er <= 2.0)).mean()),
                        gt1=int((ep > 1.0).sum()), gt5=int((ep > 5.0).sum()),
                        sec=float(sec.mean())))
    return out


def fmt(res):
    lines = ['| band    | variant  |   n |   MPE |   MOE | ok(1m,2d) | >1m | >5m | s/frame |',
             '|---------|----------|-----|-------|-------|-----------|-----|-----|---------|']
    for r in res:
        lines.append('| %-7s | %-8s | %4d | %5.2f | %5.2f |   %5.1f%%  | %3d | %3d |  %.3f  |'
                     % (r['band'], r['variant'], r['n'], r['mpe'], r['moe'],
                        100 * r['ok'], r['gt1'], r['gt5'], r['sec']))
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-control', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261022)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--check', type=int, default=0,
                    help='reproduce the baseline for N frames and diff full-online')
    ap.add_argument('--out', default='rot-coupling-subset.npz')
    a = ap.parse_args()

    t_all = time.perf_counter()
    sub = build_subset(a.n_control, a.seed)
    n = len(sub['keys'])
    if a.limit:
        n = min(n, a.limit)
    print('subset frames=%d  hard(>0.5m)=%d  control=%d'
          % (n, int(sub['hard'][:n].sum()), int((~sub['hard'][:n]).sum())))
    for lo, hi in ((0.5, 1), (1, 2), (2, 5), (5, 8), (8, 1e9)):
        m = (sub['ep0'][:n] > lo) & (sub['ep0'][:n] <= hi)
        print('  band %.1f-%.1f: %d frames' % (lo, hi, int(m.sum())))

    cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500,
                   workers=a.threads)
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    for row in refs:
        row['scan_path'] = os.path.realpath(row['scan_path'])
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
    items = [(i, sub['keys'][i], sub['t0'][i]) for i in range(n)]

    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(refs, positions, cfg)) as pool:
            rows = pool.map(one, items, chunksize=2)
    else:
        _init(refs, positions, cfg)
        rows = [one(it) for it in items]
    wall = time.perf_counter() - t_all
    print('done in %.1f s (%.1f frames/s)' % (wall, n / wall))

    sk = sum(1 for r in rows if r['skipped'])
    print('skipped (thin map): %d' % sk)

    if a.check:
        d = [np.abs(r['poses']['baseline'] - sub['online'][r['i']]).max()
             for r in rows if r['skipped'] is None]
        print('check vs full-online: n=%d  max|dpose| median=%.2e  max=%.2e'
              % (len(d), float(np.median(d)), float(np.max(d))))

    bands = [('>0.5m', lambda e: e > 0.5),
             ('control', lambda e: e <= 0.5),
             ('5-8m', lambda e: 5.0 < e <= 8.0),
             ('all', lambda e: np.ones_like(e, bool))]
    res = []
    for label, fn in bands:
        res += score(rows, sub, fn, label)
    print('\n' + fmt(res))

    # per-frame poses + diagnostics for the record
    outd = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(outd, exist_ok=True)
    np.savez(a.out,
             keys=np.array(sub['keys'][:n]),
             t0=sub['t0'][:n], gt=sub['gt'][:n], ep0=sub['ep0'][:n],
             hard=sub['hard'][:n], online=sub['online'][:n],
             **{('pose_' + nm): np.stack([r['poses'][nm] for r in rows])
                for nm in NAMES},
             **{('sec_' + nm): np.array([r['secs'][nm] for r in rows])
                for nm in NAMES})
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as fh:
        json.dump(dict(rows=[dict(i=r['i'], skipped=r['skipped'], n_map=r['n_map'],
                                  n_scan=r['n_scan'], prep=r['prep'],
                                  secs=r['secs'], diag=r['diag']) for r in rows],
                       table=res, wall=wall), fh, default=float)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
