"""EXP G: road-plane attitude, done correctly.

The earlier attempt isolated the "ground" by a z-slab in the *body* frame, which
does not contain the ground, so the fit was meaningless.  Here the ground band is
taken in the world frame after applying the truth pose (so the slab genuinely
contains the road), and the fitted normal is mapped back into the body frame.

Variants of the 2-dof correction applied to the front-end pose:
  up    : make the road normal vertical
  map   : make it parallel to the reference map's own road normal (handles slopes)
  oracle: make it parallel to the truth's road normal (upper bound, 2 dof only)
"""
import json
import os
import sys
import time

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expG_road2.npz'
N_PER_BAND = int(os.environ.get('N', '100'))
BAND = 0.35


def road_normal(pts, rng=None):
    """Normal of the lowest flat surface: select by height only, then fit."""
    if len(pts) < 400:
        return None
    z = pts[:, 2]
    z0 = np.percentile(z, 3)
    sel = pts[np.abs(z - z0) < BAND]
    if len(sel) < 300:
        return None
    n = np.array([0.0, 0, 1.0])
    for _ in range(5):
        c = sel.mean(0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        r = (sel - c) @ n
        keep = np.abs(r) < 2.5 * max(np.std(r), 0.03) + 0.03
        if keep.mean() > 0.97 or keep.sum() < 200:
            break
        sel = sel[keep]
    c = sel.mean(0)
    r = (sel - c) @ n
    return n, float(np.sqrt((r ** 2).mean())), int(len(sel))


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def ra(A, B):
    R = A.T @ B
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    t_start = time.time()
    rng = np.random.default_rng(20261022)
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()
    blob = np.load(os.path.join(PROJECT, 'results/full-online-final.npz'), allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    t0 = np.asarray(blob['t0'], np.float64)
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        date = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)

    bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 5.0), (5.0, 1e9)]
    pick = []
    for lo, hi in bands:
        idx = np.where((e0 >= lo) & (e0 < hi))[0]
        if len(idx):
            pick.append(rng.choice(idx, min(N_PER_BAND, len(idx)), replace=False))
    pick = np.concatenate(pick)
    up = np.array([0.0, 0, 1.0])

    rows = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        R, p = t0[i][:3, :3], t0[i][:3, 3]
        Rg, pg = gt[i][:3, :3], gt[i][:3, 3]
        sw = scan @ Rg.T + pg
        sw = sw[np.linalg.norm(sw[:, :2] - pg[None, :2], axis=1) < 20.0]
        gs = road_normal(sw)
        mp = maps.local(p, maps.nearby(p)[:3])
        mpn = mp[np.linalg.norm(mp[:, :2] - p[None, :2], axis=1) < 20.0]
        gm = road_normal(mpn)
        if gs is None or gm is None:
            rows.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        n_b = Rg.T @ gs[0]
        n_m = gm[0]
        a_now = ra(R, Rg)
        a_up = ra(rot_between(R @ n_b, up) @ R, Rg)
        a_map = ra(rot_between(R @ n_b, n_m) @ R, Rg)
        a_or = ra(rot_between(R @ n_b, Rg @ n_b) @ R, Rg)
        rows.append((i, a_now, a_up, a_map, a_or,
                     np.degrees(np.arccos(np.clip(Rg @ n_b @ up, -1, 1))),
                     gs[1], gm[1], np.degrees(np.arccos(np.clip(n_m @ up, -1, 1)))))
        if (j + 1) % 50 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(pick), time.time() - t_start), flush=True)

    a = np.array(rows, float)
    ok = ~np.isnan(a[:, 1])
    lab = np.array([('bulk' if e0[i] < 0.2 else 'mid' if e0[i] < 0.5 else 'hard' if e0[i] < 5 else 'tail')
                    for i in a[:, 0].astype(int)])
    print('\nroad plane fits ok: %d/%d' % (int(ok.sum()), len(a)))
    print('%-6s %5s %9s %9s %9s %9s %9s %9s' %
          ('band', 'n', 'MOE now', '→up', '→map', '→oracle', 'road tilt', 'map road tilt'))
    for b in ('bulk', 'mid', 'hard', 'tail'):
        m = ok & (lab == b)
        if m.sum() == 0:
            continue
        print('%-6s %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f'
              % (b, int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
                 a[m, 4].mean(), a[m, 5].mean(), a[m, 8].mean()))
    m = ok
    print('%-6s %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f'
          % ('all', int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
             a[m, 4].mean(), a[m, 5].mean(), a[m, 8].mean()))
    print('\nscan road-plane rms %.3f m ; map road-plane rms %.3f m'
          % (a[ok, 6].mean(), a[ok, 7].mean()))
    for c, name in ((2, 'up'), (3, 'map'), (4, 'oracle')):
        print('   %-6s improved %d/%d  median %+.3f deg' %
              (name, int((a[ok, c] < a[ok, 1]).sum()), int(ok.sum()),
               np.median(a[ok, c] - a[ok, 1])))
    np.savez(OUT, res=a)
    print('wrote %s  total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
