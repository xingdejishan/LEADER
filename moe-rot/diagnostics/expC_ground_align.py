"""EXP C (pilot): can the road plane remove the tilt half of the rotation error?

The front-end rotation error splits into yaw ~0.7 deg and tilt ~0.9 deg.  A
ground plane gives two of the three rotation degrees of freedom absolutely:
fit the road plane in the scan (body frame), fit the road plane in the reference
map at the estimated position (world frame), and rotate the pose so the two
normals agree.  Yaw is untouched.

Reported: the front-end tilt error, the error left after the correction, and the
resulting MOE, for a stratified pilot sample.
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
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expC_ground_align.npz'
N_PER_BAND = int(os.environ.get('N', '100'))
RADIUS = 25.0


def fit_plane(pts, zlo=2, zhi=40, band=0.4):
    if len(pts) < 150:
        return None
    z = pts[:, 2]
    lo, hi = np.percentile(z, [zlo, zhi])
    cand = pts[(z >= lo) & (z <= hi)]
    if len(cand) < 100:
        return None
    hist, edges = np.histogram(cand[:, 2], bins=40)
    k = int(np.argmax(hist))
    zc = 0.5 * (edges[k] + edges[k + 1])
    sel = cand[np.abs(cand[:, 2] - zc) < band]
    if len(sel) < 80:
        return None
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        c = sel.mean(0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        r = (sel - c) @ n
        keep = np.abs(r) < 2.5 * max(np.std(r), 0.05) + 0.05
        if keep.all() or keep.sum() < 80:
            break
        sel = sel[keep]
    c = sel.mean(0)
    r = (sel - c) @ n
    return n, float(np.sqrt((r ** 2).mean())), len(sel)


def rot_between(a, b):
    """minimal rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    s = np.linalg.norm(v)
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def log_vec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(ang)
    if s < 1e-8:
        return 0.5 * v
    return v * ang / (2.0 * s)


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    t_start = time.time()
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

    rng = np.random.default_rng(20261022)
    bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 5.0), (5.0, 1e9)]
    pick = []
    for lo, hi in bands:
        idx = np.where((e0 >= lo) & (e0 < hi))[0]
        if len(idx) == 0:
            continue
        pick.append(rng.choice(idx, min(N_PER_BAND, len(idx)), replace=False))
    pick = np.concatenate(pick)

    res = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        # scan plane in the body frame (the sensor sits above the road: z negative)
        gs = fit_plane(scan, 2, 40, 0.5)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        d2 = np.linalg.norm(mp[:, :2] - centre[None, :2], axis=1)
        gm = fit_plane(mp[d2 < RADIUS], 2, 40, 0.5)
        if gs is None or gm is None:
            res.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        n_b, rms_s, ns = gs
        n_w, rms_m, nm = gm
        R, p = t0[i][:3, :3], t0[i][:3, 3]
        Rg = gt[i][:3, :3]
        a_before = ang(R.T @ Rg)

        # (1) align the scan plane normal (posed) with the map plane normal
        u = R @ n_b
        Rc = rot_between(u, n_w) @ R
        a_align = ang(Rc.T @ Rg)

        # (2) cheap variant: align with world up only (no map crop needed)
        Rc2 = rot_between(u, np.array([0.0, 0.0, 1.0])) @ R
        a_up = ang(Rc2.T @ Rg)

        # (3) oracle: best tilt-only correction toward truth (2 dof upper bound)
        Rc3 = rot_between(u, Rg @ n_b) @ R
        a_oracle = ang(Rc3.T @ Rg)

        tilt_only = ang(rot_between(R @ np.array([0, 0, 1.0]), Rg @ np.array([0, 0, 1.0])))
        w = log_vec(R.T @ Rg)
        dmap = np.degrees(np.arccos(np.clip(u @ n_w, -1, 1)))
        res.append((i, a_before, a_align, a_up, a_oracle, dmap, tilt_only, np.degrees(abs(w[2]))))
        if (j + 1) % 50 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(pick), time.time() - t_start), flush=True)

    a = np.array(res, float)
    ok = ~np.isnan(a[:, 2])
    lab = np.array([('bulk' if e0[i] < 0.2 else 'mid' if e0[i] < 0.5 else 'hard' if e0[i] < 5 else 'tail') for i in a[:, 0].astype(int)])
    print('\nframes with both planes: %d/%d' % (int(ok.sum()), len(a)))
    print('\n%-6s %5s %9s %9s %9s %9s %9s %9s' % ('band', 'n', 'MOE now', 'ground→map', 'ground→up', 'oracle tilt',
                                                  'tilt err', 'yaw err'))
    for b in ('bulk', 'mid', 'hard', 'tail'):
        m = ok & (lab == b)
        if m.sum() == 0:
            continue
        print('%-6s %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f'
              % (b, int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
                 a[m, 4].mean(), a[m, 6].mean(), a[m, 7].mean()))
    m = ok
    print('%-6s %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f'
          % ('all', int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
             a[m, 4].mean(), a[m, 6].mean(), a[m, 7].mean()))
    print('\nmean angle between posed scan normal and map normal: %.3f deg'
          % a[ok, 5].mean())
    print('frames improved by ground→map: %d/%d ; by ground→up: %d/%d'
          % (int((a[ok, 2] < a[ok, 1]).sum()), int(ok.sum()),
             int((a[ok, 3] < a[ok, 1]).sum()), int(ok.sum())))
    np.savez(OUT, res=a)
    print('wrote %s  total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
