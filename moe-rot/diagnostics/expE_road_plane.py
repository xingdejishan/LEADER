"""EXP E: road-plane attitude, fitted properly (3-D RANSAC, no z-slab shortcuts).

For each frame:
  * n_m  : plane of the reference map around the estimated position (world frame)
  * n_b  : plane normal of the query scan expressed in the body frame,
           obtained by fitting the scan in the world after applying the *truth*
           pose and mapping the normal back (assumption-free)
  * compare angle(R_gt n_b, n_m)  -- does the truth agree with the map's road?
    and    angle(R_t0 n_b, n_m)  -- how far is the front end from it?
  * apply the 2-dof correction that aligns R n_b with n_m and re-measure MOE.
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
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expE_road_plane.npz'
N_PER_BAND = int(os.environ.get('N', '80'))
N_TRIALS = 400
THRESH = 0.15


def ransac_plane(pts, rng, n_trials=N_TRIALS, thresh=THRESH, up_hint=None, up_tol=0.6):
    """Largest plane among those whose normal is within up_tol of up_hint."""
    n = len(pts)
    if n < 200:
        return None
    idx = rng.integers(0, n, size=(n_trials, 3))
    A, B, C = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    nrm = np.cross(B - A, C - A)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-6
    nrm = nrm[ok] / ln[ok, None]
    A = A[ok]
    if up_hint is not None:
        keep = np.abs(nrm @ up_hint) > up_tol
        nrm, A = nrm[keep], A[keep]
    if len(nrm) == 0:
        return None
    d = -(nrm * A).sum(1)
    res = np.abs(pts @ nrm.T + d[None, :])          # n x trials
    cnt = (res < thresh).sum(0)
    best = int(np.argmax(cnt))
    if cnt[best] < 100:
        return None
    nn = nrm[best]
    if up_hint is not None and nn @ up_hint < 0:
        nn = -nn
    sel = pts[res[:, best] < thresh]
    for _ in range(3):
        c = sel.mean(0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        v = vt[-1]
        if up_hint is not None and v @ up_hint < 0:
            v = -v
        r = (sel - c) @ v
        keep = np.abs(r) < 2.0 * max(np.std(r), thresh) + thresh
        if keep.sum() < 100:
            break
        sel = sel[keep]
    c = sel.mean(0)
    r = (sel - c) @ v
    nn = v
    return nn, float(np.sqrt((r ** 2).mean())), int(len(sel)), float(c @ nn)


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


def ang2(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


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

        mp = maps.local(p, maps.nearby(p)[:3])
        mpn = mp[np.linalg.norm(mp[:, :2] - p[None, :2], axis=1) < 18.0]
        gm = ransac_plane(mpn, rng, up_hint=up)
        # scan, expressed with the truth pose, then mapped back to the body frame
        sw = scan @ Rg.T + pg
        swn = sw[np.linalg.norm(sw[:, :2] - pg[None, :2], axis=1) < 18.0]
        gs = ransac_plane(swn, rng, up_hint=up)
        if gm is None or gs is None:
            rows.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        n_m = gm[0]
        n_b = Rg.T @ gs[0]
        a_gt = ang2(Rg @ n_b, n_m)
        a_t0 = ang2(R @ n_b, n_m)
        Rt = R.T @ Rg
        a_tot = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        Rc = rot_between(R @ n_b, n_m) @ R
        a_after = np.degrees(np.arccos(np.clip((np.trace(Rc.T @ Rg) - 1) / 2, -1, 1)))
        rows.append((i, a_gt, a_t0, a_tot, a_after, gm[1], gs[1],
                     np.degrees(np.arccos(np.clip(gs[0] @ up, -1, 1)))))
        if (j + 1) % 40 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(pick), time.time() - t_start), flush=True)

    a = np.array(rows, float)
    ok = ~np.isnan(a[:, 1])
    lab = np.array([('bulk' if e0[i] < 0.2 else 'mid' if e0[i] < 0.5 else 'hard' if e0[i] < 5 else 'tail')
                    for i in a[:, 0].astype(int)])
    print('\nframes with both planes: %d/%d' % (int(ok.sum()), len(a)))
    print('%-6s %5s %10s %10s %10s %10s %10s' %
          ('band', 'n', 'a(Rgt·nb,nm)', 'a(Rt0·nb,nm)', 'MOE now', 'MOE road', 'scan plane rms'))
    for b in ('bulk', 'mid', 'hard', 'tail'):
        m = ok & (lab == b)
        if m.sum() == 0:
            continue
        print('%-6s %5d %10.3f %10.3f %10.3f %10.3f %10.3f'
              % (b, int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
                 a[m, 4].mean(), a[m, 6].mean()))
    m = ok
    print('%-6s %5d %10.3f %10.3f %10.3f %10.3f %10.3f'
          % ('all', int(m.sum()), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(),
             a[m, 4].mean(), a[m, 6].mean()))
    print('\n  frames improved: %d/%d ; median change %+.3f deg'
          % (int((a[ok, 4] < a[ok, 3]).sum()), int(ok.sum()),
             np.median(a[ok, 4] - a[ok, 3])))
    print('  scan plane tilt from world up (truth pose): mean %.2f deg' % a[ok, 7].mean())
    np.savez(OUT, res=a)
    print('wrote %s  total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
