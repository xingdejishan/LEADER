"""Debug: what exactly does the road plane say about the front-end tilt error?

Prints, for a handful of frames, the body-frame ground normal, the map ground
normal, the front-end tilt error, and the cost at the pose levelled by the road
plane, so the pieces can be checked against each other.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')


def fit_plane(pts, band=0.4):
    z = pts[:, 2]
    lo, hi = np.percentile(z, [2, 40])
    cand = pts[(z >= lo) & (z <= hi)]
    hist, edges = np.histogram(cand[:, 2], bins=40)
    k = int(np.argmax(hist))
    zc = 0.5 * (edges[k] + edges[k + 1])
    sel = cand[np.abs(cand[:, 2] - zc) < band]
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
    return n, float(np.sqrt((r ** 2).mean())), len(sel), zc


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


def ang(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def main():
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
    rng = np.random.default_rng(7)
    pick = rng.choice(np.where(e0 < 0.2)[0], 6, replace=False)

    up = np.array([0.0, 0, 1.0])
    for i in pick:
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        print('\n=== %s   scan pts %d   z range [%.2f, %.2f]  z mean %.2f'
              % (key, len(scan), scan[:, 2].min(), scan[:, 2].max(), scan[:, 2].mean()))
        h, e = np.histogram(scan[:, 2], bins=30)
        top = np.argsort(h)[-4:][::-1]
        print('    z hist top bins:', ['%.2f:%d' % (0.5 * (e[k] + e[k + 1]), h[k]) for k in top])
        n_b, rms_b, nb, zc = fit_plane(scan, 0.5)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        d2 = np.linalg.norm(mp[:, :2] - centre[None, :2], axis=1)
        n_w, rms_w, nm, zc2 = fit_plane(mp[d2 < 25.0], 0.4)
        R, p = t0[i][:3, :3], t0[i][:3, 3]
        Rg = gt[i][:3, :3]
        print('    body normal %s  rms %.3f inl %d zc %.2f' % (np.round(n_b, 4), rms_b, nb, zc))
        print('    map  normal %s  rms %.3f inl %d zc %.2f (map pts %d, near %d)'
              % (np.round(n_w, 4), rms_w, nm, zc2, len(mp), int((d2 < 25).sum())))
        print('    angle(R_gt n_b, up) %.2f deg ; angle(R_t0 n_b, up) %.2f deg ; angle(R_t0 n_b, n_w) %.2f'
              % (ang(Rg @ n_b, up), ang(R @ n_b, up), ang(R @ n_b, n_w)))
        # tilt error of t0 relative to truth
        Rt = R.T @ Rg
        a_tot = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        print('    angle(R_gt z_body, R_t0 z_body) = tilt error %.2f deg ; total err %.2f deg'
              % (ang(R @ np.array([0, 0, 1.0]), Rg @ np.array([0, 0, 1.0])), a_tot))
        # cost at t0 vs at the road-levelled pose
        q = F.spread_subset(scan, 4000)
        tree = cKDTree(mp)
        def cost(Rx, tx):
            d = tree.query(q @ Rx.T + tx, k=1, workers=4, distance_upper_bound=60.0)[0]
            d = np.where(np.isfinite(d), d, 60.0)
            return float(np.minimum(d, 1.0).mean())
        Rc = rot_between(R @ n_b, n_w) @ R
        Rc_or = rot_between(R @ n_b, Rg @ n_b) @ R
        print('    cost t0 %.4f | road-levelled %.4f | oracle-levelled %.4f | truth %.4f'
              % (cost(R, p), cost(Rc, p), cost(Rc_or, p), cost(Rg, gt[i][:3, 3])))


if __name__ == '__main__':
    main()
