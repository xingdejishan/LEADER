"""EXP F: (1) is the attitude stale by whole frames?  (2) does a plane-based cost
see rotation where the nearest-neighbour cost does not?

Part 1 compares t0 against the truth trajectory shifted by -2..+2 frames, which
separates a latency problem from a bias.

Part 2 replaces the nearest-map-point distance with the distance to the map's
local tangent *plane* (normals from the map), which is the standard fix for a
cost that is flat along surfaces, and re-runs the ordering test
(C(t0) vs C(truth)) plus the gradient-truth correlation.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expF_plane_cost.npz'
N_BULK = int(os.environ.get('NB', '90'))
N_MID = int(os.environ.get('NM', '60'))
ANGLES = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])


def exp_so3(axis, th_deg):
    th = np.radians(th_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log_vec(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)


def rot_angle(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def map_normals(pts, tree=None, k=12):
    tree = tree or cKDTree(pts)
    _, idx = tree.query(pts, k=k, workers=4)
    nb = pts[idx]                        # n x k x 3
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                    # smallest eigenvector


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

    # ---------- part 1: whole-frame shift
    print('## part 1: MOE / MPE against the truth trajectory shifted by whole frames')
    order = sorted(range(len(keys)), key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1])))
    order = np.array(order)
    dates = np.array([keys[i].split('/')[0] for i in order])
    print('   %-7s %9s %9s' % ('shift', 'MOE', 'MPE'))
    for s in (-2, -1, 0, 1, 2):
        j = order + s
        keep = (j >= 0) & (j < len(order))
        j = np.clip(j, 0, len(order) - 1)
        same = keep & (dates == dates[np.clip(np.arange(len(order)), 0, len(order) - 1)])
        i2 = order[same]
        j2 = j[same]
        moe = rot_angle(t0[i2, :3, :3], gt[j2, :3, :3]).mean()
        mpe = np.linalg.norm(t0[i2, :3, 3] - gt[j2, :3, 3], axis=1).mean()
        print('   %-7s %9.4f %9.4f' % ('%+d' % s, moe, mpe))

    # ---------- part 2: plane cost
    rng = np.random.default_rng(20261022)
    pick = np.concatenate([rng.choice(np.where(e0 < 0.2)[0], N_BULK, replace=False),
                           rng.choice(np.where((e0 >= 0.2) & (e0 < 0.5))[0], N_MID, replace=False)])
    up = np.array([0.0, 0, 1.0])
    rows = []
    for j_, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        p = t0[i][:3, 3]
        mp = maps.local(p, maps.nearby(p)[:3])
        near = mp[np.linalg.norm(mp[:, :2] - p[None, :2], axis=1) < 30.0]
        tree = cKDTree(near)
        nrm = map_normals(near, tree, k=12)

        def nn_cost(Rx, tx):
            d = tree.query(q @ Rx.T + tx, k=1, workers=4, distance_upper_bound=60.0)
            d = np.where(np.isfinite(d[0]), d[0], 60.0)
            return float(np.minimum(d, 1.0).mean())

        def pl_cost(Rx, tx):
            d, idx = tree.query(q @ Rx.T + tx, k=1, workers=4, distance_upper_bound=60.0)
            okq = np.isfinite(d)
            res = np.full(len(q), 0.5)
            if okq.any():
                v = (q[okq] @ Rx.T + tx) - near[idx[okq]]
                res[okq] = np.abs((v * nrm[idx[okq]]).sum(1))
            return float(np.minimum(res, 0.5).mean())

        R, Rg = t0[i][:3, :3], gt[i][:3, :3]
        row = dict(i=i, nn_t0=nn_cost(R, p), nn_gt=nn_cost(Rg, gt[i][:3, 3]),
                   pl_t0=pl_cost(R, p), pl_gt=pl_cost(Rg, gt[i][:3, 3]),
                   nn_gtR=nn_cost(Rg, p), pl_gtR=pl_cost(Rg, p))
        Rt = R.T @ Rg
        w = -log_vec(Rt)
        for ax, name in ((np.array([1.0, 0, 0]), 'x'), (np.array([0, 1.0, 0]), 'y'),
                         (np.array([0, 0, 1.0]), 'z')):
            for base, tag in ((R, 't0'), (Rg, 'gt')):
                c = [pl_cost(base @ exp_so3(ax, a), p) for a in ANGLES]
                col = ['pl_%s_%s_%s' % (tag, name, k) for k in range(len(ANGLES))]
                for k, cc in zip(col, c):
                    row[k] = cc
        row['true_x'], row['true_y'], row['true_z'] = w[0], w[1], w[2]
        rows.append(row)
        if (j_ + 1) % 30 == 0:
            print('  %d/%d  %.1f s' % (j_ + 1, len(pick), time.time() - t_start), flush=True)

    import collections
    allk = sorted(rows[0].keys())
    A = collections.defaultdict(list)
    for r in rows:
        for k in allk:
            A[k].append(r[k])
    d = {k: np.array(v, float) for k, v in A.items()}

    print('\n## part 2: ordering test (mean cost, translation as-is)')
    print('   NN   : C(t0) %.4f  C(truth) %.4f  C(truthR) %.4f   truthR wins %d/%d'
          % (d['nn_t0'].mean(), d['nn_gt'].mean(), d['nn_gtR'].mean(),
             int((d['nn_gtR'] < d['nn_t0']).sum()), len(rows)))
    print('   PLANE: C(t0) %.4f  C(truth) %.4f  C(truthR) %.4f   truthR wins %d/%d'
          % (d['pl_t0'].mean(), d['pl_gt'].mean(), d['pl_gtR'].mean(),
             int((d['pl_gtR'] < d['pl_t0']).sum()), len(rows)))

    print('\n## plane cost: gradient vs true rotation error (correlation over frames)')
    i0 = list(ANGLES).index(0.0)
    i1 = list(ANGLES).index(1.0)
    i2 = list(ANGLES).index(-1.0)
    for ax in ('x', 'y', 'z'):
        slope = d['pl_t0_%s_%d' % (ax, i1)] - d['pl_t0_%s_%d' % (ax, i2)]
        true = d['true_%s' % ax]
        print('   axis %s: corr = %+.3f ; mean|dC| = %.5f ; mean|true| %.2f deg'
              % (ax, np.corrcoef(true, slope)[0, 1], np.abs(slope).mean(), np.abs(true).mean()))

    print('\n## plane cost around the truth pose (is truth the minimum?)')
    for ax in ('x', 'y', 'z'):
        base = d['pl_gt_%s_%d' % (ax, i0)]
        rel = np.array([np.mean(d['pl_gt_%s_%d' % (ax, k)] - base) for k in range(len(ANGLES))])
        print('   %s  ' % ax + ' '.join('%+.5f' % v for v in rel))

    print('\n## oracle on the plane cost (per-frame best angle on each axis)')
    best = np.zeros(len(rows))
    Rt_all = np.zeros((len(rows), 3, 3))
    for m, i in enumerate(pick):
        R = t0[i][:3, :3].copy()
        for ax, name in ((np.array([1.0, 0, 0]), 'x'), (np.array([0, 1.0, 0]), 'y'),
                         (np.array([0, 0, 1.0]), 'z')):
            vals = [d['pl_t0_%s_%d' % (name, k)][m] for k in range(len(ANGLES))]
            R = R @ exp_so3(ax, ANGLES[int(np.argmin(vals))])
        best[m] = rot_angle(R, gt[i][:3, :3])
    now = rot_angle(t0[pick, :3, :3], gt[pick, :3, :3])
    print('   MOE %.4f -> plane-oracle %.4f  (improved %d, worsened %d)'
          % (now.mean(), best.mean(), int((best < now).sum()), int((best > now).sum())))

    np.savez(OUT, pick=pick, angles=ANGLES, **d)
    print('\nwrote %s  total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
