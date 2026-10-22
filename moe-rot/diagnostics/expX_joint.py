"""EXP X: joint 3-dof rotation optimisation of the deployed cost.

The earlier test took the cost-minimising angle on each axis independently and composed
the three - that is a search strategy, not a joint optimisation, so it cannot rule out
"a proper 3-dof optimisation would recover the rotation".  This does the joint version:

    for each frame, translation fixed at the front-end pose, minimise
        C(R) = mean truncated nearest-neighbour distance
    over R in SO(3) directly, by a pattern search on the Lie algebra, from several
    starting rotations (the front end itself and small perturbations of it).

Reported: MOE at the front-end pose, at the joint optimum of the cost, how much the cost
improved, and whether the optimum escapes the front-end basin.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expX_joint.npz'
N_BULK = int(os.environ.get('NB', '60'))
N_MID = int(os.environ.get('NM', '30'))
N_HARD = int(os.environ.get('NH', '30'))
TRUNC = 1.0
STEPS = (1.5, 0.75, 0.4, 0.2, 0.1, 0.05)     # degrees
AXES = np.eye(3)


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def main():
    t_start = time.time()
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()
    blob = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    t0 = np.asarray(blob['t0'], np.float64)
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        d = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)

    rng = np.random.default_rng(99)
    pick = np.concatenate([
        rng.choice(np.where(e0 < 0.2)[0], N_BULK, replace=False),
        rng.choice(np.where((e0 >= 0.2) & (e0 < 0.5))[0], N_MID, replace=False),
        rng.choice(np.where((e0 >= 0.5) & (e0 < 5.0))[0], N_HARD, replace=False)])
    print('frames %d  MPE %.2f  MOE %.2f'
          % (len(pick), e0[pick].mean(),
             ra(t0[pick, :3, :3], gt[pick, :3, :3]).mean()), flush=True)

    rows = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        tree = cKDTree(mp)

        def cost(R):
            d = tree.query(q @ R.T + centre, k=1, workers=2,
                           distance_upper_bound=TRUNC)[0]
            return float(np.where(np.isfinite(d), d, TRUNC).mean())

        R_best, c_best = t0[i][:3, :3].copy(), None
        starts = [np.eye(3)] + [exp_so3(np.radians(a) * ax)
                                for ax in AXES for a in (2.0, -2.0)]
        c_best = cost(R_best)
        for start in starts:
            R = start @ t0[i][:3, :3]
            c = cost(R)
            for step in STEPS:
                improved = True
                passes = 0
                while improved and passes < 3:
                    improved = False
                    passes += 1
                    for ax in AXES:
                        for sign in (1.0, -1.0):
                            Rt = exp_so3(np.radians(sign * step) * ax) @ R
                            ct = cost(Rt)
                            if ct < c - 1e-6:
                                R, c = Rt, ct
                                improved = True
            if c < c_best:
                R_best, c_best = R.copy(), c
        rows.append(dict(i=int(i), c_t0=float(cost(t0[i][:3, :3])), c_best=float(c_best),
                         moe_t0=float(ra(t0[i, :3, :3], gt[i, :3, :3])),
                         moe_opt=float(ra(R_best, gt[i, :3, :3])),
                         kick=float(ra(t0[i, :3, :3], R_best)),
                         mpe0=float(e0[i])))
        if (j + 1) % 20 == 0:
            print('  %d/%d  %.0f s' % (j + 1, len(pick), time.time() - t_start), flush=True)

    d = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    d['i'] = d['i'].astype(int)
    print('\n## %d frames, joint 3-dof rotation optimisation (translation fixed)' % len(rows))
    print('   cost   %.4f -> %.4f   (improved on %d/%d frames)'
          % (d['c_t0'].mean(), d['c_best'].mean(),
             int((d['c_best'] < d['c_t0'] - 1e-6).sum()), len(rows)))
    print('   MOE    %.3f -> %.3f   (improved on %d, worsened on %d)'
          % (d['moe_t0'].mean(), d['moe_opt'].mean(),
             int((d['moe_opt'] < d['moe_t0']).sum()), int((d['moe_opt'] > d['moe_t0']).sum())))
    print('   median MOE %.3f -> %.3f ; mean |rotation kicked| %.3f deg'
          % (np.median(d['moe_t0']), np.median(d['moe_opt']), d['kick'].mean()))
    print('\n   by band:')
    for lo, hi, name in ((0, 0.2, 'bulk'), (0.2, 0.5, 'mid'), (0.5, 9, 'hard')):
        m = (d['mpe0'] >= lo) & (d['mpe0'] < hi)
        if m.sum() < 3:
            continue
        print('     %-5s n=%3d  cost %.4f->%.4f  MOE %.3f->%.3f  kick %.2f'
              % (name, int(m.sum()), d['c_t0'][m].mean(), d['c_best'][m].mean(),
                 d['moe_t0'][m].mean(), d['moe_opt'][m].mean(), d['kick'][m].mean()))
    corr = np.corrcoef(d['kick'], d['moe_opt'] - d['moe_t0'])[0, 1]
    print('\n   corr(kick size, MOE change) = %+.3f  (>0 means bigger kicks hurt more)'
          % corr)
    np.savez(OUT, **d)
    print('wrote %s  total %.0f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
