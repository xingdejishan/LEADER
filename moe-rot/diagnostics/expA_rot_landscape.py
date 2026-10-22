"""EXP A: does the geometric cost carry rotation information in the bulk?

For a stratified sample of the 104,728-frame set we hold the translation fixed and
sweep rotation about the three body axes (and world z).  We record C(theta) at:
  * the front-end pose t0
  * the ground-truth pose
and then ask
  (1) is C(t0) below C(truth) even when the position is already sub-decimetre?
  (2) is the local cost slope aligned with the direction that reduces rotation error
      (regression of dC/dtheta on the true relative angle)?
  (3) how much MOE would an oracle on this cost actually remove?
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
WORKERS = int(os.environ.get('W', '4'))
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expA_rot_landscape.npz'
TAU = 1.0

ANGLES = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
N_BULK = int(os.environ.get('NB', '120'))
N_MID = int(os.environ.get('NM', '60'))


def exp_so3(axis, th_deg):
    th = np.radians(th_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot_angle(A, B):
    R = A.T @ B
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def log_vec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(ang)
    if s < 1e-8:
        return 0.5 * v
    return v * ang / (2.0 * s)


def cost(tree, q, R, t):
    d = tree.query(q @ R.T + t, k=1, workers=WORKERS, distance_upper_bound=60.0)[0]
    d = np.where(np.isfinite(d), d, 60.0)
    return float(np.minimum(d, TAU).mean()), d


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
    ang0 = np.array([rot_angle(t0[i][:3, :3], gt[i][:3, :3]) for i in range(len(keys))])

    rng = np.random.default_rng(20261022)
    bulk = np.where(e0 < 0.2)[0]
    mid = np.where((e0 >= 0.2) & (e0 < 0.5))[0]
    pick = np.concatenate([rng.choice(bulk, N_BULK, replace=False),
                           rng.choice(mid, N_MID, replace=False)])
    print('sample %d frames (bulk %d + mid %d); front-end MPE %.3f / MOE %.2f'
          % (len(pick), N_BULK, N_MID, e0[pick].mean(), ang0[pick].mean()))

    axes = {'bodyx': np.array([1.0, 0, 0]), 'bodyy': np.array([0, 1.0, 0]),
            'bodyz': np.array([0, 0, 1.0]), 'worldz': np.array([0, 0, 1.0])}
    curves = {a: np.zeros((len(pick), len(ANGLES))) for a in axes}      # at t0 (translation t0)
    curves_t = {a: np.zeros((len(pick), len(ANGLES))) for a in axes}    # at truth (translation gt)
    c_t0 = np.zeros(len(pick))
    c_gt = np.zeros(len(pick))
    c_gtR_t0t = np.zeros(len(pick))   # truth rotation, t0 translation
    nq = np.zeros(len(pick), int)

    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        nq[j] = len(q)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        tree = cKDTree(mp)
        R0, p0 = t0[i][:3, :3], t0[i][:3, 3]
        Rg, pg = gt[i][:3, :3], gt[i][:3, 3]
        c_t0[j] = cost(tree, q, R0, p0)[0]
        c_gt[j] = cost(tree, q, Rg, pg)[0]
        c_gtR_t0t[j] = cost(tree, q, Rg, p0)[0]
        for a, ax in axes.items():
            for m, th in enumerate(ANGLES):
                if a == 'worldz':
                    curves[a][j, m] = cost(tree, q, exp_so3(ax, th) @ R0, p0)[0]
                    curves_t[a][j, m] = cost(tree, q, exp_so3(ax, th) @ Rg, pg)[0]
                else:
                    curves[a][j, m] = cost(tree, q, R0 @ exp_so3(ax, th), p0)[0]
                    curves_t[a][j, m] = cost(tree, q, Rg @ exp_so3(ax, th), pg)[0]
        if (j + 1) % 20 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(pick), time.time() - t_start), flush=True)

    print('\n## C(front-end) vs C(truth), translation kept as-is')
    print('   mean C(t0) %.4f   mean C(truth) %.4f   mean C(truthR,t0t) %.4f'
          % (c_t0.mean(), c_gt.mean(), c_gtR_t0t.mean()))
    print('   frames where truth rotation scores better (at t0 translation): %d/%d'
          % (int((c_gtR_t0t < c_t0).sum()), len(pick)))
    print('   frames where truth pose scores better: %d/%d' % (int((c_gt < c_t0).sum()), len(pick)))

    print('\n## mean cost curve around t0 (translation fixed), pass/fail vs C(0)')
    for a in axes:
        base = curves[a][:, list(ANGLES).index(0.0)]
        rel = curves[a] - base[:, None]
        s = '   %-7s ' % a + ' '.join('%+.4f' % v for v in rel.mean(0))
        print(s)
        # oracle: best angle per frame
        bi = curves[a].argmin(1)
        print('             oracle theta median %+.1f  |  frames whose min is at theta=0: %d%%'
              % (np.median(ANGLES[bi]), int(100 * (ANGLES[bi] == 0).mean())))

    print('\n## truth is the minimum? (curve evaluated around the truth pose)')
    for a in axes:
        base = curves_t[a][:, list(ANGLES).index(0.0)]
        rel = curves_t[a] - base[:, None]
        print('   %-7s ' % a + ' '.join('%+.4f' % v for v in rel.mean(0)))

    print('\n## slope-vs-truth regression (is the gradient informative?)')
    for a in axes:
        # true rotation from t0 to gt, axis component of its log map
        Rt = t0[pick, :3, :3].transpose(0, 2, 1) @ gt[pick, :3, :3]
        if a == 'worldz':
            Rw = t0[pick, :3, :3] @ Rt @ t0[pick, :3, :3].transpose(0, 2, 1)
            comp = np.array([log_vec(R)[2] for R in Rw])
        else:
            k = {'bodyx': 0, 'bodyy': 1, 'bodyz': 2}[a]
            comp = np.array([log_vec(R)[k] for R in Rt])
        th_true = np.degrees(comp)
        i1 = list(ANGLES).index(-1.0)
        i2 = list(ANGLES).index(1.0)
        slope = curves[a][:, i2] - curves[a][:, i1]
        cc = np.corrcoef(th_true, slope)[0, 1]
        print('   %-7s corr(true angle, C(+1)-C(-1)) = %+.3f  ;  mean|angle| %.2f deg  mean dC %.5f'
              % (a, cc, np.abs(th_true).mean(), np.abs(slope).mean()))

    print('\n## oracle on the full rotation grid (all four axes swept independently, pick best per axis)')
    et0 = np.linalg.norm(t0[pick, :3, 3] - gt[pick, :3, 3], axis=1)
    best = np.zeros(len(pick))
    # compose the four independent argmins into one rotation
    for m, i in enumerate(pick):
        R = t0[i][:3, :3].copy()
        for a in ('bodyx', 'bodyy', 'bodyz'):
            th = ANGLES[curves[a][m].argmin()]
            R = R @ exp_so3(axes[a], th)
        th = ANGLES[curves['worldz'][m].argmin()]
        R = exp_so3(axes['worldz'], th) @ R
        best[m] = rot_angle(R, gt[i][:3, :3])
    print('   MOE %.4f -> oracle-cost MOE %.4f  (frames improved %d, worsened %d)'
          % (ang0[pick].mean(), best.mean(), int((best < ang0[pick]).sum()), int((best > ang0[pick]).sum())))

    np.savez(OUT, pick=pick, angles=ANGLES, c_t0=c_t0, c_gt=c_gt, c_gtR_t0t=c_gtR_t0t,
             nq=nq, ang0=ang0[pick], e0=e0[pick],
             **{'curve_%s' % a: curves[a] for a in axes},
             **{'curveT_%s' % a: curves_t[a] for a in axes})
    print('\nwrote %s  total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
