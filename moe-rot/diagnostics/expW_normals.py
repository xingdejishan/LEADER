"""EXP W: do surface normals carry the rotation information that positions lack?

The measured reason MOE will not move is that the correspondence score compares
*positions*: a wrong rotation slides points along the surface they belong to and
costs almost nothing.  A surface normal does not slide - a 1.4 deg rotation error
tilts every matched normal by 1.4 deg regardless of tangentially sliding points.

So: at the front-end pose and at the truth pose, compare
  (a) the mean nearest-map-point distance   (the deployed cost)
  (b) the mean angle between the matched normals   (the missing channel)
and ask which one ranks the truth pose first.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expW_normals.npz'
N_BULK = int(os.environ.get('NB', '80'))
N_MID = int(os.environ.get('NM', '40'))
K_NB = 12


def normals(pts, k=K_NB):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)), workers=4)
    nb = pts[idx]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    # curvature proxy: how planar the neighbourhood is
    lam = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    return n, lam


def score(tree, n_map, q, n_q, R, t, trunc=1.0, normal_gate=1.0):
    qw = q @ R.T + t
    nw = n_q @ R.T
    d, idx = tree.query(qw, k=1, workers=4, distance_upper_bound=trunc)
    ok = np.isfinite(d)
    if ok.sum() < 20:
        return None
    cos = np.abs((nw[ok] * n_map[idx[ok]]).sum(1))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return dict(dist=float(np.mean(np.minimum(d[ok], trunc))),
                nang=float(np.mean(np.minimum(ang, 90.0))),
                nang_med=float(np.median(ang)),
                n=int(ok.sum()))


def main():
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

    rng = np.random.default_rng(20261022)
    pick = np.concatenate([rng.choice(np.where(e0 < 0.2)[0], N_BULK, replace=False),
                           rng.choice(np.where((e0 >= 0.2) & (e0 < 0.5))[0], N_MID,
                                      replace=False)])
    rows = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        n_q, curv = normals(q)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        tree = cKDTree(mp)
        n_map, _ = normals(mp)
        st0 = score(tree, n_map, q, n_q, t0[i][:3, :3], t0[i][:3, 3])
        sgt = score(tree, n_map, q, n_q, gt[i][:3, :3], gt[i][:3, 3])
        if st0 is None or sgt is None:
            continue
        rows.append(dict(i=int(i), rms=float(np.degrees(np.mean(
            2 * np.arcsin(np.clip(np.sqrt(curv), 0, 1))))),
                         dist_t0=st0['dist'], dist_gt=sgt['dist'],
                         nang_t0=st0['nang'], nang_gt=sgt['nang'],
                         nmed_t0=st0['nang_med'], nmed_gt=sgt['nang_med'],
                         moe=float(np.degrees(np.arccos(np.clip(
                             (np.trace(t0[i][:3, :3].T @ gt[i][:3, :3]) - 1) / 2, -1, 1))))))
        if (j + 1) % 30 == 0:
            print('  %d/%d' % (j + 1, len(pick)), flush=True)

    d = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    print('\n## %d frames' % len(rows))
    print('   position cost : t0 %.4f m   truth %.4f m   truth better on %d/%d'
          % (d['dist_t0'].mean(), d['dist_gt'].mean(),
             int((d['dist_gt'] < d['dist_t0']).sum()), len(rows)))
    print('   normal  angle : t0 %.3f deg  truth %.3f deg  truth better on %d/%d'
          % (d['nang_t0'].mean(), d['nang_gt'].mean(),
             int((d['nang_gt'] < d['nang_t0']).sum()), len(rows)))
    print('   normal  median: t0 %.3f deg  truth %.3f deg'
          % (np.median(d['nmed_t0']), np.median(d['nmed_gt'])))
    # does the normal gap grow with the rotation error?  (is it the signal we want)
    print('\n   rotation error vs normal-angle gap (truth - t0):')
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 90)):
        m = (d['moe'] >= lo) & (d['moe'] < hi)
        if m.sum() < 3:
            continue
        print('     MOE %g-%-4g n=%3d  d_norm %+7.3f  d_dist %+7.4f  truth wins(norm) %d/%d'
              % (lo, hi, int(m.sum()), (d['nang_gt'] - d['nang_t0'])[m].mean(),
                 (d['dist_gt'] - d['dist_t0'])[m].mean(),
                 int((d['nang_gt'][m] < d['nang_t0'][m]).sum()), int(m.sum())))
    np.savez(OUT, **d)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
