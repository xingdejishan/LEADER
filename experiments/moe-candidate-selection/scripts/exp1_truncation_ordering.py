#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP1: does any cost setting rank the truth pose above the front-end pose?

Two poses only (front-end t0, ground truth), 43 frames whose front-end position
error is 5-8 m.  The nearest-neighbour distance vector is computed ONCE per
(frame, pose); every cost setting is then a monotone transform of that same
vector, so nine settings cost nothing more than one.

Map construction, scan preparation and query subset are the project's own:
  fast_lib.MapSource(inputs/refs.json, radius=60, 3 references)
  pipeline.prepare_scan(record, Config())        # decode, 1-60 m, voxel 0.3
  fast_lib.spread_subset(scan, 4000)

Read-only; writes one npz into the scratch dir.
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
OUT = '/home/zhangcy/.dsh_scratch/exp1_costs.npz'

TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def kernels():
    """name -> f(raw distance) ; cap = distance bound the tree query may prune at.

    For the truncated means the cap IS the radius, so `min(d, tau)` comes out
    exact.  For the soft kernels the cap is only a numerical floor; 60 m is the
    far edge of the scan filter, and beyond it both kernels are already flat
    (Huber within 1 %, Geman-McClure within 0.03 %).
    """
    out = {}
    for tau in TAUS:
        out['trunc_%.1f' % tau] = (lambda d, t=tau: np.minimum(d, t), tau)
    out['huber_1.0'] = (lambda d: np.where(d <= 1.0, 0.5 * d * d, d - 0.5), 60.0)
    out['gm_1.0'] = (lambda d: d * d / (d * d + 1.0), 60.0)
    return out


def main():
    t_start = time.time()
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()

    blob = np.load(os.path.join(PROJECT, 'results/full-online.npz'), allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    t0 = np.asarray(blob['t0'], np.float64)
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        date = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)

    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)
    band = np.where((e0 >= 5.0) & (e0 <= 8.0))[0]
    print('frames in 5-8 m band: %d   (mean front-end error %.2f m)'
          % (len(band), e0[band].mean()))

    K = kernels()
    names = list(K)
    raw_front, raw_truth, nq = [], [], []
    for j, i in enumerate(band):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        tree = cKDTree(mp)
        for dst, pose in ((raw_front, t0[i]), (raw_truth, gt[i])):
            d = tree.query(q @ pose[:3, :3].T + pose[:3, 3], k=1, workers=WORKERS,
                           distance_upper_bound=60.0)[0]
            dst.append(np.where(np.isfinite(d), d, 60.0))
        nq.append(len(q))
        if (j + 1) % 10 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(band), time.time() - t_start), flush=True)

    # ---- the table
    print('\n%-12s %10s %10s %8s %9s %9s' % ('cost setting', 'truth wins', 'front wins',
                                             'ties', 'mean gap', 'med gap'))
    rows = []
    for name in names:
        fn, cap = K[name]
        cf = np.array([fn(raw_front[j]).mean() for j in range(len(band))])
        ct = np.array([fn(raw_truth[j]).mean() for j in range(len(band))])
        gap = ct - cf
        wins = int((gap < 0).sum())
        ties = int((gap == 0).sum())
        rows.append((name, wins, int((gap > 0).sum()), ties, float(gap.mean()),
                     float(np.median(gap))))
        print('%-12s %10d %10d %8d %9.3f %9.3f'
              % (name, wins, int((gap > 0).sum()), ties, gap.mean(), np.median(gap)))

    # saturation: how many of the residuals sit at the cap, per pose
    print('\nsaturation at tau = 1 m (fraction of the residuals >= 1 m):')
    sat_f = float(np.concatenate([r >= 1.0 for r in raw_front]).mean())
    sat_t = float(np.concatenate([r >= 1.0 for r in raw_truth]).mean())
    pf = np.array([(r >= 1.0).mean() for r in raw_front])
    pt = np.array([(r >= 1.0).mean() for r in raw_truth])
    print('  pooled front-end pose %.3f   truth pose %.3f' % (sat_f, sat_t))
    print('  per-frame front-end saturation: median %.3f, min %.3f, max %.3f'
          % (np.median(pf), pf.min(), pf.max()))
    print('  per-frame truth-pose saturation: median %.3f, min %.3f, max %.3f'
          % (np.median(pt), pt.min(), pt.max()))
    print('  query points per frame: min %d, median %d, max %d'
          % (min(nq), int(np.median(nq)), max(nq)))

    # reachability: can the branch's grid even contain the truth translation?
    u = t0[band, :3, 0] / np.linalg.norm(t0[band, :3, 0], axis=1, keepdims=True)
    lat = np.cross(u, np.array([0.0, 0.0, 1.0]))
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    delta = gt[band, :3, 3] - t0[band, :3, 3]
    s_t = (delta * u).sum(1)
    d_t = (delta * lat).sum(1)
    out_of_plane = np.abs((delta * np.cross(u, lat)).sum(1))
    reach = (np.abs(s_t) <= 8.0) & (np.abs(d_t) <= 2.0)
    print('\ntruth translation in the search frame: |s|>8 m %d, |d|>2 m %d, '
          'both inside %d/43' % (int((np.abs(s_t) > 8).sum()), int((np.abs(d_t) > 2).sum()),
                                 int(reach.sum())))
    print('  |s| percentiles %s   |d| percentiles %s   vertical (out of plane) median %.2f m'
          % (np.round(np.percentile(np.abs(s_t), [50, 90]), 2),
             np.round(np.percentile(np.abs(d_t), [50, 90]), 2), np.median(out_of_plane)))
    print('  rotation error front-end vs truth: median %.2f deg, p90 %.2f deg'
          % (np.median([np.degrees(np.arccos(np.clip(
              (np.trace(t0[i][:3, :3].T @ gt[i][:3, :3]) - 1) / 2, -1, 1))) for i in band]),
             np.percentile([np.degrees(np.arccos(np.clip(
                 (np.trace(t0[i][:3, :3].T @ gt[i][:3, :3]) - 1) / 2, -1, 1))) for i in band],
                 90)))

    np.savez(OUT, band=band, e0=e0, names=np.array(names), s_t=s_t, d_t=d_t,
             reach=reach, nq=np.array(nq),
             raw_front=np.array(raw_front, dtype=object),
             raw_truth=np.array(raw_truth, dtype=object))
    print('\nwrote %s   total %.1f s' % (OUT, time.time() - t_start))


if __name__ == '__main__':
    main()
