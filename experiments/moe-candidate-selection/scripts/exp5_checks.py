#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two last checks, both cheap.

A. per-setting truth-wins restricted to the 25 frames that never move (reads the
   npz Experiment 1 wrote; no new map work).
B. is the poor fit at the truth pose a map-coverage artefact?  Re-crop the local
   map around the TRUTH position (nearest 3 references, radius 60) and score the
   truth pose there.  If it does not improve, the reference cloud simply does not
   match this scan at the truth pose.

Read-only.
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
NPZ = '/home/zhangcy/.dsh_scratch/exp1_costs.npz'

TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
K = {}
for tau in TAUS:
    K['trunc_%.1f' % tau] = (lambda d, t=tau: np.minimum(d, t))
K['huber_1.0'] = lambda d: np.where(d <= 1.0, 0.5 * d * d, d - 0.5)
K['gm_1.0'] = lambda d: d * d / (d * d + 1.0)
K['uncapped_mean'] = lambda d: d


def main():
    tick = time.time()
    z = np.load(NPZ, allow_pickle=True)
    band = z['band']
    rf = [np.asarray(a, np.float64) for a in z['raw_front']]
    rt = [np.asarray(a, np.float64) for a in z['raw_truth']]

    blob = np.load(os.path.join(PROJECT, 'results/full-online.npz'), allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    rows = {}
    for name in ('full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl'):
        with open(os.path.join(PROJECT, 'results', name)) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('key') and r.get('seconds'):
                    rows[r['key']] = r
    stuck = np.array([abs(rows[keys[i]]['s']) < 1e-9 for i in band])
    print('A. truth-wins on the %d never-moving frames, per cost setting'
          % int(stuck.sum()))
    print('   %-14s %10s %10s' % ('setting', 'truth wins', 'mean gap'))
    for name in list(K) + ['uncapped_mean']:
        fn = K[name]
        cf = np.array([fn(rf[j]).mean() for j in range(len(band))])
        ct = np.array([fn(rt[j]).mean() for j in range(len(band))])
        print('   %-14s %10d %+10.3f' % (name, int((ct[stuck] < cf[stuck]).sum()),
                                         (ct[stuck] - cf[stuck]).mean()))

    # ---------------- B: crop around truth instead of around the front end
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        date = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    t0 = np.asarray(blob['t0'], np.float64)

    a_front, a_truth, b_truth, ntriple_same = [], [], [], 0
    for n, i in enumerate(band):
        date, ts = keys[i].split('/')
        rec = {'key': keys[i],
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        tri_f = maps.nearby(t0[i][:3, 3])[:3]
        tri_t = maps.nearby(gt[i][:3, 3])[:3]
        ntriple_same += int(tri_f == tri_t)
        mpf = maps.local(t0[i][:3, 3], tri_f)
        mpt = maps.local(gt[i][:3, 3], tri_t)
        for dst, pose, mp in ((a_front, t0[i], mpf), (a_truth, gt[i], mpf),
                              (b_truth, gt[i], mpt)):
            d = cKDTree(mp).query(q @ pose[:3, :3].T + pose[:3, 3], k=1, workers=WORKERS,
                                  distance_upper_bound=60.0)[0]
            dst.append(np.where(np.isfinite(d), d, 60.0))
    print('\nB. map cropped around the front end vs around truth '
          '(%d/43 frames use the same reference triple)'
          % ntriple_same)
    for label, L in (('front pose, crop@front', a_front),
                     ('truth pose, crop@front', a_truth),
                     ('truth pose, crop@truth', b_truth)):
        u = np.array([np.minimum(x, 1.0).mean() for x in L])
        g = np.array([K['gm_1.0'](x).mean() for x in L])
        r = np.array([x.mean() for x in L])
        print('   %-24s trunc_1 %.3f   gm %.3f   uncapped %.3f'
              % (label, u.mean(), g.mean(), r.mean()))
    print('\ntotal %.0f s' % (time.time() - tick))


if __name__ == '__main__':
    main()
