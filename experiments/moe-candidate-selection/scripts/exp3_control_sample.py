#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CONTROL: what does the best new cost do to ordinary frames?

Part A  sanity + reproduction of the brief's numbers (C(front)=0.34, C(truth)=0.60)
        on the 43 band frames and on the 25 of them that never move.
Part B  500 ordinary frames (front-end error <= 1 m): does the cost still rank the
        front-end pose first, i.e. does the change do no damage on the bulk?
Part C  same, per date, to make sure the sample is not all one sequence.

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
N_CONTROL = 500

K = {}
K['trunc_1.0'] = lambda d: np.minimum(d, 1.0)
K['gm_1.0'] = lambda d: d * d / (d * d + 1.0)
K['huber_1.0'] = lambda d: np.where(d <= 1.0, 0.5 * d * d, d - 0.5)


def main():
    t0_clock = time.time()
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

    rowS = {}
    for name in ('full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl'):
        with open(os.path.join(PROJECT, 'results', name)) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('key') and r.get('seconds'):
                    rowS[r['key']] = r
    s_val = np.array([rowS[k]['s'] if k in rowS else np.nan for k in keys])

    def raw_dists(indices, want_truth=True):
        """(front raw, truth raw) distance lists, project map/scan/query set."""
        fr, tr = [], []
        for n, i in enumerate(indices):
            key = keys[i]
            date, ts = key.split('/')
            rec = {'key': key,
                   'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
            scan = P.prepare_scan(rec, cfg)
            q = F.spread_subset(scan, 4000)
            centre = t0[i][:3, 3]
            mp = maps.local(centre, maps.nearby(centre)[:3])
            tree = cKDTree(mp)
            for dst, pose in ((fr, t0[i]), (tr, gt[i])):
                d = tree.query(q @ pose[:3, :3].T + pose[:3, 3], k=1, workers=WORKERS,
                               distance_upper_bound=60.0)[0]
                dst.append(np.where(np.isfinite(d), d, 60.0))
            if (n + 1) % 100 == 0:
                print('    %d/%d  %.0f s' % (n + 1, len(indices), time.time() - t0_clock),
                      flush=True)
        return fr, tr

    # ---------------- Part A
    band = np.where((e0 >= 5.0) & (e0 <= 8.0))[0]
    stuck = band[np.abs(s_val[band]) < 1e-9]
    print('band %d frames, of which |s|<1e-9 in the online run: %d'
          % (len(band), len(stuck)))
    fr, tr = raw_dists(band)
    for tag, sel in (('all 43', np.arange(len(band))),
                     ('25 stuck', np.array([j for j, i in enumerate(band) if i in set(stuck)]))):
        cf = np.array([np.minimum(fr[j], 1.0).mean() for j in sel])
        ct = np.array([np.minimum(tr[j], 1.0).mean() for j in sel])
        print('  %-9s C(front) %.3f  C(truth) %.3f  gap %+.3f  truth wins %d/%d'
              % (tag, cf.mean(), ct.mean(), (ct - cf).mean(), int((ct < cf).sum()), len(sel)))

    # ---------------- Part B
    rng = np.random.RandomState(20261022)
    cand = np.where(e0 <= 1.0)[0]
    keep = rng.permutation(cand)[:N_CONTROL]
    keep.sort()
    dates = np.array([keys[i].split('/')[0] for i in keep])
    print('\ncontrol sample: %d ordinary frames (front-end error <= 1 m), %d dates, '
          'front-end error median %.3f m, p95 %.3f m'
          % (len(keep), len(set(dates)), np.median(e0[keep]), np.percentile(e0[keep], 95)))
    cfr, ctr = raw_dists(keep)
    print('\n%-12s %12s %12s %10s %10s' % ('cost setting', 'front wins', 'truth wins',
                                           'mean gap', 'med gap'))
    n = len(keep)
    cf_all = {}
    for name in ('trunc_1.0', 'gm_1.0', 'huber_1.0'):
        fn = K[name]
        cf = np.array([fn(cfr[j]).mean() for j in range(n)])
        ct = np.array([fn(ctr[j]).mean() for j in range(n)])
        gap = ct - cf
        cf_all[name] = (cf, ct)
        print('%-12s %12d %12d %10.4f %10.4f'
              % (name, int((gap > 0).sum()), int((gap < 0).sum()), gap.mean(),
                 np.median(gap)))
    sat = np.array([(cfr[j] >= 1.0).mean() for j in range(n)])
    print('  residual saturation at 1 m on these frames: median %.3f' % np.median(sat))
    same = ((cf_all['trunc_1.0'][1] - cf_all['trunc_1.0'][0]) > 0) == \
           ((cf_all['gm_1.0'][1] - cf_all['gm_1.0'][0]) > 0)
    print('  the two settings pick the same pose on %d/%d frames' % (int(same.sum()), n))

    # ---------------- Part C
    print('\nper date (gm_1.0): date, n, front wins, median gap')
    for d in sorted(set(dates)):
        m = dates == d
        gap = cf_all['gm_1.0'][1][m] - cf_all['gm_1.0'][0][m]
        print('  %s %5d %6d %10.4f' % (d, int(m.sum()), int((gap > 0).sum()), np.median(gap)))
    print('\ntotal %.1f s' % (time.time() - t0_clock))


if __name__ == '__main__':
    main()
