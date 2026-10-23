#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Why the alternative kernels do not help: raw (uncapped) distances and rotation.

Three poses per frame instead of two:
   F  (R0, t0)        front-end pose
   G  (Rg, tg)        truth pose
   FG (R0, tg)        truth translation, front-end rotation  <- what the branch could reach
plus the uncapped mean nearest-neighbour distance, which is the limit of the soft
kernels (Huber grows linearly, so it is a weighted raw mean).

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
K = {'trunc_1.0': lambda d: np.minimum(d, 1.0),
     'gm_1.0': lambda d: d * d / (d * d + 1.0),
     'huber_1.0': lambda d: np.where(d <= 1.0, 0.5 * d * d, d - 0.5)}


def main():
    tick = time.time()
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

    D = {'F': [], 'G': [], 'FG': []}
    for n, i in enumerate(band):
        date, ts = keys[i].split('/')
        rec = {'key': keys[i],
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        q = F.spread_subset(scan, 4000)
        centre = t0[i][:3, 3]
        tree = cKDTree(maps.local(centre, maps.nearby(centre)[:3]))
        for tag, R, t in (('F', t0[i][:3, :3], t0[i][:3, 3]),
                          ('G', gt[i][:3, :3], gt[i][:3, 3]),
                          ('FG', t0[i][:3, :3], gt[i][:3, 3])):
            d = tree.query(q @ R.T + t, k=1, workers=WORKERS, distance_upper_bound=60.0)[0]
            D[tag].append(np.where(np.isfinite(d), d, 60.0))
        if (n + 1) % 15 == 0:
            print('  %d/%d %.0f s' % (n + 1, len(band), time.time() - tick), flush=True)

    u = t0[band, :3, 0] / np.linalg.norm(t0[band, :3, 0], axis=1, keepdims=True)
    lat = np.cross(u, np.array([0.0, 0.0, 1.0]))
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    delta = gt[band, :3, 3] - t0[band, :3, 3]
    reach = (np.abs((delta * u).sum(1)) <= 8.0) & (np.abs((delta * lat).sum(1)) <= 2.0)
    rot = np.array([np.degrees(np.arccos(np.clip(
        (np.trace(t0[i][:3, :3].T @ gt[i][:3, :3]) - 1) / 2, -1, 1))) for i in band])
    stuck = np.array([abs(rows[keys[i]]['s']) < 1e-9 for i in band])

    def arr(tag, fn=lambda d: d):
        return np.array([fn(D[tag][j]).mean() for j in range(len(band))])

    print('\n%-22s %8s %8s %8s %8s' % ('statistic', 'F', 'G', 'FG', 'G-F'))
    for label, fn in (('uncapped mean d (m)', lambda d: d),
                      ('trunc_1.0 cost', K['trunc_1.0']),
                      ('gm_1.0 cost', K['gm_1.0']),
                      ('huber_1.0 cost', K['huber_1.0'])):
        a, b, c = arr('F', fn), arr('G', fn), arr('FG', fn)
        print('%-22s %8.3f %8.3f %8.3f %+8.3f' % (label, a.mean(), b.mean(), c.mean(),
                                                  (b - a).mean()))
    a, b, c = arr('F'), arr('G'), arr('FG')
    for tag, sel in (('all 43', np.ones(len(band), bool)),
                     ('reachable 33', reach),
                     ('25 stuck', stuck),
                     ('not stuck 18', ~stuck)):
        print('  %-13s n=%2d  truth-vs-front wins %2d  (G<F)   FG<F %2d   FG<G %2d'
              % (tag, int(sel.sum()), int((b[sel] < a[sel]).sum()),
                 int((c[sel] < a[sel]).sum()), int((c[sel] < b[sel]).sum())))
    print('\nrotation error F vs G: median %.2f deg p90 %.2f deg;  '
          'FG cost vs F cost: better on %d/43 (uncapped mean)'
          % (np.median(rot), np.percentile(rot, 90), int((c < a).sum())))
    print('total %.0f s' % (time.time() - tick))


if __name__ == '__main__':
    main()
