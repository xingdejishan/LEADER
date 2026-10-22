"""Frame convention check: which world axis is up?

Prints, for a few frames, the percentile spread of the world-transformed scan
along x/y/z and the largest 3-D plane in that cloud with no orientation hint.
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

rng = np.random.default_rng(3)
for i in rng.choice(len(keys), 4, replace=False):
    key = keys[i]
    date, ts = key.split('/')
    rec = {'key': key, 'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
    raw = F.decode(rec['scan_path'])
    scan = P.prepare_scan(rec, cfg)
    gt = truth[(date, str(ts))]
    print('\n=== %s' % key)
    print('   raw pts %d  |x| p99 %.1f  radius p99 %.1f' %
          (len(raw), np.percentile(np.abs(raw[:, 0]), 99), np.percentile(np.linalg.norm(raw, axis=1), 99)))
    print('   truth T_WB translation %s' % np.round(gt[:3, 3], 3))
    print('   truth R:\n%s' % np.round(gt[:3, :3], 3))
    sw = scan @ gt[:3, :3].T + gt[:3, 3]
    for c, name in enumerate('xyz'):
        q = np.percentile(sw[:, c], [1, 5, 50, 95, 99])
        print('   world %s pct[1,5,50,95,99] %s  std %.2f' % (name, np.round(q, 2), sw[:, c].std()))
    # largest plane, no hint
    n = len(sw)
    idx = rng.integers(0, n, size=(800, 3))
    A, B, C = sw[idx[:, 0]], sw[idx[:, 1]], sw[idx[:, 2]]
    nrm = np.cross(B - A, C - A)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-6
    nrm, A = nrm[ok] / ln[ok, None], A[ok]
    d = -(nrm * A).sum(1)
    res = np.abs(sw @ nrm.T + d[None, :])
    cnt = (res < 0.15).sum(0)
    b = int(np.argmax(cnt))
    print('   largest plane: normal %s  inliers %d/%d  (|nz| %.2f)'
          % (np.round(nrm[b], 3), int(cnt[b]), n, abs(nrm[b][2])))
