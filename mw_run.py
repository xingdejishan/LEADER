"""Run the multi-window geometry expert over the >5 m subset: one process, one shard.

Writes one JSONL row per frame: the candidate windows (reference id, channel, centre),
the incumbent pose, every window's pose and both costs, the arbitration diagnostics, and
the timing of each stage.  No ground truth is read anywhere in this file.

    python mw_run.py --multi-window 1 --shard 0/12 --out /tmp/mw/out
    python mw_run.py --multi-window 0 --shard 0/12 --out /tmp/mw/out0   # the deployed expert
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))

import fast_lib as F          # noqa: E402
import pipeline as P          # noqa: E402
import mw_geometry            # noqa: E402
from mw_windows import candidate_windows, POSITION_CHANNEL  # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'

ap = argparse.ArgumentParser()
ap.add_argument('--multi-window', type=int, default=1, choices=(0, 1),
                help='0: the deployed single window at T0 (bit-identical to it)')
ap.add_argument('--shard', default='0/1')
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--cls', default='/tmp/mw/cls/cls-ids.npz')
ap.add_argument('--out', default='/tmp/mw/out')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

subset = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in subset['keys']]
t0 = np.asarray(subset['t0'], float)
cls = np.load(a.cls, allow_pickle=True)
cls_keys = [str(k) for k in cls['keys']]
if cls_keys != keys:
    raise SystemExit('CLS table keys do not match the subset, refusing to guess the order')
cls_ids = np.asarray(cls['ids'], int)

refs = json.load(open(os.path.join(HERE, 'inputs/refs.json')))
refpos = np.asarray([r['world_from_body'] for r in refs], float)[:, :3, 3]
pos_tree = cKDTree(refpos)
maps = F.MapSource(refs, refpos, radius=60.0, max_refs=256, max_maps=64)
cfg = P.Config(workers=int(a.threads))

si, sn = (int(x) for x in a.shard.split('/'))
index = list(range(len(keys)))[si::sn]
if a.limit:
    index = index[:a.limit]
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, 'mw%d_s%d.jsonl' % (a.multi_window, si))

started = time.time()
done = 0
with open(path, 'w') as fh:
    for i in index:
        key = keys[i]
        date, stamp = key.split('/')
        record = dict(key=key,
                      scan_path=os.path.join(DATA, 'NCLT', date, 'velodyne_sync', stamp + '.bin'))
        T0 = t0[i]
        row = dict(key=key, index=int(i), multi_window=int(a.multi_window))
        t_frame = time.time()
        try:
            if a.multi_window:
                distance = np.linalg.norm(refpos - T0[:3, 3], axis=1)
                position_ids = [int(j) for j in np.argsort(distance)[:POSITION_CHANNEL]]
                windows = candidate_windows(T0[:3, 3], refpos, position_ids, cls_ids[i])
                row['position_ids'] = position_ids
                row['cls_ids'] = [int(j) for j in cls_ids[i]]
            else:
                windows = ()
            pose, info = mw_geometry.run_frame(record, T0, maps, cfg, windows,
                                               multi_window=bool(a.multi_window))
            row['pose'] = np.asarray(pose, float).ravel().tolist()
            row['info'] = info
        except BaseException as exc:
            row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        row['seconds'] = round(time.time() - t_frame, 3)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        done += 1
        if done % 10 == 0:
            print('shard %d %d/%d %.0fs' % (si, done, len(index), time.time() - started), flush=True)
print('shard %d done frames=%d wall=%.0fs -> %s' % (si, done, time.time() - started, path), flush=True)
