"""Consistency check: the new entry point with the switch off, against the existing one.

Runs the *existing* per-frame function (`three_route_unified.one` in mode "route3", the
deployed single-window geometry path) on the 290 frames of the study, with the frame list
and the initial poses replaced by this study's own (`results/band5-nopose-subset.npz`).
Nothing else about that module is touched: its `_G` and `_OPT` dicts are the knobs it
already exposes, and `one()` is called as is.

The comparison is done by `mw_report.py` against `mw_run.py --multi-window 0`.

    python mw_check_single.py --out /tmp/mw/check
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))

import three_route_unified as T  # noqa: E402  (existing orchestrator, unmodified)

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'

ap = argparse.ArgumentParser()
ap.add_argument('--shard', default='0/1')
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--out', default='/tmp/mw/check')
a = ap.parse_args()

sub = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in sub['keys']]
t0 = np.asarray(sub['t0'], float)

# the knobs the module already reads: the deployed route3 configuration
T._OPT.update(mode='route3', gate='merged', gate2=-1.0, solver='adaptive',
              coarse_points=1000, fine_points=1500, threads=1, experts='parallel',
              visual_source='none', k=8.4)
T.init()
T._G['frames'] = [dict(key=k,
                       scan_path=os.path.join(DATA, 'NCLT', k.split('/')[0],
                                              'velodyne_sync', k.split('/')[1] + '.bin'))
                  for k in keys]
T._G['poses'] = t0

si, sn = (int(x) for x in a.shard.split('/'))
index = list(range(len(keys)))[si::sn]
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, 'check_s%d.jsonl' % si)
started = time.time()
with open(path, 'w') as fh:
    for i in index:
        t_frame = time.time()
        row = dict(key=keys[i], index=int(i))
        try:
            _, pose, diag = T.one(i)
            row.update(pose=np.asarray(pose, float).ravel().tolist(),
                       ran_fine=bool(diag.get('ran_fine')),
                       s=diag.get('s'), d=diag.get('d'), gain=diag.get('gain'))
        except BaseException as exc:
            row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        row['seconds'] = round(time.time() - t_frame, 3)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
print('shard %d done frames=%d wall=%.0fs -> %s' % (si, len(index), time.time() - started, path),
      flush=True)
