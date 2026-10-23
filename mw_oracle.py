"""The oracle-window upper bound: exactly one window per frame, centred at the truth.

This is the only runner that reads ground truth, and it reads it for one purpose: to
choose the single candidate window the expert gets.  The window centre is the library
reference closest to the true camera position -- i.e. a real road reference, the best
this candidate source could have offered -- and the search, the map assembly, the
evidence and the switch test are the same code the multi-window run uses.

It answers one question and nothing else: if the window selection were perfect, how far
could this geometry path go?

    python mw_oracle.py --shard 0/12 --out /tmp/mw/oracle
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

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH = DATA + '/evaluator-only-truth/%s.json'

ap = argparse.ArgumentParser()
ap.add_argument('--shard', default='0/1')
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--out', default='/tmp/mw/oracle')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()

subset = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in subset['keys']]
t0 = np.asarray(subset['t0'], float)
truth = {d: json.load(open(TRUTH % d)) for d in sorted({k.split('/')[0] for k in keys})}
gt = np.stack([np.asarray(truth[k.split('/')[0]][k.split('/')[1]]['T_WB'], float) for k in keys])

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
path = os.path.join(a.out, 'oracle_s%d.jsonl' % si)

started = time.time()
done = 0
with open(path, 'w') as fh:
    for i in index:
        key = keys[i]
        date, stamp = key.split('/')
        record = dict(key=key,
                      scan_path=os.path.join(DATA, 'NCLT', date, 'velodyne_sync', stamp + '.bin'))
        T0 = t0[i]
        row = dict(key=key, index=int(i))
        t_frame = time.time()
        try:
            distance, j = pos_tree.query(gt[i][:3, 3], k=1)
            centre = refpos[int(j)].copy()
            windows = [dict(ref=int(j), channel='oracle', also_cls=False, rank=0,
                            centre=centre,
                            distance_to_t0=float(np.linalg.norm(centre - T0[:3, 3])))]
            pose, info = mw_geometry.run_frame(record, T0, maps, cfg, windows,
                                               multi_window=True)
            row.update(pose=np.asarray(pose, float).ravel().tolist(),
                       oracle_ref=int(j), oracle_centre=centre.tolist(),
                       oracle_centre_err=float(distance),
                       oracle_pose=info['windows'][0]['pose'] if info['windows'] else None,
                       info=info)
        except BaseException as exc:
            row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        row['seconds'] = round(time.time() - t_frame, 3)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        done += 1
        if done % 10 == 0:
            print('shard %d %d/%d %.0fs' % (si, done, len(index), time.time() - started), flush=True)
print('shard %d done frames=%d wall=%.0fs -> %s' % (si, done, time.time() - started, path), flush=True)
