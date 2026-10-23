"""The finished version in its own operating mode: the deployed arbitration, not forced adoption.

The +1414.74 m figure of the finished version is what the visual expert would save if every
pose it produced were adopted unconditionally (band5-nopose-report.md section 0).  The
deployed pipeline does not do that: it scores the geometry incumbent and the visual
hypothesis on one shared evidence and switches only when the incumbent is worse by
k * sigma_C.  This script reproduces exactly that rule on the same 290 frames, using

    geometry incumbent   the deployed single-window search (pipeline.localize from T0)
    visual hypothesis    the C5 pose the expert actually produced (results/band5-nopose-perframe.npz)
    evidence             union of the two crops
    test                 arbitration.decide(k = 8.4), unchanged

so the multi-window result has a like-for-like baseline as well as the forced-adoption one.

    python mw_visual_arb.py --shard 0/12 --out /tmp/mw/visual
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
import arbitration as ARB     # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'

ap = argparse.ArgumentParser()
ap.add_argument('--shard', default='0/1')
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--perframe', default=os.path.join(HERE, 'results/band5-nopose-perframe.npz'))
ap.add_argument('--out', default='/tmp/mw/visual')
ap.add_argument('--k', type=float, default=8.4)
ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()

sub = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in sub['keys']]
t0 = np.asarray(sub['t0'], float)
perf = np.load(a.perframe, allow_pickle=True)
assert [str(k) for k in perf['keys']] == keys
vis_pose = np.asarray(perf['C5_T_adopted'], float)
vis_ok = np.asarray(perf['C5_has_pose'], bool)

refs = json.load(open(os.path.join(HERE, 'inputs/refs.json')))
refpos = np.asarray([r['world_from_body'] for r in refs], float)[:, :3, 3]
maps = F.MapSource(refs, refpos, radius=60.0, max_refs=256, max_maps=64)
cfg = P.Config(workers=int(a.threads))

si, sn = (int(x) for x in a.shard.split('/'))
index = list(range(len(keys)))[si::sn]
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, 'visual_s%d.jsonl' % si)
started = time.time()
with open(path, 'w') as fh:
    for i in index:
        key = keys[i]
        record = dict(key=key, scan_path=os.path.join(DATA, 'NCLT', key.split('/')[0],
                                                      'velodyne_sync', key.split('/')[1] + '.bin'))
        T0 = t0[i]
        row = dict(key=key, index=int(i))
        t_frame = time.time()
        try:
            scan = P.prepare_scan(record, cfg)
            T_inc, d_inc = P.localize(record, T0, maps, cfg, scan=scan)
            info = dict(incumbent_cost_own=float(d_inc['cost']))
            chosen = T_inc
            if vis_ok[i]:
                centre_v = vis_pose[i][:3, 3]
                triple_v = maps.nearby(centre_v)[:cfg.reference_count]
                mp_v = maps.local(centre_v, triple_v) if len(triple_v) else None
                mp0 = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
                parts = [mp0] if len(mp0) else []
                if mp_v is not None and len(mp_v):
                    parts.append(mp_v)
                evidence = ARB.Evidence(ARB.union_maps(*parts), scan,
                                        trunc=1.0, n_points=4000)
                hyps = [ARB.Hypothesis(T_inc, 'geo_t0', {'incumbent': True}),
                        ARB.Hypothesis(vis_pose[i], 'visual')]
                picked, arb = ARB.decide(hyps, evidence, a.k, scan=scan)
                chosen = picked.pose
                info.update(switched=bool(arb['switched']), costs=arb['costs'],
                            sigma_c=float(arb['sigma_c']), thresh=float(arb['thresh']),
                            margin=float(arb['margin']), chosen=arb['chosen'],
                            n_map=int(evidence.n_map))
            else:
                info.update(switched=False, chosen='geo_t0')
            row.update(pose=np.asarray(chosen, float).ravel().tolist(),
                       incumbent=np.asarray(T_inc, float).ravel().tolist(), info=info)
        except BaseException as exc:
            row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        row['seconds'] = round(time.time() - t_frame, 3)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
print('shard %d done frames=%d -> %s' % (si, len(index), path), flush=True)
