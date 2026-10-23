"""Read-only: can the arbitration use the SAME truncated cost as everything else?

Currently the visual branch is accepted on two foreign scalars (ambiguity ratio + a
marginal geometry score from a frozen external verifier). Here we test the unified
rule: accept the visual hypothesis iff it has a lower truncated scan-to-map cost than
route 3, evaluated on the same local map with the same cost function the search uses.

Nothing is written; all numbers are printed.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402

SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
TRUNC = 1.0
NPTS = 4000

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
hard_keys = w['hard_keys']
gt = w['hard_gt']

frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
pos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
cfg = P.Config()
maps = F.MapSource(refs, pos, radius=cfg.crop_radius)

z3 = np.load(os.path.join(ROOT, 'results/final.npz'), allow_pickle=True)
r3 = {str(k): np.asarray(p, float) for k, p in zip(z3['keys'], z3['poses'])}
zm = np.load(os.path.join(ROOT, 'results/three_route.npz'), allow_pickle=True)
merged = {str(k): np.asarray(p, float) for k, p in zip(zm['keys'], zm['poses'])}
diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_diag.jsonl'))]

n = len(hard_keys)
key2i = {k: i for i, k in enumerate(hard_keys)}
e3 = np.array([np.linalg.norm(r3[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
em = np.array([np.linalg.norm(merged[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
ratio = np.array([d['ratio'] for d in diag])
vis_frozen = np.array([d['visual'] for d in diag])
severe = np.where(np.isfinite(ratio) & (ratio < 0.21))[0]

vis_pose = {}
for i in range(n):
    rec = json.load(open(VERIFY % i))['poses']
    best = None
    for name, row in rec.items():
        if not name.startswith('candidate_'):
            continue
        sc = (row.get('fair') or {}).get('geometry_score')
        if sc is None:
            continue
        if best is None or sc > best[0]:
            best = (float(sc), np.asarray(row['pose'], float))
    if best is not None:
        vis_pose[hard_keys[i]] = best[1]


def cost(tree, pts):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=TRUNC)[0]
    return float(np.where(np.isfinite(d), d, TRUNC).mean())


rows = []
for i in severe:
    k = hard_keys[i]
    T3 = r3[k]
    Tv = vis_pose.get(k)
    if Tv is None:
        continue
    mp = maps.local(T3[:3, 3], maps.nearby(T3[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(frames[i], cfg)
    q = scan[:min(NPTS, len(scan))]
    tree = cKDTree(mp)
    c3 = cost(tree, q @ T3[:3, :3].T + T3[:3, 3])
    cv = cost(tree, q @ Tv[:3, :3].T + Tv[:3, 3])
    ev = float(np.linalg.norm(Tv[:3, 3] - gt[i][:3, 3]))
    rows.append((i, k, c3, cv, e3[i], ev))

print('severe frames evaluated: %d' % len(rows))
c3 = np.array([r[2] for r in rows]); cv = np.array([r[3] for r in rows])
err3 = np.array([r[4] for r in rows]); errv = np.array([r[5] for r in rows])
vis_better = errv < err3
pred_better = cv < c3
tp = int((vis_better & pred_better).sum()); fp = int((~vis_better & pred_better).sum())
fn = int((vis_better & ~pred_better).sum()); tn = int((~vis_better & ~pred_better).sum())
print('same-cost rule vs truth (visual better?): TP=%d FP=%d FN=%d TN=%d' % (tp, fp, fn, tn))
print('  cost(route3) median %.4f   cost(visual) median %.4f' % (np.median(c3), np.median(cv)))
print('  visual-better subset: dcost median %.4f' % np.median((cv - c3)[vis_better]))
print('  visual-worse  subset: dcost median %.4f' % np.median((cv - c3)[~vis_better]))

base = e3.copy()
print('\nMPE over all 1502 frames (route 3 elsewhere), by acceptance rule on the severe set:')
print('  frozen rule (ratio+score margin) : %.4f   accepts=%d  >10m=%d'
      % (em.mean(), int(vis_frozen.sum()), int((em > 10).sum())))
print('  route 3 only                     : %.4f   accepts=0   >10m=%d'
      % (base.mean(), int((base > 10).sum())))
ev_arr = np.full(n, np.nan)
for i, k in enumerate(hard_keys):
    if k in vis_pose:
        ev_arr[i] = np.linalg.norm(vis_pose[k][:3, 3] - gt[i][:3, 3])

for delta in (-0.01, 0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
    acc = np.zeros(n, bool)
    for (i, k, a, b, _e1, _e2) in rows:
        acc[i] = bool(b < a - delta)
    out = np.where(acc, np.nan_to_num(ev_arr, nan=1e9), e3)
    print('  cost rule, delta=%+.3f            : %.4f   accepts=%d  >10m=%d'
          % (delta, out.mean(), int(acc.sum()), int((out > 10).sum())))

orc = e3.copy()
for (i, k, _a, _b, _e1, e2) in rows:
    orc[i] = min(e3[i], e2)
print('  oracle on severe set             : %.4f   >10m=%d'
      % (orc.mean(), int((orc > 10).sum())))
