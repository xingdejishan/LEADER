"""Read-only: clean numbers for a single-objective (same-cost) arbitration rule.

Rule: score each hypothesis on a local map cropped around its own position, with the
same truncated cost the search minimises; accept the visual hypothesis only if it is
cheaper by a margin delta (a switching penalty expressed in cost units).

Reports MPE, MOE, confusion against "which hypothesis is actually better", the captured
share of the oracle gain, and the sensitivity to delta. Writes nothing.
"""
import importlib.util
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
TRUNC, NPTS = 1.0, 4000

spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
hard_keys, gt = w['hard_keys'], w['hard_gt']

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
e3 = np.array([np.linalg.norm(r3[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
r3rot = np.array([su.rot_deg(r3[k][None, :3, :3], gt[i][None, :3, :3])[0]
                  for i, k in enumerate(hard_keys)])
em = np.array([np.linalg.norm(merged[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
rm = np.array([su.rot_deg(merged[k][None, :3, :3], gt[i][None, :3, :3])[0]
               for i, k in enumerate(hard_keys)])
ratio = np.array([d['ratio'] for d in diag])
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


ev = np.full(n, np.nan)
rv = np.full(n, np.nan)
dcost = np.full(n, np.nan)
for i in severe:
    k = hard_keys[i]
    Tv = vis_pose.get(k)
    if Tv is None:
        continue
    T3 = r3[k]
    q = P.prepare_scan(frames[i], cfg)[:NPTS]
    c3 = cost(cKDTree(maps.local(T3[:3, 3], maps.nearby(T3[:3, 3])[:cfg.reference_count])),
              q @ T3[:3, :3].T + T3[:3, 3])
    cv = cost(cKDTree(maps.local(Tv[:3, 3], maps.nearby(Tv[:3, 3])[:cfg.reference_count])),
              q @ Tv[:3, :3].T + Tv[:3, 3])
    dcost[i] = cv - c3
    ev[i] = np.linalg.norm(Tv[:3, 3] - gt[i][:3, 3])
    rv[i] = su.rot_deg(Tv[None, :3, :3], gt[i][None, :3, :3])[0]

m = np.isfinite(dcost)
better = m & (ev < e3)
print('severe=%d  evaluated=%d  visual-better=%d' % (len(severe), int(m.sum()), int(better.sum())))
pred = m & (dcost < 0)
print('sign of cost difference vs truth: TP=%d FP=%d FN=%d TN=%d'
      % (int((better & pred).sum()), int((~better & pred).sum()),
         int((better & ~pred).sum()), int((~better & ~pred).sum())))
print('dcost on visual-better: median %.4f | on visual-worse: median %.4f'
      % (np.median(dcost[better]), np.median(dcost[m & ~better])))

orc_p = e3.copy(); orc_r = r3rot.copy()
orc_p[m] = np.minimum(e3[m], ev[m])
orc_r[m] = np.where(ev[m] < e3[m], rv[m], r3rot[m])
print('\nreference: route3  MPE=%.4f MOE=%.4f ok=%d'
      % (e3.mean(), r3rot.mean(), int(((e3 <= 1) & (r3rot <= 2)).sum())))
print('           frozen  MPE=%.4f MOE=%.4f ok=%d'
      % (em.mean(), rm.mean(), int(((em <= 1) & (rm <= 2)).sum())))
print('           oracle  MPE=%.4f MOE=%.4f ok=%d'
      % (orc_p.mean(), orc_r.mean(), int(((orc_p <= 1) & (orc_r <= 2)).sum())))

print('\naccept if visual is cheaper by delta (cost units = metres of mean residual):')
for delta in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
    acc = m & (dcost < -delta)
    P_out = np.where(acc, np.nan_to_num(ev, nan=1e9), e3)
    R_out = np.where(acc, np.nan_to_num(rv, nan=1e9), r3rot)
    gain = (e3.mean() - P_out.mean()) / (e3.mean() - orc_p.mean()) * 100
    print('  delta=%.2f  accepts=%2d  MPE=%.4f  MOE=%.4f  ok(1m,2deg)=%d  >10m=%d  '
          'captured=%.0f%%'
          % (delta, int(acc.sum()), P_out.mean(), R_out.mean(),
             int(((P_out <= 1) & (R_out <= 2)).sum()), int((P_out > 10).sum()), gain))
