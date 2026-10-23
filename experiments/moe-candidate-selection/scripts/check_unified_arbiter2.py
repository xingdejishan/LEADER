"""Read-only follow-up: the same-cost test was unfair to the visual hypothesis.

The visual pose can be tens of metres away from route 3, so scoring it on a map cropped
around route 3 is hostile. Here each hypothesis is scored on a map cropped around its own
position (deployment semantics), and additionally on a neutral map cropped at the midpoint.
Nothing is written.
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
TRUNC = 1.0
NPTS = 4000

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
e3 = np.array([np.linalg.norm(r3[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
em = np.array([np.linalg.norm(merged[k][:3, 3] - gt[i][:3, 3]) for i, k in enumerate(hard_keys)])
ratio = np.array([d['ratio'] for d in diag])
severe = np.where(np.isfinite(ratio) & (ratio < 0.21))[0]

vis_pose, vis_ok = {}, {}
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


def score_at(T, q, centre):
    mp = maps.local(centre, maps.nearby(centre)[:cfg.reference_count])
    return cost(cKDTree(mp), q @ T[:3, :3].T + T[:3, 3])


rows = []
for i in severe:
    k = hard_keys[i]
    Tv = vis_pose.get(k)
    if Tv is None:
        continue
    T3 = r3[k]
    scan = P.prepare_scan(frames[i], cfg)
    q = scan[:min(NPTS, len(scan))]
    c3_own = score_at(T3, q, T3[:3, 3])
    cv_own = score_at(Tv, q, Tv[:3, 3])
    mid = 0.5 * (T3[:3, 3] + Tv[:3, 3])
    tree_mid = cKDTree(maps.local(mid, maps.nearby(mid)[:cfg.reference_count]))
    c3_mid = cost(tree_mid, q @ T3[:3, :3].T + T3[:3, 3])
    cv_mid = cost(tree_mid, q @ Tv[:3, :3].T + Tv[:3, 3])
    rows.append((i, k, c3_own, cv_own, c3_mid, cv_mid, e3[i],
                 float(np.linalg.norm(Tv[:3, 3] - T3[:3, 3]))))

print('severe frames evaluated: %d' % len(rows))
d_pose = np.array([r[7] for r in rows])
print('pose separation route3 vs visual: median %.1f m, >10m in %d/%d frames'
      % (np.median(d_pose), int((d_pose > 10).sum()), len(rows)))

ev_arr = np.array([min(1e9, np.linalg.norm(vis_pose[r[1]][:3, 3] - gt[r[0]][:3, 3]))
                   for r in rows])
err3 = np.array([r[6] for r in rows])
vis_better = ev_arr < err3


def report(tag, c3a, cva):
    pred = cva < c3a
    tp = int((vis_better & pred).sum()); fp = int((~vis_better & pred).sum())
    fn = int((vis_better & ~pred).sum()); tn = int((~vis_better & ~pred).sum())
    print('\n[%s] TP=%d FP=%d FN=%d TN=%d' % (tag, tp, fp, fn, tn))
    for delta in (0.0, 0.01, 0.02, 0.05):
        acc = np.zeros(n, bool)
        for r, a, b in zip(rows, c3a, cva):
            acc[r[0]] = bool(b < a - delta)
        out = np.where(acc, np.nan_to_num(
            np.array([np.linalg.norm(vis_pose[k][:3, 3] - gt[i][:3, 3])
                      if k in vis_pose else 1e9 for i, k in enumerate(hard_keys)]), nan=1e9), e3)
        print('   delta=%.3f  accepts=%2d  MPE=%.4f  >10m=%d'
              % (delta, int(acc.sum()), out.mean(), int((out > 10).sum())))


report('each pose on its own map', [r[2] for r in rows], [r[3] for r in rows])
report('both on a midpoint map', [r[4] for r in rows], [r[5] for r in rows])

print('\nreference points:')
print('  frozen rule        MPE=%.4f accepts=%d' % (em.mean(), int(np.array(
    [d['visual'] for d in diag]).sum())))
print('  route3 only        MPE=%.4f' % e3.mean())
orc = e3.copy()
for r in rows:
    orc[r[0]] = min(e3[r[0]], ev_arr[list(rows).index(r)])
print('  oracle severe set  MPE=%.4f' % orc.mean())

# where do the catastrophic frames sit? (visual fixes them, does cost see it?)
print('\ncatastrophic severe frames (route3 err > 5 m):')
for r in rows:
    if r[6] > 5:
        print('  err3=%7.2f  err_vis=%7.2f  dpose=%5.1f  c3own=%.3f cvown=%.3f  c3mid=%.3f cvmid=%.3f'
              % (r[6], np.linalg.norm(vis_pose[r[1]][:3, 3] - gt[r[0]][:3, 3]), r[7],
                 r[2], r[3], r[4], r[5]))
