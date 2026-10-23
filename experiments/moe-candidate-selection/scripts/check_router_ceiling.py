"""Read-only: what is the ceiling of a *learned* router over the same two proposals?

For every frame we know route 3's pose (final.npz) and, from the frozen visual table,
the best candidate pose plus its geometry score and the baseline score. So we can ask:
if an oracle picked route3 or visual per frame, what would MPE be? That is the best a
perfect learned gate could ever do without new hypotheses.
"""
import importlib.util
import json
import os

import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')

spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
w = su.load_world()
hard_keys = w['hard_keys']
gt = w['hard_gt']
idx = {k: i for i, k in enumerate(hard_keys)}


def load(path):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    P = np.asarray(z['poses'], float)
    return {k: P[i] for i, k in enumerate(keys)}


r3 = load(os.path.join(ROOT, 'results/final.npz'))
merged = load(os.path.join(ROOT, 'results/three_route.npz'))
diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_diag.jsonl'))]

n = len(hard_keys)
vis_pose = [None] * n
vis_score = np.full(n, np.nan)
base_score = np.full(n, np.nan)
for i in range(n):
    rec = json.load(open(VERIFY % i))['poses']
    base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
    if base is not None:
        base_score[i] = float(base)
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
        vis_pose[i] = best[1]
        vis_score[i] = best[0]

e3 = np.empty(n)
em = np.empty(n)
ev = np.full(n, np.nan)
for i, k in enumerate(hard_keys):
    e3[i] = np.linalg.norm(r3[k][:3, 3] - gt[i][:3, 3])
    em[i] = np.linalg.norm(merged[k][:3, 3] - gt[i][:3, 3])
    if vis_pose[i] is not None:
        ev[i] = np.linalg.norm(vis_pose[i][:3, 3] - gt[i][:3, 3])

ratio = np.array([d['ratio'] for d in diag])
vis_used = np.array([d['visual'] for d in diag])
severe = np.isfinite(ratio) & (ratio < 0.21)
has_vis = np.isfinite(ev)
print('frames=%d  with visual candidate=%d  severe=%d  accepted=%d'
      % (n, has_vis.sum(), severe.sum(), vis_used.sum()))

print('\n-- current --')
print('route3 only       MPE=%.4f  >10m=%d' % (e3.mean(), (e3 > 10).sum()))
print('three-route(frozen) MPE=%.4f  >10m=%d' % (em.mean(), (em > 10).sum()))

print('\n-- oracle routers over the same proposals --')
orc_all = np.where(has_vis, np.minimum(e3, np.nan_to_num(ev, nan=1e9)), e3)
print('oracle, all frames          MPE=%.4f  >10m=%d' % (orc_all.mean(), (orc_all > 10).sum()))
orc_sev = e3.copy()
m = severe & has_vis
orc_sev[m] = np.minimum(e3[m], ev[m])
print('oracle, only severe frames  MPE=%.4f  >10m=%d' % (orc_sev.mean(), (orc_sev > 10).sum()))

print('\n-- on the %d severe frames: which proposal is better? --' % int(m.sum()))
better = m & (ev < e3 - 1e-9)
worse = m & (ev > e3 + 1e-9)
print('visual better = %d, visual worse = %d, tie = %d'
      % (int(better.sum()), int(worse.sum()), int((m & ~better & ~worse).sum())))
print('median route3 err on severe = %.2f m, median visual err = %.2f m'
      % (float(np.median(e3[m])), float(np.median(ev[m]))))
print('severe frames route3 already within 1 m: %d' % int((m & (e3 <= 1)).sum()))
print('severe frames visual within 1 m:        %d' % int((m & (ev <= 1)).sum()))

# how separable are the two classes with the three scalars we already have?
print('\n-- separability of "visual is better" on severe frames (scalars) --')
feat = {'ratio': ratio, 'margin=vis-base': vis_score - base_score}
for name, f in feat.items():
    a, b = f[m & better], f[m & worse]
    if len(a) and len(b):
        print('%-16s better: med=%.4f  worse: med=%.4f' % (name, np.median(a), np.median(b)))

print('\n-- what the frozen rule costs against the oracle (severe set only) --')
print('frozen rule on severe: %d accepted; their route3 err median %.1f -> %.1f m'
      % (int((vis_used & severe).sum()),
         float(np.median(e3[vis_used & severe])), float(np.median(em[vis_used & severe]))))
