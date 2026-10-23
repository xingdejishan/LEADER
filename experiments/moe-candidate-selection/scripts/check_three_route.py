"""Read-only re-scoring of the three-route output with the frozen unified scorer.

Reads only: results/three_route.npz, results/final.npz, results/three_route_diag.jsonl
Writes nothing.
"""
import importlib.util
import json
import os

import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'

spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)

w = su.load_world()
hard_keys = w['hard_keys']
gt = w['hard_gt']
idx = {k: i for i, k in enumerate(hard_keys)}


def score(path, tag):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    P = np.asarray(z['poses'], float)
    order = [keys.index(k) for k in hard_keys]
    A = P[order]
    e = su.pos_err(A, gt)
    r = su.rot_deg(A[:, :3, :3], gt[:, :3, :3])
    print('%-14s n=%d  MPE=%.4f  MOE=%.4f  ok(1m,2deg)=%d  >10m=%d  median=%.3f'
          % (tag, len(e), e.mean(), r.mean(), int(((e <= 1) & (r <= 2)).sum()),
             int((e > 10).sum()), float(np.median(e))))
    return A, e, r


print('checking keys/order ...', 'aligned' if len(hard_keys) == 1502 else 'UNEXPECTED')
A3, e3, r3 = score(os.path.join(ROOT, 'results/three_route.npz'), 'three_route')
A1, e1, r1 = score(os.path.join(ROOT, 'results/final.npz'), 'route3 only')

diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_diag.jsonl'))]
print('diag frames=%d  visual=%d  fine=%d' % (
    len(diag), sum(d['visual'] for d in diag), sum(d['ran_fine'] for d in diag)))
severe = [d for d in diag if np.isfinite(d['ratio']) and d['ratio'] < 0.21]
print('severe(ratio<0.21)=%d  visual_chosen_inside_severe=%d'
      % (len(severe), sum(d['visual'] for d in severe)))

# per-frame effect of taking the visual pose instead of route 3
vis = np.array([d['visual'] for d in diag])
if vis.any():
    print('on the %d visual-accepted frames: MPE route3=%.3f -> merged=%.3f'
          % (int(vis.sum()), float(e1[vis].mean()), float(e3[vis].mean())))
    print('  of those, route3 error >1m: %d ; merged error >1m: %d'
          % (int((e1[vis] > 1).sum()), int((e3[vis] > 1).sum())))
    print('  frames where merging HURT by >0.5 m: %d'
          % int(((e3[vis] - e1[vis]) > 0.5).sum()))
