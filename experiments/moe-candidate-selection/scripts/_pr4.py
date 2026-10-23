import json
import os
import sys
import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as S

w = S.load_world()
canon = w['hard_keys']
for name in ('three_route.npz', 'three_route_unified.npz', 'three_route_merged.npz',
             'final.npz'):
    p = os.path.join(ROOT, 'results', name)
    z = np.load(p, allow_pickle=True)
    look = {str(k): i for i, k in enumerate(z['keys'])}
    ph = np.asarray(z['poses'], np.float64)[[look[k] for k in canon]]
    t = S.pos_err(ph, w['hard_gt'])
    r = S.rot_deg(ph[:, :3, :3], w['hard_gt'][:, :3, :3])
    print('%-24s MPE=%.4f MOE=%.4f ok=%d >10m=%d >5m=%d'
          % (name, t.mean(), r.mean(), int(np.sum((t <= 1.0) & (r <= 2.0))),
             int(np.sum(t > 10.0)), int(np.sum(t > 5.0))))

rows = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
for i in range(5):
    print(i, rows[i])
print('visual frames:', [i for i, r in enumerate(rows) if r.get('visual')])
print('nan ratio:', sum(1 for r in rows if not np.isfinite(r.get('ratio', float('nan')))))
