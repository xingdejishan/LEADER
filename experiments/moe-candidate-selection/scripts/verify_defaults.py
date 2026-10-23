"""Compare the 200-frame run under the new defaults with the assembled best table."""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
z = np.load(P + 'full-online-final.npz', allow_pickle=True)
ref = {str(k): np.asarray(p, float) for k, p in zip(z['keys'], z['poses'])}
t0ref = {str(k): np.asarray(p, float) for k, p in zip(z['keys'], z['t0'])}

rows = [json.loads(l) for l in open(P + 'verify-search-shard0.jsonl') if l.strip()]
rows = [r for r in rows if r.get('seconds')]
print('frames run: %d' % len(rows))
d, d0, n_rot, n_fine = [], [], 0, 0
for r in rows:
    k = r['key']
    if k not in ref:
        continue
    P_new = np.asarray(r['pose'], float).reshape(4, 4)
    d.append(np.abs(P_new - ref[k]).max())
    d0.append(np.abs(np.asarray(r['t0'], float).reshape(4, 4) - t0ref[k]).max())
    n_rot += int(bool(r.get('rotated')))
    n_fine += int(bool(r.get('ran_fine')))
d = np.array(d)
print('pose difference vs assembled table: max %.3e  median %.3e  frames >1e-6: %d'
      % (d.max(), np.median(d), int((d > 1e-6).sum())))
print('front-end t0 difference: max %.3e (should be 0)' % np.max(d0))
print('search ran on %d frames, rotation applied on %d' % (n_fine, n_rot))
print('frame totals: %.2f s median' % np.median([r['total'] for r in rows]))


