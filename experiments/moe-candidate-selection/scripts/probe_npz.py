import os, sys, json
import numpy as np
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as S

w = S.load_world()
gt = w['hard_gt']
hk = w['hard_keys']
RES = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'

def load(p):
    z = np.load(p, allow_pickle=True)
    arr = np.asarray(z['poses'], float)
    if 'keys' in z.files:
        lk = {str(k): i for i, k in enumerate(z['keys'])}
        assert all(k in lk for k in hk)
        arr = arr[[lk[k] for k in hk]]
    return arr

for n in ['three_route.npz', 'three_route_merged.npz', 'three_route_unified.npz',
          'verify_route3.npz', 'verify_frozen.npz', 'pipeline.npz', 'final.npz']:
    p = os.path.join(RES, n)
    if not os.path.exists(p):
        print(n, 'MISSING'); continue
    ph = load(p)
    t = S.pos_err(ph, gt); r = S.rot_deg(ph[:, :3, :3], gt[:, :3, :3])
    print('%-24s MPE=%.4f MOE=%.4f ok=%d >5m=%d >10m=%d'
          % (n, t.mean(), r.mean(), int(((t <= 1) & (r <= 2)).sum()),
             int((t > 5).sum()), int((t > 10).sum())))

# how many frames does merged differ from verify_route3
a = load(os.path.join(RES, 'verify_route3.npz')); b = load(os.path.join(RES, 'three_route_merged.npz'))
d = np.abs(a - b).max(axis=(1, 2))
print('merged vs verify_route3 differ frames:', int((d > 1e-9).sum()))
d2 = [json.loads(l) for l in open(os.path.join(RES, 'three_route_merged_diag.jsonl'))]
print('merged diag n=%d visual=%d cols=%s' % (len(d2), sum(1 for x in d2 if x.get('visual')), sorted(d2[0].keys())))
