import os, sys, json
import numpy as np
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as S

w = S.load_world(); gt = w['hard_gt']; hk = w['hard_keys']
RES = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'
V = '/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF/%06d.json'

z = np.load('/home/zhangcy/.dsh_scratch/route3_1502.npz', allow_pickle=True)
r3 = np.asarray(z['poses'], float)
assert [str(k) for k in z['keys']] == hk
m = np.load(os.path.join(RES, 'three_route_merged.npz'), allow_pickle=True)
mg = np.asarray(m['poses'], float)
d = [json.loads(l) for l in open(os.path.join(RES, 'three_route_merged_diag.jsonl'))]
vis = np.array([bool(x.get('visual')) for x in d])
diff = np.abs(r3 - mg).max(axis=(1, 2))
print('route3 vs merged: identical on %d/%d non-visual frames; differ on %d frames (%d of them baseline-visual)'
      % (int((diff[~vis] < 1e-9).sum()), int((~vis).sum()), int((diff > 1e-9).sum()),
         int((diff[vis] > 1e-9).sum())))
t = S.pos_err(r3, gt); r = S.rot_deg(r3[:, :3, :3], gt[:, :3, :3])
print('route3 pure: MPE=%.4f MOE=%.4f ok=%d >5m=%d >10m=%d'
      % (t.mean(), r.mean(), int(((t <= 1) & (r <= 2)).sum()), int((t > 5).sum()), int((t > 10).sum())))

# usable-candidate definitions
def defs():
    n_valid = []; n_ver = []; n_sc = []; n_all = []
    for i in range(1502):
        rec = json.load(open(V % i))['poses']
        c = [(k, v) for k, v in rec.items() if k.startswith('candidate_')]
        ok_sc = [v for k, v in c if (v.get('fair') or {}).get('geometry_score') is not None]
        ok_v = [v for v in ok_sc if (v.get('fair') or {}).get('valid') is True]
        ok_g = [v for v in ok_sc if (v.get('fair') or {}).get('geometry_verified') is True]
        ok_b = [v for v in ok_v if (v.get('fair') or {}).get('geometry_verified') is True]
        n_sc.append(len(ok_sc)); n_valid.append(len(ok_v)); n_ver.append(len(ok_g)); n_all.append(len(ok_b))
    return n_sc, n_valid, n_ver, n_all

ns, nv, ng, nb = defs()
for name, arr in (('score_only', ns), ('valid', nv), ('verified', ng), ('valid&verified', nb)):
    a = np.array(arr)
    print('%-16s frames>=1: %4d  (>=2: %4d, >=3: %4d)' % (name, int((a > 0).sum()), int((a >= 2).sum()), int((a >= 3).sum())))
