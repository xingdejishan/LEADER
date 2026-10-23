import json, os
import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TR = os.path.join(DATA, 'evaluator-only-truth')

z = np.load(os.path.join(P, 'results/full-online.npz'), allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], float)
poses = np.asarray(z['poses'], float)
print('n', len(keys))

gt = {}
for d in ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26'):
    obj = json.load(open(os.path.join(TR, d + '.json')))
    for ts, rec in obj.items():
        gt[d + '/' + ts] = np.asarray(rec['T_WB'], float)
print('truth n', len(gt))

miss = [k for k in keys if k not in gt]
print('missing truth', len(miss))

G = np.stack([gt[k] for k in keys])
ep0 = np.linalg.norm(t0[:, :3, 3] - G[:, :3, 3], axis=1)
ep = np.linalg.norm(poses[:, :3, 3] - G[:, :3, 3], axis=1)


def aerr(A, B):
    R = A[:, :3, :3] @ np.transpose(B[:, :3, :3], (0, 2, 1))
    c = (np.trace(R, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


er0 = aerr(t0, G)
er = aerr(poses, G)
for name, e in (('front-end', ep0), ('full-online', ep)):
    print(name, 'n>0.5', int((e > 0.5).sum()), 'n>1', int((e > 1).sum()),
          'n>5', int((e > 5).sum()), 'n>8', int((e > 8).sum()), 'mean', e.mean())
for lo, hi in ((0.5, 1), (1, 2), (2, 5), (5, 8), (8, 1e9)):
    m = (ep0 > lo) & (ep0 <= hi)
    print('band %.1f-%.1f  n=%d  within1m(full-online)=%.3f' % (
        lo, hi, m.sum(), (ep[m] < 1.0).mean() if m.any() else float('nan')))

refs = json.load(open(os.path.join(P, 'inputs/refs.json')))
RP = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
from scipy.spatial import cKDTree
tree = cKDTree(RP)
sel = np.where(ep0 > 0.5)[0]
d, idx = tree.query(t0[sel, :3, 3], k=3)
print('\nselected', len(sel), 'ref dist k1/k3 median', np.median(d[:, 0]), np.median(d[:, 2]))
print('n with k3 > 30 m:', int((d[:, 2] > 30).sum()))
