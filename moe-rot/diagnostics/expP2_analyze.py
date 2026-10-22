"""Analyse expP: when is the image rotation usable, and does a gated fusion help?"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SRC = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expP_fusion.json'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])

rows = json.load(open(SRC))
ok = [r for r in rows if r.get('ok')]
print('frames %d ; with an image rotation %d' % (len(rows), len(ok)))

mi = np.array([r['moe_image'] for r in ok])
mp = np.array([r['moe0'] for r in ok])
print('image MOE: median %.2f  p75 %.2f  p90 %.2f  max %.1f' %
      (np.median(mi), np.percentile(mi, 75), np.percentile(mi, 90), mi.max()))
print('prior MOE: median %.2f  mean %.2f' % (np.median(mp), mp.mean()))

print('\n## by inlier count')
print('   %-12s %5s %9s %9s %9s' % ('inliers', 'n', 'MOE prior', 'MOE image', 'image wins'))
for lo, hi in ((0, 20), (20, 40), (40, 80), (80, 1000)):
    sel = [r for r in ok if lo <= r['n_inl'] < hi]
    if not sel:
        continue
    a = np.array([r['moe0'] for r in sel])
    b = np.array([r['moe_image'] for r in sel])
    print('   [%3d,%4d) %5d %9.2f %9.2f %5d/%d'
          % (lo, hi if hi < 1000 else 999, len(sel), a.mean(), b.mean(),
             int((b < a).sum()), len(sel)))

print('\n## by reprojection residual')
print('   %-12s %5s %9s %9s %9s' % ('resid px', 'n', 'MOE prior', 'MOE image', 'image wins'))
for lo, hi in ((0, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 99)):
    sel = [r for r in ok if lo <= r['resid'] < hi]
    if not sel:
        continue
    a = np.array([r['moe0'] for r in sel])
    b = np.array([r['moe_image'] for r in sel])
    print('   [%.1f,%.1f) %5d %9.2f %9.2f %5d/%d'
          % (lo, hi if hi < 99 else 9.9, len(sel), a.mean(), b.mean(),
             int((b < a).sum()), len(sel)))


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def expv(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


# independence of the two errors (body frame): is the image error correlated with
# the front end's error?  If not, averaging them must reduce the variance.
ind = [r for r in ok if r['n_inl'] >= 40 and r['resid'] < 1.2 and r['moe_image'] < 10]
print('\n## independence check on the usable subset (n=%d)' % len(ind))
if len(ind) >= 10:
    ep, ei, de = [], [], []
    for r in ind:
        i = r['i']
        G = gt[i, :3, :3]
        R0 = t0[i, :3, :3]
        Ri = R0 @ expv(np.array(r['delta']))
        ep.append(logv(R0.T @ G))
        ei.append(logv(Ri.T @ G))
        de.append(np.array(r['delta']))
    ep, ei = np.degrees(np.array(ep)), np.degrees(np.array(ei))
    print('   mean |prior err| %.2f  |image err| %.2f'
          % (np.linalg.norm(ep, axis=1).mean(), np.linalg.norm(ei, axis=1).mean()))
    cc = np.corrcoef(ep.ravel(), ei.ravel())[0, 1]
    print('   corr(prior err, image err) = %+.3f   (0 => independent, so fusing helps)'
          % cc)
    print('   %-6s %8s %8s' % ('lambda', 'MOE', 'ok2deg%'))
    for lam in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 1.0):
        e = []
        for j, r in enumerate(ind):
            i = r['i']
            R = t0[i, :3, :3] @ expv(lam * np.array(r['delta']))
            e.append(ra(R, gt[i, :3, :3]))
        e = np.array(e)
        print('   %-6.2f %8.3f %8.1f' % (lam, e.mean(), 100 * (e <= 2).mean()))

# global gate: keep the fusion only where the evidence is strong
print('\n## gated fusion (apply only on the usable subset, keep the prior elsewhere)')
for nmin, rmax in ((40, 1.2), (30, 1.0), (20, 0.9)):
    for lam in (0.0, 0.2, 0.3, 0.4, 0.5):
        tot, hit = [], 0
        for r in rows:
            i = r['i']
            R = t0[i, :3, :3]
            if r.get('ok') and r['n_inl'] >= nmin and r['resid'] <= rmax and r['moe_image'] < 10:
                R = R @ expv(lam * np.array(r['delta']))
                hit += 1
            tot.append(ra(R, gt[i, :3, :3]))
        tot = np.array(tot)
        print('   nmin %2d rmax %.1f lam %.2f  applied %3d/%d  MOE %.3f  ok2deg %.1f%%'
              % (nmin, rmax, lam, hit, len(rows), tot.mean(), 100 * (tot <= 2).mean()))
