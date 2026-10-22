"""Analyse expP with the frame conversion fixed.

expP stored the rotation offset in the *camera* frame (the solver works with
world->camera).  The body-in-world rotation is R_body = R_cw^T R_BC^T, so a
camera-frame offset dc converts to a body-frame offset db = R_BC dc.  This
recomputes everything in the body frame before judging the image estimate.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
LIB = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json'
SRC = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expP_fusion.json'

R_BC = np.asarray(json.load(open(LIB))['T_BC'], float)[:3, :3]
print('R_BC loaded')

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])


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


rows = json.load(open(SRC))
ok = [r for r in rows if r.get('ok')]
for r in ok:
    dc = np.array(r['delta'], float)
    r['db'] = R_BC @ dc                       # camera-frame offset -> body-frame
mp, mi = [], []
for r in ok:
    i = r['i']
    Rb = t0[i, :3, :3] @ expv(r['db'])
    mp.append(ra(t0[i, :3, :3], gt[i, :3, :3]))
    mi.append(ra(Rb, gt[i, :3, :3]))
mp, mi = np.array(mp), np.array(mi)
print('\nimage MOE: mean %.3f median %.3f p90 %.3f max %.1f'
      % (mi.mean(), np.median(mi), np.percentile(mi, 90), mi.max()))
print('prior MOE: mean %.3f median %.3f' % (mp.mean(), np.median(mp)))
print('image better on %d/%d frames' % (int((mi < mp).sum()), len(ok)))

print('\n## by inlier count')
print('   %-12s %5s %9s %9s %9s' % ('inliers', 'n', 'MOE prior', 'MOE image', 'better'))
for lo, hi in ((0, 20), (20, 40), (40, 80), (80, 1000)):
    sel = [k for k, r in enumerate(ok) if lo <= r['n_inl'] < hi]
    if not sel:
        continue
    print('   [%3d,%4d) %5d %9.2f %9.2f %5d/%d'
          % (lo, hi if hi < 1000 else 999, len(sel), mp[sel].mean(), mi[sel].mean(),
             int((mi[sel] < mp[sel]).sum()), len(sel)))

print('\n## blunder analysis')
bl = mi > 5
print('   frames with an image error above 5 deg: %d/%d' % (int(bl.sum()), len(ok)))
if bl.any():
    print('   their inliers median %d, residual median %.2f px'
          % (int(np.median([ok[k]['n_inl'] for k in range(len(ok)) if bl[k]])),
             np.median([ok[k]['resid'] for k in range(len(ok)) if bl[k]])))
good = (~bl) & (mi < mp)
print('   frames where the image is right (<5 deg and better than the prior): %d' % int(good.sum()))

print('\n## fusion sweep on the frames that are not blunders')
for label, sel in (('all', np.ones(len(ok), bool)),
                   ('image<5deg', ~bl),
                   ('image<5 and inl>=20', (~bl) & np.array([r['n_inl'] >= 20 for r in ok]))):
    print('   --- %s (n=%d)' % (label, int(sel.sum())))
    for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        e = []
        for k, r in enumerate(ok):
            if not sel[k]:
                continue
            i = r['i']
            R = t0[i, :3, :3] @ expv(lam * r['db'])
            e.append(ra(R, gt[i, :3, :3]))
        e = np.array(e)
        print('      lam %.1f  MOE %.3f  median %.3f  ok2deg %.1f%%'
              % (lam, e.mean(), np.median(e), 100 * (e <= 2).mean()))

print('\n## independence: correlation of the two error vectors (body frame)')
sel = (~bl) & np.array([r['n_inl'] >= 20 for r in ok])
ep, ei = [], []
for k, r in enumerate(ok):
    if not sel[k]:
        continue
    i = r['i']
    R0 = t0[i, :3, :3]
    Ri = R0 @ expv(r['db'])
    ep.append(np.degrees(_log(R0.T @ gt[i, :3, :3])) if False else None)
print('   (skipped: needs the log map, computed below)')


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


ep, ei = [], []
for k, r in enumerate(ok):
    if not sel[k]:
        continue
    i = r['i']
    R0 = t0[i, :3, :3]
    Ri = R0 @ expv(r['db'])
    ep.append(np.degrees(logv(R0.T @ gt[i, :3, :3])))
    ei.append(np.degrees(logv(Ri.T @ gt[i, :3, :3])))
ep, ei = np.array(ep), np.array(ei)
print('   n=%d  mean|prior| %.2f  mean|image| %.2f  corr %+.3f'
      % (len(ep), np.linalg.norm(ep, axis=1).mean(),
         np.linalg.norm(ei, axis=1).mean(), np.corrcoef(ep.ravel(), ei.ravel())[0, 1]))
