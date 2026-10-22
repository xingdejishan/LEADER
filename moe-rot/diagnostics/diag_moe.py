"""Decompose MOE on the full 104,728-frame set.

Questions:
  1. how much of the mean MOE comes from which position-error buckets?
  2. is the rotation error yaw-dominated or tilt-dominated?
  3. is there a systematic rotation offset (calibration / gravity) that a single
     constant correction would remove?
"""
import json
import os

import numpy as np

R = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'


def log_so3(Rm):
    tr = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    w = np.stack([Rm[..., 2, 1] - Rm[..., 1, 2],
                  Rm[..., 0, 2] - Rm[..., 2, 0],
                  Rm[..., 1, 0] - Rm[..., 0, 1]], axis=-1)
    s = np.sin(ang)
    small = s < 1e-8
    scale = np.where(small, 0.5, ang / (2.0 * np.where(small, 1.0, s)))
    return w * scale[..., None]


def rot_angle(Rm):
    tr = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def quat_from_R(Rm):
    n = Rm.shape[0]
    q = np.zeros((n, 4))
    tr = np.trace(Rm, axis1=-2, axis2=-1)
    m00, m11, m22 = Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]
    big = tr > 0
    if big.any():
        q[big, 0] = 0.5 * np.sqrt(1.0 + tr[big])
        den = 4.0 * q[big, 0]
        q[big, 1] = (Rm[big, 2, 1] - Rm[big, 1, 2]) / den
        q[big, 2] = (Rm[big, 0, 2] - Rm[big, 2, 0]) / den
        q[big, 3] = (Rm[big, 1, 0] - Rm[big, 0, 1]) / den
    for i in np.where(~big)[0]:
        M = Rm[i]
        if m00[i] >= m11[i] and m00[i] >= m22[i]:
            s = 2.0 * np.sqrt(1.0 + m00[i] - m11[i] - m22[i])
            q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s,
                    (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif m11[i] >= m22[i]:
            s = 2.0 * np.sqrt(1.0 + m11[i] - m00[i] - m22[i])
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s,
                    0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m22[i] - m00[i] - m11[i])
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s,
                    (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def R_from_quat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_truth():
    t = {}
    for f in os.listdir(TRUTH):
        d = f.replace('.json', '')
        for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
            t[(d, str(ts))] = np.asarray(v['T_WB'], float)
    return t


Z = np.load(R + 'results/full-online-final.npz', allow_pickle=True)
keys = np.array([str(k) for k in Z['keys']])
poses = Z['poses']
t0 = Z['t0']
truth = load_truth()

dates = np.array([k.split('/')[0] for k in keys])
g = np.array([truth.get((k.split('/')[0], k.split('/')[1]), np.full((4, 4), np.nan))
              for k in keys])
ok = ~np.isnan(g[:, 0, 0])
print('frames %d  with truth %d' % (len(keys), int(ok.sum())))


def errs(P):
    et = np.linalg.norm(P[:, :3, 3] - g[:, :3, 3], axis=1)
    Re = np.swapaxes(P[:, :3, :3], -1, -2) @ g[:, :3, :3]
    return et, rot_angle(Re)


def table(name, et, er):
    sel = ok
    print('\n## %s   n=%d' % (name, int(sel.sum())))
    print('   MPE %.4f   MOE %.4f   medMPE %.4f   medMOE %.4f   ok(1m,2d) %.1f%%'
          % (et[sel].mean(), er[sel].mean(), np.median(et[sel]), np.median(er[sel]),
             100 * ((et[sel] <= 1) & (er[sel] <= 2)).mean()))
    buckets = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
               (5.0, 8.0), (8.0, 1e9)]
    print('   %-14s %7s %7s %8s %8s %10s' % ('MPE bucket', 'n', 'share', 'MPE', 'MOE', 'MOE share'))
    for lo, hi in buckets:
        m = sel & (et >= lo) & (et < hi)
        n = int(m.sum())
        if n == 0:
            continue
        print('   [%5.1f,%5s) %7d %6.1f%% %8.3f %8.3f %9.1f%%'
              % (lo, ('%5.1f' % hi) if hi < 1e8 else '  inf', n,
                 100.0 * n / sel.sum(), et[m].mean(), er[m].mean(),
                 100.0 * (n * er[m].mean()) / (sel.sum() * er[sel].mean())))
    for d in sorted(set(dates.tolist())):
        m = sel & (dates == d)
        print('   %s  n=%6d  MPE %.4f  MOE %.4f  medMOE %.3f'
              % (d, int(m.sum()), et[m].mean(), er[m].mean(), np.median(er[m])))


et0, er0 = errs(t0)
etf, erf = errs(poses)
table('t0 (V7 front end, refined_V7)', et0, er0)
table('current default (no rotation branch)', etf, erf)

print('\n## yaw vs tilt split (world-frame log map)')
for tag, P, et, er in (('t0', t0, et0, er0), ('now', poses, etf, erf)):
    Re = np.swapaxes(P[:, :3, :3], -1, -2) @ g[:, :3, :3]
    w = np.degrees(log_so3(Re))
    wz = np.abs(w[:, 2])
    wt = np.linalg.norm(w[:, :2], axis=1)
    for lab, m in (('all', ok), ('MPE<1', ok & (et < 1.0)), ('MPE>=1', ok & (et >= 1.0))):
        print('   %-4s %-8s n=%6d  |yaw| %.3f  |tilt| %.3f  angle %.3f'
              % (tag, lab, int(m.sum()), wz[m].mean(), wt[m].mean(), er[m].mean()))
    print('   %-4s signed mean w: x=%+.3f y=%+.3f z=%+.3f deg | median: %+.3f %+.3f %+.3f'
          % (tag, w[ok, 0].mean(), w[ok, 1].mean(), w[ok, 2].mean(),
             np.median(w[ok, 0]), np.median(w[ok, 1]), np.median(w[ok, 2])))

print('\n## one global constant rotation offset (quaternion mean of R_err)')
for tag, P, et in (('t0', t0, et0), ('now', poses, etf)):
    E = np.swapaxes(P[:, :3, :3], -1, -2)[ok] @ g[ok][:, :3, :3]
    q = quat_from_R(E)
    q = q * np.sign(q @ q[0])[:, None]
    qm = q.mean(axis=0)
    qm /= np.linalg.norm(qm)
    Rc = R_from_quat(qm)
    a1 = rot_angle(E)
    a2 = rot_angle(Rc @ E)
    bulk = et[ok] < 1.0
    print('   %-4s offset %.3f deg;  MOE %.4f => %.4f   bulk(<1m) %.4f => %.4f'
          % (tag, rot_angle(Rc[None])[0], a1.mean(), a2.mean(), a1[bulk].mean(), a2[bulk].mean()))
    for d in sorted(set(dates.tolist())):
        m = ok & (dates == d)
        Ed = np.swapaxes(P[m][:, :3, :3], -1, -2) @ g[m][:, :3, :3]
        qd = quat_from_R(Ed)
        qd = qd * np.sign(qd @ qd[0])[:, None]
        qmd = qd.mean(axis=0)
        qmd /= np.linalg.norm(qmd)
        Rcd = R_from_quat(qmd)
        print('      %s offset %.3f deg  MOE %.4f => %.4f'
              % (d, rot_angle(Rcd[None])[0], rot_angle(Ed).mean(),
                 rot_angle(Rcd @ Ed).mean()))

print('\n## 1502 hard subset: initial vs hybrid_rotation')
H = np.load(R + 'inputs/initial.npz', allow_pickle=True)
I = np.load(R + 'results/hybrid_rotation.npz', allow_pickle=True)
F = np.load(R + 'results/final.npz', allow_pickle=True)
kk = [str(k) for k in H['keys']]
gi = np.array([truth[(k.split('/')[0], k.split('/')[1])] for k in kk])
for tag, P in (('initial(V7)', H['poses']), ('hybrid_rotation', I['poses']), ('final', F['poses'])):
    et = np.linalg.norm(P[:, :3, 3] - gi[:, :3, 3], axis=1)
    er = rot_angle(np.swapaxes(P[:, :3, :3], -1, -2) @ gi[:, :3, :3])
    print('   %-16s MPE %.4f  MOE %.4f  medMOE %.3f  ok(1m,2d) %.1f%%'
          % (tag, et.mean(), er.mean(), np.median(er), 100 * ((et <= 1) & (er <= 2)).mean()))
