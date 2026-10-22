"""EXP D: is the front-end rotation error white in time?

The rotation error is ~1.4 deg on essentially every frame while the position is
already 0.11 m.  If that error is temporally white, the attitude signal can be
denoised by a small centred average over neighbouring frames (the vehicle's
attitude is smooth at 10 Hz), which would cut MOE without touching the map.

Also reports what a centred moving average of width k does to MOE.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')


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
    n = q.shape[0]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((n, 3, 3))
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def rot_angle(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


blob = np.load(f'{PROJECT}/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in blob['keys']]
t0 = np.asarray(blob['t0'], np.float64)
poses = np.asarray(blob['poses'], np.float64)
truth = {}
for f in os.listdir(TRUTH_DIR):
    date = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
et0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)

# order frames per date by timestamp
order = sorted(range(len(keys)), key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1])))
order = np.array(order)
ts = np.array([int(keys[i].split('/')[1]) for i in order], dtype=np.int64)
dt = np.diff(ts) / 1e6
print('frame spacing: median %.3f s  p10 %.3f  p90 %.3f  gap>1s: %d'
      % (np.median(dt), np.percentile(dt, 10), np.percentile(dt, 90), int((dt > 1.0).sum())))
print('dates order:', [keys[order[j]].split('/')[0] for j in range(0, len(order), len(order) // 4)])

Ro = t0[order, :3, :3]
Go = gt[order, :3, :3]
err = rot_angle(Ro, Go)
print('\nt0 MOE on the ordered set: %.4f deg' % err.mean())

# whiteness: |error(t+1) - error(t)| vs |error|
Rerr = np.swapaxes(Ro, -1, -2) @ Go          # per-frame error rotation
w = np.zeros((len(order), 3))
for i, R in enumerate(Rerr):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    w[i] = np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)
same = dt < 0.6
dw = np.linalg.norm(np.diff(w, axis=0), axis=1)
print('level of the error vector  mean|w| %.3f deg' % np.linalg.norm(w, axis=1).mean())
print('consecutive difference     mean|dw| %.3f deg   (white => sqrt(2)x level = %.3f)'
      % (dw[same].mean(), np.sqrt(2) * np.linalg.norm(w, axis=1).mean()))
cc = np.corrcoef(w[:-1][same].ravel(), w[1:][same].ravel())[0, 1]
print('lag-1 correlation of the error components: %.3f  (white => 0)' % cc)

# centred moving average in quaternion space, respecting gaps
q = quat_from_R(Ro)
qm = quat_from_R(Go)
q = q * np.sign(q @ q[0])[:, None]


def smooth(q, k, gap):
    half = k // 2
    out = q.copy()
    n = len(q)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = [j for j in range(lo, hi) if gap[j] == gap[i]]
        if len(idx) < 2:
            continue
        qq = q[idx] * np.sign(q[idx] @ q[i])[:, None]
        m = qq.mean(0)
        out[i] = m / np.linalg.norm(m)
    return out


gap = np.zeros(len(order), int)
for i in range(1, len(order)):
    gap[i] = gap[i - 1] + (1 if dt[i - 1] > 0.6 else 0)

print('\n## centred quaternion average of the front-end attitude')
print('   %-4s %9s %9s %9s %9s' % ('k', 'MOE', 'MOE bulk', 'medMOE', 'ok(1m,2d)%'))
for k in (1, 3, 5, 7, 9, 11, 15, 21):
    qs = smooth(q, k, gap)
    Rs = R_from_quat(qs)
    e = rot_angle(Rs, Go)
    bulk = et0[order] < 0.2
    ok = ((et0[order] <= 1) & (e <= 2)).mean() * 100
    print('   %-4d %9.4f %9.4f %9.4f %9.2f' % (k, e.mean(), e[bulk].mean(), np.median(e), ok))

print('\n## same smoothing on the truth attitude itself (how much real motion it removes)')
for k in (5, 11, 21):
    Ms = R_from_quat(smooth(quat_from_R(Go), k, gap))
    print('   k=%-3d  truth smoothed vs truth: %.4f deg' % (k, rot_angle(Ms, Go).mean()))
