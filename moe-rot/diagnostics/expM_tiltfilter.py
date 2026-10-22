"""EXP M: filter the tilt, keep the heading.

The attitude error splits into heading (0.82 deg) and tilt (0.99 deg).  On a car
the tilt is set by the road and changes slowly, while the heading follows the
steering and changes fast (7 deg/s).  So a centred temporal average of the body's
up direction - leaving the heading untouched - should remove part of the tilt
error without touching the part that carries real motion.

Also reports how much of the *true* tilt change the same filter would smear, so
the gain can be read net of the bias it introduces.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])

order = np.array(sorted(range(len(keys)),
                        key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1]))))
ts = np.array([int(keys[i].split('/')[1]) for i in order], dtype=np.int64)
dt = np.diff(ts) / 1e6
gap = np.zeros(len(order), int)
for i in range(1, len(order)):
    gap[i] = gap[i - 1] + (1 if dt[i - 1] > 0.6 else 0)


def split(R):
    """(up direction, heading angle) of a rotation matrix."""
    u = R[:, 2]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return u, yaw


def build(u, yaw):
    u = u / np.linalg.norm(u)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = xc - (xc @ u) * u
    x /= np.linalg.norm(x)
    y = np.cross(u, x)
    return np.stack([x, y, u], axis=1)


def filt(R, k):
    U = np.array([split(Ro)[0] for Ro in R])
    Y = np.array([split(Ro)[1] for Ro in R])
    half = k // 2
    out = np.empty_like(R)
    for i in range(len(R)):
        lo, hi = max(0, i - half), min(len(R), i + half + 1)
        idx = [j for j in range(lo, hi) if gap[j] == gap[i]]
        u = U[idx].mean(0)
        out[i] = build(u, Y[i])
    return out


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


Ro = t0[order, :3, :3]
Go = gt[order, :3, :3]
base = ra(Ro, Go)
print('MOE %.4f  (bulk %.4f)' % (base.mean(), base[ra(t0[order, :3, :3], Go) * 0 < 1].mean()))

print('\n## true tilt change between neighbouring frames')
Ut = np.array([split(R_)[0] for R_ in Go])
d_ut = np.degrees(np.arccos(np.clip((Ut[:-1] * Ut[1:]).sum(1), -1, 1)))
print('   mean angle between consecutive true up-vectors: %.3f deg' % d_ut.mean())

print('\n## tilt-only centred average')
print('   %-4s %9s %9s %9s' % ('k', 'MOE', 'medMOE', 'ok(1m,2d)%'))
for k in (1, 3, 5, 7, 9, 13, 17, 25):
    F = filt(Ro, k)
    e = ra(F, Go)
    ok = ((np.linalg.norm(t0[order, :3, 3] - gt[order, :3, 3], axis=1) <= 1) & (e <= 2)).mean() * 100
    print('   %-4d %9.4f %9.4f %9.2f' % (k, e.mean(), np.median(e), ok))

print('\n## the same filter applied to the truth itself (bias it introduces)')
for k in (3, 5, 9, 17):
    F = filt(Go, k)
    print('   k=%-3d truth -> smoothed: %.4f deg' % (k, ra(F, Go).mean()))

# how much of the error is in the up-vector vs the heading?
u0 = np.array([split(R_)[0] for R_ in Ro])
y0 = np.array([split(R_)[1] for R_ in Ro])
ug = np.array([split(R_)[0] for R_ in Go])
yg = np.array([split(R_)[1] for R_ in Go])
print('\ntilt error (angle between estimated and true up): %.3f deg'
      % np.degrees(np.arccos(np.clip((u0 * ug).sum(1), -1, 1))).mean())
print('heading error (absolute yaw difference):           %.3f deg'
      % np.degrees(np.abs(np.arctan2(np.sin(y0 - yg), np.cos(y0 - yg)))).mean())
