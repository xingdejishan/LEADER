"""EXP I: what is left if the attitude is made level with the world vertical?

This does not use the map at all: it measures how much of MOE is a *tilt* of the
estimated attitude relative to the world frame's vertical (the frame the truth is
expressed in), which is the degree of freedom a gravity / road-plane prior would
supply.  Yaw is untouched.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'


def rot_angle(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
et = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)
dates = np.array([k.split('/')[0] for k in keys])
up = np.array([0.0, 0, 1.0])

z_axis = t0[:, :3, 0] * 0 + np.array([0.0, 0, 1.0])          # body z in the body frame
u = np.einsum('nij,j->ni', t0[:, :3, :3], np.array([0.0, 0, 1.0]))  # body z in the world
lev = t0.copy()
for i in range(len(t0)):
    lev[i, :3, :3] = rot_between(u[i], up) @ t0[i, :3, :3]

tilt_before = np.array([np.degrees(np.arccos(np.clip(u[i] @ up, -1, 1))) for i in range(len(u))])
moe_before = rot_angle(t0[:, :3, :3], gt[:, :3, :3])
moe_lev = rot_angle(lev[:, :3, :3], gt[:, :3, :3])
print('frames %d' % len(t0))
print('mean angle of the estimated body-z from world up before: %.3f deg' % tilt_before.mean())
print('MOE %.4f -> %.4f after levelling   (median %.3f -> %.3f)'
      % (moe_before.mean(), moe_lev.mean(), np.median(moe_before), np.median(moe_lev)))
print('ok(1m,2d) %.1f%% -> %.1f%%'
      % (100 * ((et <= 1) & (moe_before <= 2)).mean(), 100 * ((et <= 1) & (moe_lev <= 2)).mean()))
print('frames improved %d / worsened %d' % (int((moe_lev < moe_before).sum()), int((moe_lev > moe_before).sum())))
print('\n%-12s %6s %9s %9s %9s' % ('date', 'n', 'tilt', 'MOE', 'MOE level'))
for d in sorted(set(dates.tolist())):
    m = dates == d
    print('%-12s %6d %9.3f %9.3f %9.3f'
          % (d, int(m.sum()), tilt_before[m].mean(), moe_before[m].mean(), moe_lev[m].mean()))

print('\n%-10s %7s %9s %9s' % ('MPE band', 'n', 'MOE', 'MOE level'))
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, 1e9)):
    m = (et >= lo) & (et < hi)
    print('[%4.1f,%s) %7d %9.3f %9.3f'
          % (lo, ('%4.1f' % hi) if hi < 1e8 else ' inf', int(m.sum()), moe_before[m].mean(), moe_lev[m].mean()))

# and how much would a *yaw-only* correction recover if the yaw were known?
print('\nfor reference: a perfect yaw correction would leave the tilt part, '
      'a perfect tilt correction would leave the yaw part.')
