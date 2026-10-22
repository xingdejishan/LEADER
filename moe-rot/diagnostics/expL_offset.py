"""Best possible single constant rotation offset (both sides), properly applied."""
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


def quat(Rm):
    n = len(Rm)
    q = np.zeros((n, 4))
    tr = np.trace(Rm, axis1=-2, axis2=-1)
    big = tr > 0
    q[big, 0] = 0.5 * np.sqrt(1 + tr[big])
    d = 4 * q[big, 0]
    q[big, 1] = (Rm[big, 2, 1] - Rm[big, 1, 2]) / d
    q[big, 2] = (Rm[big, 0, 2] - Rm[big, 2, 0]) / d
    q[big, 3] = (Rm[big, 1, 0] - Rm[big, 0, 1]) / d
    for i in np.where(~big)[0]:
        M = Rm[i]
        if M[0, 0] >= M[1, 1] and M[0, 0] >= M[2, 2]:
            s = 2 * np.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2])
            q[i] = [(M[2, 1] - M[1, 2]) / s, .25 * s, (M[0, 1] + M[1, 0]) / s,
                    (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] >= M[2, 2]:
            s = 2 * np.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2])
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, .25 * s,
                    (M[1, 2] + M[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1])
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s,
                    (M[1, 2] + M[2, 1]) / s, .25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def Rf(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


base = ra(t0[:, :3, :3], gt[:, :3, :3]).mean()
print('MOE now %.4f' % base)
for side in ('right', 'left'):
    E = (np.swapaxes(t0[:, :3, :3], -1, -2) @ gt[:, :3, :3]) if side == 'right' \
        else (gt[:, :3, :3] @ np.swapaxes(t0[:, :3, :3], -1, -2))
    q = quat(E)
    q = q * np.sign(q @ q[0])[:, None]
    m = q.mean(0)
    m /= np.linalg.norm(m)
    Rc = Rf(m)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rc) - 1) / 2, -1, 1)))
    # 'right': residual = Rpred @ Rc ; 'left': residual = Rc @ Rpred
    P = t0.copy()
    if side == 'right':
        P[:, :3, :3] = t0[:, :3, :3] @ Rc
    else:
        P[:, :3, :3] = Rc @ t0[:, :3, :3]
    print('  %-5s offset %.3f deg -> MOE %.4f' % (side, ang, ra(P[:, :3, :3], gt[:, :3, :3]).mean()))
# also the naive product already tried
E = np.swapaxes(t0[:, :3, :3], -1, -2) @ gt[:, :3, :3]
q = quat(E)
q = q * np.sign(q @ q[0])[:, None]
m = q.mean(0)
m /= np.linalg.norm(m)
Rc = Rf(m)
P = t0.copy()
P[:, :3, :3] = Rc @ t0[:, :3, :3]
print('  (the product tried earlier, Rc @ err) -> %.4f' % ra(P[:, :3, :3], gt[:, :3, :3]).mean())
# mean angle of the error with its own mean removed, in the body frame
E2 = np.swapaxes(t0[:, :3, :3], -1, -2) @ gt[:, :3, :3]
RcI = Rc.T
print('  residual after removing the offset in the body frame: %.4f'
      % ra(RcI @ E2, np.array([np.eye(3)] * len(E2))))
