"""Heading-only blend, done with observable quantities only.

The previous table rotated by the image's *true error* (an oracle).  The only thing
available per frame is the increment between the image pose and the front-end pose,
Delta_w = R_image R_prior^T, expressed in the world frame.  A heading-only blend
applies just its vertical part:

    psi = [log(Delta_w)]_z ,   R_new = Exp(lambda * psi * e_z) * R_prior

which leaves the tilt of the front end untouched, so the question is whether the
image increment carries heading information.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
SRC = {'default': '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_validate.npz',
       'v8up': '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_v8up.npz'}

truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])


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


for tag, path in SRC.items():
    if not os.path.exists(path):
        continue
    rec = list(np.load(path, allow_pickle=True)['rec'])
    print('\n===== variant %s (%d records)' % (tag, len(rec)))
    for solver, key in (('anchored', 'db_an'), ('prior-free', 'db_fr')):
        sel = [r for r in rec if key in r]
        if not sel:
            continue
        i = np.array([r['i'] for r in sel], int)
        d = np.array([r[key] for r in sel], float)          # body-frame increment
        R0 = t0[i, :3, :3]
        G = gt[i, :3, :3]
        # world-frame increment and its vertical part
        base = ra(R0, G)
        psi = np.array([np.degrees(logv(R0[j] @ expv(d[j]) @ R0[j].T)[2])
                        for j in range(len(i))])
        print('  %-11s n=%3d  |increment| median %.2f deg, |heading part| median %.2f deg'
              % (solver, len(i), np.median(np.degrees(np.linalg.norm(d, axis=1))),
                 np.median(np.abs(psi))))
        print('     %-6s %8s %8s %9s' % ('lam', 'MOE', 'median', 'improved%'))
        for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            e = []
            for j in range(len(i)):
                w = np.radians(lam * psi[j]) * np.array([0.0, 0, 1.0])
                e.append(ra(expv(w) @ R0[j], G[j]))
            e = np.array(e)
            print('     %-6.1f %8.3f %8.3f %8.1f%%'
                  % (lam, e.mean(), np.median(e), 100 * (e < base).mean()))
