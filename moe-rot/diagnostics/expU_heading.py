"""Decompose both rotation errors into heading (world yaw) and tilt, per solver.

If the image is a better heading sensor than the front end even though its total
error is larger, a heading-only blend is worth something - the front end's error
splits roughly half heading, half tilt.
"""
import json
import os
import sys

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


def world_log(Rpred, Rgt):
    R = Rgt @ Rpred.T                      # prediction -> truth, expressed in the world
    return np.degrees(logv(R))


for tag, path in SRC.items():
    if not os.path.exists(path):
        print('%s missing' % tag)
        continue
    rec = list(np.load(path, allow_pickle=True)['rec'])
    print('\n===== variant %s (%d records)' % (tag, len(rec)))
    for solver, key in (('anchored', 'db_an'), ('prior-free', 'db_fr')):
        sel = [r for r in rec if key in r]
        if not sel:
            continue
        i = np.array([r['i'] for r in sel], int)
        d = np.array([r[key] for r in sel], float)
        wp, wi = [], []
        for j, k in enumerate(i):
            wp.append(world_log(t0[k, :3, :3], gt[k, :3, :3]))
            wi.append(world_log(t0[k, :3, :3] @ expv(d[j]), gt[k, :3, :3]))
        wp, wi = np.array(wp), np.array(wi)
        print('  %-11s n=%3d  heading err %.3f -> %.3f   tilt err %.3f -> %.3f'
              % (solver, len(i), np.abs(wp[:, 2]).mean(), np.abs(wi[:, 2]).mean(),
                 np.linalg.norm(wp[:, :2], axis=1).mean(),
                 np.linalg.norm(wi[:, :2], axis=1).mean()))
        # heading-only blend
        print('     heading-only blend (tilt left at the front end):')
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            e = []
            for j, k in enumerate(i):
                W = np.array([0.0, 0.0, 0.0])
                W[2] = lam * wi[j, 2]
                R = expv(np.radians(W)) @ t0[k, :3, :3]
                e.append(ra(R, gt[k, :3, :3]))
            print('        lam %.1f -> MOE %.3f' % (lam, np.mean(e)))
