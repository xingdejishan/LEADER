"""EXP H: split the two modalities - translation from geometry, rotation from vision.

The geometric cost has no rotation signal (measured), the visual branch returns a
full 6-dof candidate but was judged on the same geometric cost.  So the natural
remedy is to keep the geometric translation and take the rotation from the visual
candidate.  This measures every combination on the 1502-frame set.
"""
import json
import os

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def rot_angle(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def log_vec(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)


truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

I = np.load(PROJECT + 'inputs/initial.npz', allow_pickle=True)
keys = [str(k) for k in I['keys']]
v7 = np.asarray(I['poses'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])

# geometric branch outputs on the same key set
geo = {}
for name in ('results/final.npz', 'results/hybrid_rotation.npz',
             'results/three_route_merged.npz', 'results/three_route_unified.npz',
             'results/three_route.npz', 'results/yaw_sweep.npz'):
    p = PROJECT + name
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    kz = [str(k) for k in z['keys']]
    if kz == keys:
        geo[name.split('/')[-1]] = np.asarray(z['poses'], float)

# visual candidates
n = len(keys)
vis_ok = np.zeros(n, bool)
vis_pose = np.zeros((n, 4, 4))
vis_score = np.full(n, np.nan)
n_cand = np.zeros(n, int)
for i in range(n):
    p = VERIFY % i
    if not os.path.exists(p):
        continue
    rec = json.load(open(p))['poses']
    best = None
    c = 0
    for name, row in rec.items():
        if not name.startswith('candidate_'):
            continue
        c += 1
        sc = (row.get('fair') or {}).get('geometry_score')
        if sc is None:
            continue
        if best is None or sc > best[0]:
            best = (float(sc), np.asarray(row['pose'], float))
    n_cand[i] = c
    if best is not None:
        vis_pose[i] = best[1]
        vis_score[i] = best[0]
        vis_ok[i] = True
print('frames %d ; with a visual candidate %d ; candidates per frame median %d'
      % (n, int(vis_ok.sum()), int(np.median(n_cand))))

rows = []


def row(tag, P, mask=None):
    m = np.ones(len(P), bool) if mask is None else mask
    et = np.linalg.norm(P[m, :3, 3] - gt[m, :3, 3], axis=1)
    er = rot_angle(P[m, :3, :3], gt[m, :3, :3])
    rows.append((tag, int(m.sum()), et.mean(), er.mean(), np.median(er),
                 100 * ((et <= 1) & (er <= 2)).mean(),
                 int((et > 1).sum()), int((et > 5).sum())))
    return rows[-1]


def show():
    print('\n%-38s %5s %8s %8s %8s %9s %6s %6s' %
          ('configuration', 'n', 'MPE', 'MOE', 'medMOE', 'ok(1m,2d)%', '>1m', '>5m'))
    for r in rows:
        print('%-38s %5d %8.4f %8.4f %8.3f %9.1f %6d %6d' % r)


row('V7 front end (baseline)', v7)
for name, P in geo.items():
    row('geometric branch: %s' % name, P)

V = vis_ok
row('visual candidate (as is)', vis_pose, V)

# combination: geometric translation + visual rotation
for name, P in geo.items():
    C = P.copy()
    C[V, :3, :3] = vis_pose[V, :3, :3]
    row('geo translation + visual rotation (%s)' % name, C, V)

# combination: V7 translation + visual rotation
C = v7.copy()
C[V, :3, :3] = vis_pose[V, :3, :3]
row('V7 translation + visual rotation', C, V)

show()

# how good is the visual rotation really?
er_v7 = rot_angle(v7[V, :3, :3], gt[V, :3, :3])
er_vis = rot_angle(vis_pose[V, :3, :3], gt[V, :3, :3])
print('\n## rotation-only comparison on the %d frames with a candidate' % int(V.sum()))
print('   V7    MOE %.4f   median %.3f   tilt %.3f  yaw %.3f'
      % (er_v7.mean(), np.median(er_v7),
         np.mean([np.linalg.norm(log_vec(a.T @ b)[:2]) for a, b in
                  zip(v7[V, :3, :3], gt[V, :3, :3])]),
         np.mean([abs(log_vec(a.T @ b)[2]) for a, b in
                  zip(v7[V, :3, :3], gt[V, :3, :3])])))
print('   visual MOE %.4f  median %.3f   tilt %.3f  yaw %.3f'
      % (er_vis.mean(), np.median(er_vis),
         np.mean([np.linalg.norm(log_vec(a.T @ b)[:2]) for a, b in
                  zip(vis_pose[V, :3, :3], gt[V, :3, :3])]),
         np.mean([abs(log_vec(a.T @ b)[2]) for a, b in
                  zip(vis_pose[V, :3, :3], gt[V, :3, :3])])))
print('   frames where the visual rotation is better: %d/%d (%.1f%%)'
      % (int((er_vis < er_v7).sum()), int(V.sum()), 100 * (er_vis < er_v7).mean()))

# what if we take the *better* of the two rotations per frame (oracle selector)?
er_best = np.minimum(er_v7, er_vis)
print('   oracle picks the better rotation: MOE %.4f' % er_best.mean())

# and on the gated subset only (ratio < 0.21 was the gate; approximate here by
# the frames the arbitration actually switched)
for name, P in geo.items():
    if 'hybrid' in name:
        changed = np.linalg.norm(P[:, :3, 3] - v7[:, :3, 3], axis=1) > 1e-6
        print('   %s: %d frames differ from V7' % (name, int(changed.sum())))
