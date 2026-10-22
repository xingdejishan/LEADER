"""Do the query poses inherit the reference poses' attitude error?

The map is built by placing reference scans with their own (front-end estimated)
poses.  If those poses carry the same ~1.4 deg attitude error, the map is a
rotated copy of the world and every fit to it reproduces that error - which would
explain why the reference cloud prefers a wrong rotation.

Test: for each query frame, average the rotation error of its three nearest
reference frames (only where the references have ground truth) and correlate it
with the query's own error.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'

refs = json.load(open(PROJECT + 'inputs/refs.json'))
print('refs %d  keys %s' % (len(refs), list(refs[0].keys())))
print('sample path %s' % refs[0]['scan_path'])


def truth_ts(path):
    parts = path.split('/')
    date = parts[parts.index('NCLT') + 1]
    ts = os.path.basename(path).replace('.bin', '')
    return date, ts


truth = {}
for f in os.listdir(TRUTH_DIR):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

R_ref = np.stack([np.asarray(r['world_from_body'], float)[:3, :3] for r in refs])
p_ref = np.stack([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs])
cover = [truth.get(truth_ts(r['scan_path'])) is not None for r in refs]
print('refs with ground truth: %d/%d' % (sum(cover), len(refs)))

idx = [i for i, c in enumerate(cover) if c]
G = np.stack([truth[truth_ts(refs[i]['scan_path'])][:3, :3] for i in idx])
wi = []
for i in idx:
    R = R_ref[i].T @ truth[truth_ts(refs[i]['scan_path'])][:3, :3]
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    wi.append(np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2))
wi = np.array(wi)
print('reference pose rotation error: mean %.3f deg  median %.3f'
      % (np.linalg.norm(wi, axis=1).mean(), np.median(np.linalg.norm(wi, axis=1))))

Z = np.load(PROJECT + 'results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
pt = t0[:, :3, 3]

tree = cKDTree(p_ref)
# nearest three references, restricted to those with truth
ref_pos = p_ref[idx]
tree_t = cKDTree(ref_pos)
d, nn = tree_t.query(pt, k=3)
we = wi[nn]                                  # query x 3 x 3
wbar = we.mean(1)
wq = np.zeros((len(t0), 3))
for i in range(len(t0)):
    R = t0[i, :3, :3].T @ gt[i, :3, :3]
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    wq[i] = np.degrees(v * a / (2 * s)) if s > 1e-8 else np.degrees(v / 2)

near = d[:, 0] < 30.0
print('\nframes whose nearest reference is within 30 m: %d/%d' % (near.sum(), len(t0)))
for ax, name in enumerate('xyz'):
    cc = np.corrcoef(wbar[near, ax], wq[near, ax])[0, 1]
    sl = np.polyfit(wbar[near, ax], wq[near, ax], 1)[0]
    print('  axis %s: corr(query err, mean ref err) = %+.3f   slope %.3f' % (name, cc, sl))
print('  vector magnitude: mean|query err| %.3f ; mean|mean ref err| %.3f'
      % (np.linalg.norm(wq[near], axis=1).mean(), np.linalg.norm(wbar[near], axis=1).mean()))
print('  mean|query err - mean ref err| = %.3f'
      % np.linalg.norm(wq[near] - wbar[near], axis=1).mean())

# would subtracting the reference error help?
P = t0.copy()
P[:, :3, :3] = t0[:, :3, :3] @ np.array([np.eye(3)] * 1)[0]
for i in range(len(t0)):
    v = np.radians(wbar[i])
    th = np.linalg.norm(v)
    if th < 1e-9:
        continue
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    P[i, :3, :3] = t0[i, :3, :3] @ (np.eye(3) - np.sin(th) * K + (1 - np.cos(th)) * (K @ K))


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


print('\nMOE now %.4f -> after subtracting the mean reference error %.4f (on the near subset: %.4f -> %.4f)'
      % (ra(t0[:, :3, :3], gt[:, :3, :3]).mean(),
         ra(P[:, :3, :3], gt[:, :3, :3]).mean(),
         ra(t0[near, :3, :3], gt[near, :3, :3]).mean(),
         ra(P[near, :3, :3], gt[near, :3, :3]).mean()))
