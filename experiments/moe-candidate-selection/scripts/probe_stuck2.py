import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1/stages/fast_search')
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1/stages')
import fast_lib as F
import pipeline as P

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
S = '/home/zhangcy/.dsh_scratch/rot-coupling-subset.npz'
z = np.load(S, allow_pickle=True)
keys = [str(k) for k in z['keys']]
gt, t0, ep0 = z['gt'], z['t0'], z['ep0']
poseB = z['pose_B']
J = json.load(open(S.replace('.npz', '.json')))
rows = J['rows']

refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
for row in refs:
    row['scan_path'] = os.path.realpath(row['scan_path'])
maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                   radius=60.0, max_refs=60, max_maps=20)
cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=4)


def cost_at(tree, scan, R, t, trunc=1.0):
    d = tree.query(scan @ R.T + t, k=1, workers=4, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


m = np.where((ep0 > 5) & (ep0 <= 8))[0]
sv = np.array([rows[i]['diag']['B']['s'] for i in m])
stuck = m[np.abs(sv) < 1e-9]
print('stuck n=', len(stuck))
acc = {k: [] for k in ('gt_t/gt_R', 'gt_t/T0_R', 'found_t/gt_R', 'found_t/found_R', 'T0_t/T0_R')}
for i in stuck:
    key = keys[i]
    date, ts = key.split('/')
    rec = {'key': key, 'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
    scan = P.prepare_scan(rec, cfg)
    T0 = t0[i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
    tree = cKDTree(mp)
    G = gt[i]; B = poseB[i]
    acc['gt_t/gt_R'].append(cost_at(tree, scan, G[:3, :3], G[:3, 3]))
    acc['gt_t/T0_R'].append(cost_at(tree, scan, T0[:3, :3], G[:3, 3]))
    acc['found_t/gt_R'].append(cost_at(tree, scan, G[:3, :3], B[:3, 3]))
    acc['found_t/found_R'].append(cost_at(tree, scan, B[:3, :3], B[:3, 3]))
    acc['T0_t/T0_R'].append(cost_at(tree, scan, T0[:3, :3], T0[:3, 3]))
for k, v in acc.items():
    print('  %-18s mean cost %.3f' % (k, np.mean(v)))
