import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1/stages/fast_search')
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1/stages')
import fast_lib as F
import pipeline as P

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
S = '/home/zhangcy/.dsh_scratch/rot-coupling-subset.npz'

z = np.load(S, allow_pickle=True)
keys = [str(k) for k in z['keys']]
gt, t0, ep0 = z['gt'], z['t0'], z['ep0']
poseB = z['pose_B']; poseA = z['pose_A']; pose0 = z['pose_baseline']

gtall = {}
for name in sorted(os.listdir(TRUTH_DIR)):
    if name.endswith('.json'):
        for ts, rec in json.load(open(os.path.join(TRUTH_DIR, name))).items():
            gtall[name[:-5] + '/' + ts] = np.asarray(rec['T_WB'], float)

refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
for row in refs:
    row['scan_path'] = os.path.realpath(row['scan_path'])
maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                   radius=60.0, max_refs=60, max_maps=20)
cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500, workers=4)


def ang(A, G):
    R = A[:3, :3] @ G[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def cost_at(tree, scan, R, t, trunc=1.0):
    d = tree.query(scan @ R.T + t, k=1, workers=4, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


bands = {'5-8m': np.where((ep0 > 5) & (ep0 <= 8))[0],
         '>8m': np.where(ep0 > 8)[0][:25],
         '2-5m': np.where((ep0 > 2) & (ep0 <= 5))[0][:25],
         'ctrl': np.where(ep0 <= 0.5)[0][:15]}
print('band  n   s_true(med)  |d_true|>2   |s|>8   cost: truth / base / A / B   '
      'MPE base/A/B   er0')
for name, idx in bands.items():
    st, dt, out2, out8 = [], [], 0, 0
    ct, cb, ca, cbb = [], [], [], []
    eb, ea, ebb = [], [], []
    for i in idx:
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key, 'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        T0 = t0[i]
        centre = T0[:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        tree = cKDTree(mp)
        u = P.along_track_direction(T0)
        lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
        dv = gt[i][:3, 3] - T0[:3, 3]
        s = float(dv @ u); d = float(dv @ lat)
        st.append(s); dt.append(d)
        out2 += int(abs(d) > 2.0); out8 += int(abs(s) > 8.0)
        ct.append(cost_at(tree, scan, gt[i][:3, :3], gt[i][:3, 3]))
        cb.append(cost_at(tree, scan, pose0[i][:3, :3], pose0[i][:3, 3]))
        ca.append(cost_at(tree, scan, poseA[i][:3, :3], poseA[i][:3, 3]))
        cbb.append(cost_at(tree, scan, poseB[i][:3, :3], poseB[i][:3, 3]))
        eb.append(np.linalg.norm(pose0[i][:3, 3] - gt[i][:3, 3]))
        ea.append(np.linalg.norm(poseA[i][:3, 3] - gt[i][:3, 3]))
        ebb.append(np.linalg.norm(poseB[i][:3, 3] - gt[i][:3, 3]))
    print('%-5s %3d  %6.2f       %3d/%d      %3d/%d   %.3f / %.3f / %.3f / %.3f   '
          '%.2f/%.2f/%.2f   %.2f'
          % (name, len(idx), np.median(st), out2, len(idx), out8, len(idx),
             np.mean(ct), np.mean(cb), np.mean(ca), np.mean(cbb),
             np.mean(eb), np.mean(ea), np.mean(ebb),
             np.mean([ang(t0[i], gt[i]) for i in idx])))
