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
poseA, poseB, pose0 = z['pose_A'], z['pose_B'], z['pose_baseline']
J = json.load(open(S.replace('.npz', '.json')))
rows = J['rows']

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


m = np.where((ep0 > 5) & (ep0 <= 8))[0]
sv = np.array([rows[i]['diag']['B']['s'] for i in m])
stuck = m[np.abs(sv) < 1e-9]
moved = m[np.abs(sv) >= 1e-9]
print('5-8m: stuck(s_B=0)=%d  moved=%d' % (len(stuck), len(moved)))

for name, idx in (('stuck', stuck), ('moved', moved)):
    st, dt, ct, cf, err = [], [], [], [], []
    for i in idx:
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key, 'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        T0 = t0[i]
        mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:3])
        tree = cKDTree(mp)
        u = P.along_track_direction(T0)
        lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
        dv = gt[i][:3, 3] - T0[:3, 3]
        st.append(float(dv @ u)); dt.append(float(dv @ lat))
        ct.append(cost_at(tree, scan, gt[i][:3, :3], gt[i][:3, 3]))
        cf.append(cost_at(tree, scan, poseB[i][:3, :3], poseB[i][:3, 3]))
        err.append(np.linalg.norm(poseB[i][:3, 3] - gt[i][:3, 3]))
    print('  %-6s n=%2d  s_true med %.2f  |d_true|>2: %d/%d  cost truth %.3f vs found %.3f  '
          'MPE %.2f  ok %.2f' % (
              name, len(idx), np.median(st), int((np.abs(dt) > 2).sum()), len(idx),
              np.mean(ct), np.mean(cf), np.mean(err),
              np.mean([(np.linalg.norm(poseB[i][:3, 3] - gt[i][:3, 3]) <= 1 and
                        ang(poseB[i], gt[i]) <= 2) for i in idx])))

print('\nB yaw chosen by band:')
for label, msk in (('>0.5m', ep0 > 0.5), ('control', ep0 <= 0.5), ('5-8m', (ep0 > 5) & (ep0 <= 8))):
    yaws = np.array([rows[i]['diag']['B']['yaw'] for i in np.where(msk)[0]])
    print('  %-7s n=%4d  ' % (label, len(yaws)) +
          '  '.join('%+.0f:%.2f' % (d, (yaws == d).mean()) for d in (-3, -2, -1, 0, 1, 2, 3)))

print('\ncontrol: ok rate vs chosen yaw (B) vs baseline')
idx = np.where(ep0 <= 0.5)[0]
yaws = np.array([rows[i]['diag']['B']['yaw'] for i in idx])
okb = np.array([(np.linalg.norm(pose0[i][:3, 3] - gt[i][:3, 3]) <= 1 and ang(pose0[i], gt[i]) <= 2) for i in idx])
okB = np.array([(np.linalg.norm(poseB[i][:3, 3] - gt[i][:3, 3]) <= 1 and ang(poseB[i], gt[i]) <= 2) for i in idx])
for y in (-3, -2, -1, 0, 1, 2, 3):
    k = yaws == y
    if k.sum():
        print('  yaw=%+.0f n=%4d  base_ok %.3f  B_ok %.3f  base_er %.2f B_er %.2f'
              % (y, k.sum(), okb[k].mean(), okB[k].mean(),
                 np.mean([ang(pose0[i], gt[i]) for i in idx[k]]),
                 np.mean([ang(poseB[i], gt[i]) for i in idx[k]])))
