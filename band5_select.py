"""Select every frame of full-online-final.npz whose t0 translation error is > 5 m.

Truth is read here and nowhere else in the study:
    /data2/.../evaluator-only-truth/<date>.json, {"T_WB": 4x4} per timestamp.

    python band5_select.py --out results/band5-nopose-subset.npz \
        --frames results/band5-nopose-frames.json
"""
import argparse, collections, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH = DATA + '/evaluator-only-truth/%s.json'
FINAL = os.path.join(HERE, 'results/full-online-final.npz')

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--frames', default=os.path.join(HERE, 'results/band5-nopose-frames.json'))
ap.add_argument('--threshold', type=float, default=5.0)
a = ap.parse_args()

sync = np.load(FINAL, allow_pickle=True)
keys = [str(k) for k in sync['keys']]
t0 = np.asarray(sync['t0'], float)
poses = np.asarray(sync['poses'], float)
truth = {d: json.load(open(TRUTH % d)) for d in sorted({k.split('/')[0] for k in keys})}
n = len(keys)
err = np.full(n, np.nan)
rot = np.full(n, np.nan)
for i, key in enumerate(keys):
    record = truth[key.split('/')[0]].get(key.split('/')[1])
    if record is None:
        continue
    T = np.asarray(record['T_WB'], float)
    err[i] = np.linalg.norm(t0[i][:3, 3] - T[:3, 3])
    rot[i] = np.degrees(np.arccos(np.clip((np.trace(t0[i][:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
sel = np.where(np.isfinite(err) & (err > a.threshold))[0]
frames = [dict(key=keys[i],
               scan_path='%s/NCLT/%s/velodyne_sync/%s.bin' % (DATA, keys[i].split('/')[0],
                                                             keys[i].split('/')[1]),
               err_t0=round(float(err[i]), 6), rot_t0_deg=round(float(rot[i]), 6),
               band=('5-8' if err[i] < 8 else ('8-20' if err[i] < 20 else '>20')))
          for i in sel]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(frames, open(a.frames, 'w'), indent=1)
np.savez_compressed(a.out, keys=np.array([keys[i] for i in sel]), t0=t0[sel], poses=poses[sel],
                    err_t0=err[sel], rot_t0=rot[sel], index=sel)
print('frames %d ; selected %d ; bands %s ; t0 MPE %.4f / %.4f ; MOE %.4f / %.4f' % (
    n, len(sel), collections.Counter([f['band'] for f in frames]),
    err[sel].mean(), np.median(err[sel]), rot[sel].mean(), np.median(rot[sel])))
print('wrote', a.out, 'and', a.frames)
