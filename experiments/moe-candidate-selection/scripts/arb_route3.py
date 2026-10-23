"""Recover the route-3 hypothesis for all 1502 frames (mode=route3 replay).

results/three_route_merged.npz stores the *chosen* pose, so on the frames where
the arbiter accepted the visual hypothesis the route-3 output is not in the file.
This re-runs exactly the route-3 path (coarse surface, gates, fine solver,
rotation block) and writes the poses to the scratch dir.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
OUT = '/home/zhangcy/.dsh_scratch/arb_shared'
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)
import three_route_unified as U  # noqa: E402

U._OPT.update(mode='route3', gate='merged', gate2=-1.0, delta=0.05, rot=True,
              solver='adaptive', coarse_points=1000, fine_points=1500, threads=4)


def main():
    t0 = time.time()
    U.init()
    frames = U._G['frames']
    idx = list(range(len(frames)))
    with Pool(8, initializer=U.init) as p:
        res = p.map(U._w, idx, chunksize=4)
    res.sort(key=lambda r: r[0])
    poses = np.stack([r[1] for r in res])
    np.savez(os.path.join(OUT, 'route3_poses.npz'),
             keys=np.array([str(frames[i]['key']) for i in idx]),
             poses=poses)
    print('[route3] %d frames in %.1f s -> %s/route3_poses.npz'
          % (len(idx), time.time() - t0, OUT), flush=True)

    z = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
    stored = np.asarray(z['poses'], float)
    d = np.linalg.norm(poses[:, :3, 3] - stored[:, :3, 3], axis=1)
    same = np.abs(poses - stored).max(axis=(1, 2))
    print('[compare] frames where replayed route3 == stored chosen pose: %d / %d'
          % (int((same < 1e-12).sum()), len(d)), flush=True)
    print('[compare] max deviation on the %d differing frames: %.3f m'
          % (int((same >= 1e-12).sum()), d[same >= 1e-12].max() if (same >= 1e-12).any() else 0),
          flush=True)
    diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
    acc = np.array([bool(x.get('visual')) for x in diag])
    print('[compare] differing frames == diag-accepted set:',
          bool(((same >= 1e-12) == acc).all()), ' accepted', int(acc.sum()), flush=True)

    sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
    import score_unified as SU
    w = SU.load_world()
    t = SU.pos_err(poses, w['hard_gt'])
    r = SU.rot_deg(poses[:, :3, :3], w['hard_gt'][:, :3, :3])
    print('[route3-only] MPE %.4f MOE %.4f ok1/2 %d >5m %d >10m %d'
          % (t.mean(), r.mean(), int(((t <= 1) & (r <= 2)).sum()),
             int((t > 5).sum()), int((t > 10).sum())), flush=True)


if __name__ == '__main__':
    main()
