"""EXP N: single-frame rotation from the image, with the translation held.

Diagnosis says the LiDAR cost carries no rotation gradient at the 1 deg scale.
The image does: once the camera centre is known (the geometric branch supplies it
to 0.11 m), every 2D-3D correspondence becomes a bearing pair (world direction ->
camera direction), and the rotation alone is a Wahba problem with a closed-form
solution.  That is far better conditioned than 6-dof PnP, which has to trade
rotation against translation.

This runs the project's own visual branch unchanged (retrieval, SuperPoint,
LightGlue, per-reference LiDAR projection for the 3D points) and replaces ONLY the
pose solver: robust (graduated) rotation-only Kabsch seeded from the front-end
rotation.

Pilot mode by default (30 frames); set N to run more.
"""
import json
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')   # GPUs 0-2 and 4-7 are busy
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, PROJECT)
from visual_branch import VisualBranch      # noqa: E402

N_FRAMES = int(os.environ.get('N', '30'))
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expN_rot_only.json'


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def kabsch(b, d, w):
    """R minimising sum_i w_i |R b_i - d_i|^2 (b, d are n x 3 unit vectors)."""
    H = (b * w[:, None]).T @ d                       # sum w b d^T
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _selftest():
    rng = np.random.default_rng(0)
    def randR():
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    R = randR()
    b = rng.normal(size=(50, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    d = b @ R.T
    d += 0.002 * rng.normal(size=d.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    R2 = kabsch(b, d, np.ones(50))
    err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R2) - 1) / 2, -1, 1)))
    print('kabsch self-test: recovered rotation off by %.5f deg' % err)
    assert err < 0.1, 'kabsch sign convention is wrong'


class RotationOnlyBranch(VisualBranch):
    """Same matching, same 3D binding, only the pose step is rotation-only."""

    def solve(self, world, pixels, query_ids):
        pose_prior = getattr(self, 'prior_pose', None)
        if pose_prior is None:
            return None
        K = self.C.K
        T0 = np.asarray(pose_prior, dtype=np.float64)
        C_cw0 = np.linalg.inv(T0 @ self.T_BC)
        R0 = C_cw0[:3, :3].copy()
        centre = -R0.T @ C_cw0[:3, 3]

        world = np.asarray(world, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        ids = np.asarray(query_ids, dtype=np.int32)
        if len(world) < 6:
            return None
        v = world - centre[None, :]
        n = np.linalg.norm(v, axis=1)
        keep = n > 1e-6
        v, pixels, ids = v[keep] / n[keep, None], pixels[keep], ids[keep]
        if len(v) < 6:
            return None
        rays = np.linalg.inv(K) @ np.column_stack([pixels, np.ones(len(pixels))]).T
        rays /= np.linalg.norm(rays, axis=0)
        d = rays.T                                        # n x 3 camera-frame bearings

        R = R0.copy()
        w = np.ones(len(v))
        for gate in (10.0, 7.0, 5.0, 3.5, 2.5, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0):
            pred = v @ R.T
            ang = np.degrees(np.arccos(np.clip((pred * d).sum(1), -1, 1)))
            w = (ang <= gate).astype(float)
            if w.sum() < 6:
                break
            R = kabsch(v, d, w)
        # never let the image move the attitude more than this from the prior
        drift = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
        pred = v @ R.T
        ang = np.degrees(np.arccos(np.clip((pred * d).sum(1), -1, 1)))
        inliers = np.flatnonzero(ang <= 12.0)
        if drift > 20.0 or len(inliers) < 6:
            return None

        C_cw = np.eye(4)
        C_cw[:3, :3] = R
        C_cw[:3, 3] = -R @ centre
        pose = np.linalg.inv(C_cw) @ np.linalg.inv(self.T_BC)
        pose[:3, 3] = T0[:3, 3]                          # translation stays geometric

        camera = world @ R.T + C_cw[:3, 3]
        proj = camera @ K.T
        residual = np.linalg.norm(proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12) - pixels,
                                  axis=1)
        ok = np.flatnonzero((residual <= 12.0) & (camera[:, 2] > 0.5))
        if len(ok) < 6:
            return None
        uniq = np.unique(ids[ok])
        cells = np.floor(pixels[ok] / [808 / 4.0, 616 / 4.0]).astype(int).clip(0, 3)
        coverage = len(np.unique(cells[:, 1] * 4 + cells[:, 0]))
        cfg = self.pnp_cfg
        return dict(pose=pose, inlier_indices=ok, unique_query_inliers=int(len(uniq)),
                    inlier_ratio=float(len(uniq) / max(1, len(np.unique(ids)))),
                    median_reprojection_px=float(np.median(residual[ok])),
                    coverage_cells=int(coverage),
                    qualified=bool(len(uniq) >= cfg['min_inliers']
                                   and coverage >= cfg['min_4x4_cells']
                                   and len(uniq) / max(1, len(np.unique(ids)))
                                   >= cfg['min_inlier_ratio']))


def main():
    _selftest()
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        d = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
    Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
    keys = [str(k) for k in Z['keys']]
    t0 = np.asarray(Z['t0'], float)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)

    rng = np.random.default_rng(11)
    bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 2.0), (2.0, 1e9)]
    pick = []
    per = max(1, N_FRAMES // len(bands))
    for lo, hi in bands:
        idx = np.where((e0 >= lo) & (e0 < hi))[0]
        if len(idx):
            pick.append(rng.choice(idx, min(per, len(idx)), replace=False))
    pick = np.sort(np.concatenate(pick))
    print('frames %d (%s)' % (len(pick), np.bincount(np.digitize(e0[pick], [0.2, 0.5, 2.0]))))

    branch = RotationOnlyBranch(device='cuda')
    print('branch ready')

    rows = []
    t_start = time.time()
    for j, i in enumerate(pick):
        key = keys[i]
        date, stamp = key.split('/')
        image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
        if not os.path.exists(image_path):
            print('  no image for %s' % key)
            continue
        branch.prior_pose = t0[i]
        t1 = time.time()
        hyp = branch.hypothesize(key, image_path, budget=1, pose_prior=t0[i])
        dt = time.time() - t1
        if hyp is None:
            rows.append(dict(key=key, ok=False, mpe0=e0[i],
                             moe0=float(ra(t0[i, :3, :3], gt[i, :3, :3])),
                             seconds=dt))
            print('  %-30s no hypothesis  (%.1fs)' % (key, dt), flush=True)
            continue
        P = np.asarray(hyp['pose'], float)
        rows.append(dict(key=key, ok=True, mpe0=float(e0[i]),
                         moe0=float(ra(t0[i, :3, :3], gt[i, :3, :3])),
                         mpe1=float(np.linalg.norm(P[:3, 3] - gt[i, :3, 3])),
                         moe1=float(ra(P[:3, :3], gt[i, :3, :3])),
                         drift=float(ra(t0[i, :3, :3], P[:3, :3])),
                         inliers=int(hyp['detail']['scores'][0]['inliers']),
                         score=float(hyp['score']),
                         channel=hyp['channel'], seconds=dt))
        r = rows[-1]
        print('  %-30s inl %3d  drift %5.2f  MOE %.2f -> %.2f   (%.1fs)'
              % (key, r['inliers'], r['drift'], r['moe0'], r['moe1'], dt), flush=True)

    ok = [r for r in rows if r['ok']]
    print('\n## pilot result  (%d/%d frames produced a pose, %.0f s total)'
          % (len(ok), len(rows), time.time() - t_start))
    if ok:
        m0 = np.array([r['moe0'] for r in ok])
        m1 = np.array([r['moe1'] for r in ok])
        p0 = np.array([r['mpe0'] for r in ok])
        p1 = np.array([r['mpe1'] for r in ok])
        print('   MOE  %.3f -> %.3f   (improved %d, worsened %d)'
              % (m0.mean(), m1.mean(), int((m1 < m0).sum()), int((m1 > m0).sum())))
        print('   MPE  %.3f -> %.3f' % (p0.mean(), p1.mean()))
        print('   median MOE %.3f -> %.3f ; rotation drift from the prior: median %.2f deg'
              % (np.median(m0), np.median(m1), np.median([r['drift'] for r in ok])))
        print('   inliers median %d ; seconds/frame median %.2f'
              % (int(np.median([r['inliers'] for r in ok])),
                 np.median([r['seconds'] for r in rows])))
        for lo, hi, name in ((0, .2, 'bulk'), (.2, .5, 'mid'), (.5, 2, 'hard'), (2, 1e9, 'tail')):
            sel = [r for r in ok if lo <= r['mpe0'] < hi]
            if sel:
                print('   %-5s n=%2d  MOE %.2f -> %.2f'
                      % (name, len(sel), np.mean([r['moe0'] for r in sel]),
                         np.mean([r['moe1'] for r in sel])))
    with open(OUT, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
