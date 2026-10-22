"""EXP N2: rotation-only image solve, with matching variants.

Same idea as expN (image gives the rotation, geometry keeps the translation) but
(a) fixes the branch's dict-cache crash without touching its source,
(b) reports correspondence quality so the failure mode is visible,
(c) sweeps the matcher variants that exist in this project: the frozen LightGlue
    vs the V8 fine-tuned one, few vs many keypoints and references.

Acceptance is selective on single-frame evidence only (inlier count and residual);
frames with weak support keep the front-end rotation.
"""
import json
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

from collections import OrderedDict

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, PROJECT)
import visual_branch as VB      # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expN2_variants.json'
N_FRAMES = int(os.environ.get('N', '24'))
MIN_INLIERS = int(os.environ.get('MINI', '24'))
MAX_RESIDUAL = float(os.environ.get('MAXRES', '3.0'))


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def kabsch(b, d, w):
    H = (b * w[:, None]).T @ d
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


class RotationOnlyBranch(VB.VisualBranch):
    def __init__(self, *a, **kw):
        kw.setdefault('device', 'cuda')
        super(RotationOnlyBranch, self).__init__(*a, **kw)
        self._features = OrderedDict()      # the base uses popitem(last=False)
        self.diag = []

    def retrieve(self, image_path, pose_prior=None, channels=None):
        return super(RotationOnlyBranch, self).retrieve(
            image_path, pose_prior, channels or self.channels)

    def solve(self, world, pixels, query_ids):
        T0 = getattr(self, 'prior_pose', None)
        if T0 is None:
            return None
        K = self.C.K
        T0 = np.asarray(T0, dtype=np.float64)
        C_cw0 = np.linalg.inv(T0 @ self.T_BC)
        R0 = C_cw0[:3, :3].copy()
        centre = -R0.T @ C_cw0[:3, 3]

        world = np.asarray(world, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        ids = np.asarray(query_ids, dtype=np.int32)
        v = world - centre[None, :]
        n = np.linalg.norm(v, axis=1)
        keep = n > 1e-6
        v, pixels, ids = v[keep] / n[keep, None], pixels[keep], ids[keep]
        if len(v) < 8:
            return None
        rays = np.linalg.inv(K) @ np.column_stack([pixels, np.ones(len(pixels))]).T
        rays /= np.linalg.norm(rays, axis=0)
        d = rays.T

        R = R0.copy()
        for gate in (6.0, 4.0, 2.5, 1.5, 1.0, 1.0, 1.0):
            pred = v @ R.T
            ang = np.degrees(np.arccos(np.clip((pred * d).sum(1), -1, 1)))
            w = (ang <= gate).astype(float)
            if w.sum() < 8:
                break
            R = kabsch(v, d, w)
        pred = v @ R.T
        ang = np.degrees(np.arccos(np.clip((pred * d).sum(1), -1, 1)))
        inl = ang <= 2.0
        if inl.sum() < 8:
            return None
        self.diag.append(dict(n_corr=int(len(v)), n_inl=int(inl.sum()),
                              resid=float(np.median(ang[inl])),
                              resid_all=float(np.median(ang))))

        C_cw = np.eye(4)
        C_cw[:3, :3] = R
        C_cw[:3, 3] = -R @ centre
        pose = np.linalg.inv(C_cw) @ np.linalg.inv(self.T_BC)
        pose[:3, 3] = T0[:3, 3]
        camera = world @ R.T + C_cw[:3, 3]
        proj = camera @ K.T
        residual = np.linalg.norm(proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12) - pixels,
                                  axis=1)
        ok = np.flatnonzero((residual <= 12.0) & (camera[:, 2] > 0.5))
        if len(ok) < 8:
            return None
        uniq = np.unique(ids[ok])
        cells = np.floor(pixels[ok] / [808 / 4.0, 616 / 4.0]).astype(int).clip(0, 3)
        coverage = len(np.unique(cells[:, 1] * 4 + cells[:, 0]))
        return dict(pose=pose, inlier_indices=ok, unique_query_inliers=int(len(uniq)),
                    inlier_ratio=float(len(uniq) / max(1, len(np.unique(ids)))),
                    median_reprojection_px=float(np.median(residual[ok])),
                    coverage_cells=int(coverage), qualified=True,
                    resid_angle=float(np.median(ang[inl])))


def main():
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
    print('frames %d' % len(pick))

    variants = [
        dict(name='frozen/256/5', arm='frozen_noprune', budget=256, channels=5),
        dict(name='v8/256/5', arm='v8_noprune', budget=256, channels=5),
        dict(name='v8/1024/10', arm='v8_noprune', budget=1024, channels=10),
    ]
    all_rows = {}
    for variant in variants:
        print('\n### %s' % variant['name'])
        branch = RotationOnlyBranch(device='cuda', arm=variant['arm'],
                                    keypoint_budget=variant['budget'])
        branch.channels = variant['channels']
        rows = []
        t_start = time.time()
        for i in pick:
            key = keys[i]
            date, stamp = key.split('/')
            image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
            branch.prior_pose = t0[i]
            branch.diag = []
            try:
                hyp = branch.hypothesize(key, image_path, budget=1, pose_prior=t0[i])
            except Exception as exc:                          # noqa: BLE001
                print('  %s failed: %s' % (key, exc))
                hyp = None
            moe0 = float(ra(t0[i, :3, :3], gt[i, :3, :3]))
            row = dict(key=key, moe0=moe0, mpe0=float(e0[i]), ok=hyp is not None)
            if hyp is not None:
                P = np.asarray(hyp['pose'], float)
                row.update(moe1=float(ra(P[:3, :3], gt[i, :3, :3])),
                           drift=float(ra(t0[i, :3, :3], P[:3, :3])),
                           inliers=int(hyp['detail']['scores'][0]['inliers']),
                           corr=branch.diag[-1] if branch.diag else None)
                accept = (row['inliers'] >= MIN_INLIERS
                          and (row['corr'] or {}).get('resid', 9e9) <= MAX_RESIDUAL)
                row['accept'] = bool(accept)
                row['moe_final'] = row['moe1'] if accept else moe0
            else:
                row['accept'] = False
                row['moe_final'] = moe0
            rows.append(row)
            print('  %-28s MPE0 %.2f  MOE %.2f -> %s  inl %s  resid %s'
                  % (key, row['mpe0'], moe0,
                     ('%.2f' % row['moe1']) if row['ok'] else '--',
                     row.get('inliers', '--'),
                     ('%.2f' % row['corr']['resid']) if row.get('corr') else '--'),
                  flush=True)
        ok = [r for r in rows if r['ok']]
        m0 = np.array([r['moe0'] for r in rows])
        mf = np.array([r['moe_final'] for r in rows])
        print('  -> %d/%d produced a pose, %.0f s' % (len(ok), len(rows), time.time() - t_start))
        print('  -> MOE %.3f -> %.3f (selective, %d accepted) ; raw proposal MOE %.3f'
              % (m0.mean(), mf.mean(), int(sum(1 for r in rows if r['accept'])),
                 np.mean([r['moe1'] for r in ok]) if ok else float('nan')))
        all_rows[variant['name']] = rows
        del branch
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    with open(OUT, 'w') as fh:
        json.dump(all_rows, fh, indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
