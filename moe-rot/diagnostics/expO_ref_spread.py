"""EXP O: why does the image-rotation disagree with the front end?

The 2D-3D points are bound per *reference* frame (a keypoint takes the LiDAR return
that lands within 3.5 px of it in that reference's projection).  If the reference
poses carry attitude error, every reference contributes a rigidly wrong 3D set and
its own rotation solution lands on that reference's error.

So: solve the rotation separately from each reference, print the spread, and
compare with the mixed solution.  If the per-reference solutions scatter by several
degrees, the ceiling of any image-based rotation on this library is the library's
own attitude accuracy, not the matcher.
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

from collections import OrderedDict

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, PROJECT)
import visual_branch as VB      # noqa: E402

N_FRAMES = int(os.environ.get('N', '8'))


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def kabsch(b, d, w):
    H = (b * w[:, None]).T @ d
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


class Branch(VB.VisualBranch):
    def __init__(self, *a, **kw):
        kw.setdefault('device', 'cuda')
        super(Branch, self).__init__(*a, **kw)
        self._features = OrderedDict()

    def _cached_features(self, image_path, tag):
        cache = self._features
        if tag in cache:
            value = cache.pop(tag)
            cache[tag] = value
            return value
        value = self.extract(image_path)
        cache[tag] = value
        while len(cache) > 64:
            cache.popitem(last=False)
        return value


def rot_from_corr(world, pixels, ids, T0, T_BC, K):
    C_cw0 = np.linalg.inv(np.asarray(T0) @ T_BC)
    R0 = C_cw0[:3, :3].copy()
    centre = -R0.T @ C_cw0[:3, 3]
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
    C_cw = np.eye(4)
    C_cw[:3, :3] = R
    C_cw[:3, 3] = -R @ centre
    pose = np.linalg.inv(C_cw) @ np.linalg.inv(T_BC)
    pose[:3, 3] = np.asarray(T0)[:3, 3]
    return dict(pose=pose, n_corr=int(len(v)), n_inl=int(inl.sum()),
                resid=float(np.median(ang[inl])) if inl.any() else 9e9,
                drift=float(ra(R0, R)))


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
    pick = rng.choice(np.where(e0 < 0.5)[0], N_FRAMES, replace=False)
    branch = Branch(device='cuda')
    K = branch.C.K

    out = []
    for i in pick:
        key = keys[i]
        date, stamp = key.split('/')
        image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
        T0 = t0[i]
        refs, _retrieval = branch.retrieve(image_path, T0)
        q_model, candidates, _, _ = branch._match_frame(image_path, refs)
        by_ref = {}
        for qi, s, ri, aid in candidates:
            by_ref.setdefault(ri, ([], []))
            by_ref[ri][0].append(qi)
            by_ref[ri][1].append(aid)
        print('\n=== %s   refs %s   candidates %d   MOE(t0) %.2f'
              % (key, refs, len(candidates), ra(T0[:3, :3], gt[i, :3, :3])))
        per = []
        for ri, (ids, aid) in by_ref.items():
            ids = np.array(ids)
            aid = np.array(aid)
            table = branch._projection(branch.references[ri])
            world = table['world'][aid]
            pixels = branch.C.model_to_raw(branch.matrix, q_model[ids])
            sol = rot_from_corr(world, pixels, ids, T0, branch.T_BC, K)
            if sol is None:
                print('   ref %d: too few' % ri)
                continue
            moe = ra(sol['pose'][:3, :3], gt[i, :3, :3])
            per.append(sol['drift'])
            print('   ref %d: n_corr %3d  inl %2d  resid %.2f px  drift %5.2f  MOE %.2f'
                  % (ri, sol['n_corr'], sol['n_inl'], sol['resid'], sol['drift'], moe))
        capped = branch._cap(candidates)
        if len(capped) >= 8:
            ids = np.array([c[0] for c in capped])
            world = np.array([branch._projection(branch.references[c[2]])['world'][c[3]]
                              for c in capped])
            pixels = branch.C.model_to_raw(branch.matrix, q_model[ids])
            sol = rot_from_corr(world, pixels, ids, T0, branch.T_BC, K)
            if sol is not None:
                moe = ra(sol['pose'][:3, :3], gt[i, :3, :3])
                print('   ALL : n_corr %3d  inl %2d  resid %.2f px  drift %5.2f  MOE %.2f'
                      % (sol['n_corr'], sol['n_inl'], sol['resid'], sol['drift'], moe))
                per.append(sol['drift'])
        if len(per) >= 2:
            per = np.array(per)
            print('   spread of the per-reference drifts: std %.2f deg, range %.2f'
                  % (per.std(), per.max() - per.min()))
        out.append(dict(key=key, drifts=per))
    print('\n## summary over %d frames' % len(out))
    print('   mean per-frame spread of per-reference solutions: %.2f deg'
          % np.mean([np.std(o['drifts']) for o in out if len(o['drifts']) > 1]))


if __name__ == '__main__':
    main()
