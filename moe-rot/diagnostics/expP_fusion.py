"""EXP P: fuse the front-end attitude with the image attitude (single frame).

The image rotation is not better than the front end on its own (its 3D points
inherit the reference poses' attitude error), but it is an *independent* second
measurement of the same quantity.  Two independent measurements of one rotation
average: R = R_prior * Exp(lambda * log(R_prior^T R_image)).  lambda = 0 keeps the
front end, lambda = 1 takes the image, and the useful value is in between.

Everything here is per frame: one image, one scan, one pose.  The translation is
never touched.
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

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expP_fusion.json'
N_BULK = int(os.environ.get('NB', '60'))
N_MID = int(os.environ.get('NM', '30'))
N_HARD = int(os.environ.get('NH', '30'))


def log_so3(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


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
        self._projections = OrderedDict()

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


def rotation_from(world, pixels, T0, T_BC, K):
    C_cw0 = np.linalg.inv(np.asarray(T0) @ T_BC)
    R0 = C_cw0[:3, :3].copy()
    centre = -R0.T @ C_cw0[:3, 3]
    v = np.asarray(world, float) - centre[None, :]
    n = np.linalg.norm(v, axis=1)
    keep = n > 1e-6
    v, pixels = v[keep] / n[keep, None], np.asarray(pixels, float)[keep]
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
    return dict(R=R, n_corr=int(len(v)), n_inl=int(inl.sum()),
                resid=float(np.median(ang[inl])),
                delta=log_so3(R0.T @ R))


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

    rng = np.random.default_rng(4242)
    pick = np.concatenate([
        rng.choice(np.where(e0 < 0.2)[0], N_BULK, replace=False),
        rng.choice(np.where((e0 >= 0.2) & (e0 < 1.0))[0], N_MID, replace=False),
        rng.choice(np.where(e0 >= 1.0)[0], N_HARD, replace=False),
    ])
    print('frames %d' % len(pick), flush=True)
    branch = Branch(device='cuda')
    K = branch.C.K

    rows = []
    for n, i in enumerate(pick):
        key = keys[i]
        date, stamp = key.split('/')
        image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
        T0 = t0[i]
        row = dict(key=key, i=int(i), mpe0=float(e0[i]),
                   moe0=float(ra(T0[:3, :3], gt[i, :3, :3])), ok=False)
        try:
            refs, _ = branch.retrieve(image_path, T0)
            q_model, candidates, _, _ = branch._match_frame(image_path, refs)
            capped = branch._cap(candidates)
            sol = None
            if len(capped) >= 8:
                ids = np.array([c[0] for c in capped])
                world = np.array([branch._projection(branch.references[c[2]])['world'][c[3]]
                                  for c in capped])
                pixels = branch.C.model_to_raw(branch.matrix, q_model[ids])
                sol = rotation_from(world, pixels, T0, branch.T_BC, K)
            if sol is None:
                best = None
                by_ref = {}
                for qi, s, ri, aid in candidates:
                    by_ref.setdefault(ri, ([], []))
                    by_ref[ri][0].append(qi)
                    by_ref[ri][1].append(aid)
                for ri, (ids, aid) in by_ref.items():
                    if len(ids) < 8:
                        continue
                    table = branch._projection(branch.references[ri])
                    cand = rotation_from(table['world'][np.array(aid)],
                                         branch.C.model_to_raw(branch.matrix,
                                                               q_model[np.array(ids)]),
                                         T0, branch.T_BC, K)
                    if cand and (best is None or cand['n_inl'] > best['n_inl']):
                        best = cand
                sol = best
            if sol is not None:
                row.update(ok=True, delta=np.asarray(sol['delta']).tolist(),
                           n_corr=sol['n_corr'], n_inl=sol['n_inl'],
                           resid=sol['resid'],
                           moe_image=float(ra(sol['R'], gt[i, :3, :3])))
        except Exception as exc:                     # noqa: BLE001
            row['error'] = str(exc)
        rows.append(row)
        if (n + 1) % 10 == 0:
            good = sum(1 for r in rows if r['ok'])
            print('  %d/%d  usable %d' % (n + 1, len(rows), good), flush=True)

    ok = [r for r in rows if r['ok']]
    print('\n## coverage: %d/%d frames produced an image rotation' % (len(ok), len(rows)))
    if ok:
        print('   inliers median %d (p10 %d) ; residual median %.2f px'
              % (int(np.median([r['n_inl'] for r in ok])),
                 int(np.percentile([r['n_inl'] for r in ok], 10)),
                 np.median([r['resid'] for r in ok])))
        m0 = np.array([r['moe0'] for r in ok])
        mi = np.array([r['moe_image'] for r in ok])
        print('   MOE prior %.3f  image %.3f  (image better on %d/%d frames)'
              % (m0.mean(), mi.mean(), int((mi < m0).sum()), len(ok)))
        print('\n   %-6s %8s %8s %8s %8s' % ('lambda', 'MOE', 'median', 'ok2deg%', 'help/hurt'))
        deltas = [np.array(r['delta']) for r in ok]
        G = np.stack([gt[r['i'], :3, :3] for r in ok])
        R0 = np.stack([t0[r['i'], :3, :3] for r in ok])
        for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            Rf = np.stack([R0[j] @ exp_so3(lam * deltas[j]) for j in range(len(ok))])
            e = ra(Rf, G)
            print('   %-6.1f %8.3f %8.3f %8.1f %5d/%d'
                  % (lam, e.mean(), np.median(e), 100 * (e <= 2).mean(),
                     int((e < m0).sum()), int((e > m0).sum())))
    with open(OUT, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
