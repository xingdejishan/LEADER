"""Robust rotation averaging over the visual candidates, versus taking the best one.

Every frame already has several PnP candidates.  The shipped rule keeps the single
highest-scoring one; the literature on pose ambiguity instead fuses the hypotheses
(robust rotation averaging: Hartley et al., IJCV 2013; clique-constrained averaging,
ICRA 2020; multi-hypothesis SLAM, 2021).

Fused with an IRLS chordal L1 mean over the candidates that pass the LiDAR
verification, so a single bad candidate cannot drag the answer.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def geodesic(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def robust_mean(mats, weights, iters=8):
    R = mats[int(np.argmax(weights))].copy()
    for _ in range(iters):
        d = np.array([geodesic(R, M) for M in mats])
        w = weights / np.maximum(d, 1.0)
        M = np.einsum('n,nij->ij', w, mats)
        U, _, Vt = np.linalg.svd(M)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
    return R


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)

    best_err, fuse_err, n_c, used = [], [], [], []
    for i in range(len(gt)):
        rec = json.load(open(VERIFY % i))['poses']
        mats, wts = [], []
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_') or not row.get('geometry_verified'):
                continue
            P = np.asarray(row['pose'], float)
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            mats.append(P[:3, :3])
            wts.append(max(float(sc), 1e-3))
            e = geodesic(P[:3, :3], gt[i][:3, :3])
            if best is None or sc > best[0]:
                best = (float(sc), e)
        if best is None:
            continue
        n_c.append(len(mats))
        used.append(i)
        best_err.append(best[1])
        if len(mats) >= 2:
            R = robust_mean(np.array(mats), np.array(wts))
            fuse_err.append(geodesic(R, gt[i][:3, :3]))
        else:
            fuse_err.append(best[1])

    b = np.array(best_err)
    f = np.array(fuse_err)
    print('frames with a verified candidate: %d   (median candidates %.0f)'
          % (len(b), np.median(n_c)))
    print('rotation error  best-by-score   : median %.2f  mean %.2f' % (np.median(b), b.mean()))
    print('rotation error  robust averaging: median %.2f  mean %.2f' % (np.median(f), f.mean()))
    print('averaging better on %d of %d frames (%.0f%%)'
          % (int((f < b).sum()), len(b), 100 * (f < b).mean()))
    for k in (1, 2, 3, 5):
        m = np.array(n_c) >= k
        if m.sum():
            print('  frames with >=%d candidates: %4d   best %.2f -> fused %.2f'
                  % (k, int(m.sum()), b[m].mean(), f[m].mean()))


if __name__ == '__main__':
    main()
