"""Snapshot of the project's rot_only, eps exposed (generated, do not edit).

Source: /home/zhangcy/projects/e2e-moe-20261020-v1/.backups/remove-rotation-20261022-182520/three_route_unified.py
Only change: the hardcoded acceptance margin 1e-6 -> `eps`.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def rot_only(tree, geom, q, T, eps=1e-6):
    """Rotation block: translation frozen, step accepted on a held-out half."""
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        W = val @ M[:3, :3].T + M[:3, 3]
        d = tree.query(W, k=1)[0]
        return float(np.minimum(d, 1.2).mean()), int((d < 0.6).sum())

    t = np.array(T, float)
    centre = t[:3, 3].copy()
    value, hit = obj(t)
    for _ in range(6):
        W = train @ t[:3, :3].T + centre
        dist, idx = tree.query(W, k=1)
        ok = dist < 1.2
        if ok.sum() < 30:
            break
        A, B = W[ok] - centre, geom[idx[ok]] - centre
        a, b = A.mean(0), B.mean(0)
        U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
        if s[1] < 1e-8:
            break
        corr = np.eye(3)
        corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ corr @ U.T
        vec = Rotation.from_matrix(R).as_rotvec()
        scale = min(1.0, np.deg2rad(0.5) / max(np.linalg.norm(vec), 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre
        nv, nh = obj(cand)
        if not (nv < value - eps and nh >= hit):
            break
        t, value, hit = cand, nv, nh
    return t


