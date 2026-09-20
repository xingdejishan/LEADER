"""Per-view explicit quality cues (spec section 7). All numpy, no GT.

q_ij = [range, border_dist, depth_margin, depth_evidence_valid,
        sharpness, contrast, saturation, cross_view_consistency,
        consistency_valid]
"""
try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # minimal fallbacks so quality.py works without opencv
    _HAS_CV2 = False

import numpy as np


def _gray(im):
    if _HAS_CV2:
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return im[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _laplacian_var_patch(g, lap, v0, v1, u0, u1):
    # 4-neighbour Laplacian on the patch, variance over it
    p = g[v0:v1, u0:u1]
    if _HAS_CV2:
        return float(lap[v0:v1, u0:u1].var())
    lapp = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1])
    if lapp.size < 2:
        return 0.0
    return float(lapp.var())



def quality_vector(cam_xyz, uv, valid, depth_margin, depth_evidence,
                   image, desc_feats, config):
    """Compute (N, 6, Dq) quality features.

    cam_xyz: (N,6,3) camera-frame points | uv: (N,6,2) pixels
    valid: (N,6) bool after occlusion filtering
    depth_margin/evidence: (N,6) from visibility.zbuffer
    image: list of 6 (H,W,3) uint8 | desc_feats: (N,6,128) sampled features
    """
    qcfg = config["quality"]
    W, H = config["image"]["width"], config["image"]["height"]
    n = cam_xyz.shape[0]
    n_cam = cam_xyz.shape[1]
    dq = 9
    q = np.zeros((n, n_cam, dq), dtype=np.float32)

    rng = np.linalg.norm(cam_xyz, axis=-1)  # (N,6)
    q[:, :, 0] = np.log1p(np.where(np.isfinite(rng), rng, 0.0))

    border = np.stack([
        np.minimum.reduce([uv[:, j, 0], W - 1 - uv[:, j, 0],
                           uv[:, j, 1], H - 1 - uv[:, j, 1]])
        for j in range(n_cam)], axis=1)
    q[:, :, 1] = np.clip(border / (min(W, H) / 2.0), 0, 1)

    q[:, :, 2] = np.clip(depth_margin / 10.0, 0, 2)  # normalized depth margin
    q[:, :, 3] = depth_evidence.astype(np.float32)

    patch = qcfg["sharpness_patch"] // 2
    gray = [_gray(im) for im in image]
    low, high = qcfg["saturation_low"], qcfg["saturation_high"]
    lap = [None] * n_cam
    if _HAS_CV2 and n_cam > 0:
        lap = [cv2.Laplacian(g, cv2.CV_32F) for g in gray]
    for j in range(n_cam):
        sel = np.where(valid[:, j])[0]
        if sel.size == 0:
            continue
        im, g = image[j], gray[j]
        u = np.clip(uv[sel, j, 0].astype(int), 0, W - 1)
        v = np.clip(uv[sel, j, 1].astype(int), 0, H - 1)
        sharp = np.empty(sel.shape, np.float32)
        contrast = np.empty(sel.shape, np.float32)
        satur = np.empty(sel.shape, np.float32)
        lap_j = lap[j]
        for idx, (uu, vv) in enumerate(zip(u, v)):
            v0, v1 = max(0, vv - patch), min(H, vv + patch + 1)
            u0, u1 = max(0, uu - patch), min(W, uu + patch + 1)
            p = g[v0:v1, u0:u1]
            sharp[idx] = _laplacian_var_patch(g, lap_j, v0, v1, u0, u1)
            contrast[idx] = float(p.std())
            pc = im[v0:v1, u0:u1]
            satur[idx] = float(((pc.min(-1) < low) | (pc.max(-1) > high)).mean())
        q[sel, j, 4] = np.log1p(sharp)
        q[sel, j, 5] = contrast / 128.0
        q[sel, j, 6] = satur

    # cross-view descriptor consistency (median cosine vs other valid views)
    eps = 1e-8
    fn = desc_feats / (np.linalg.norm(desc_feats, axis=-1, keepdims=True) + eps)
    for i in sel if sel.size else range(0):
        pass
    for i in range(n):
        js = np.where(valid[i])[0]
        if js.size < 2:
            continue
        sims = fn[i, js] @ fn[i, js].T  # (k,k)
        for idx, j in enumerate(js):
            others = np.delete(sims[idx], idx)
            q[i, j, 7] = float(np.median(others))
        q[i, js, 8] = 1.0
    return q


