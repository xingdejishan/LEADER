"""GLACE as an independent camera local-to-global correspondence branch.

Interface split (fixed by design):

    LEADER: p_i^B -> P_i^W   (3D local -> 3D world, SC2-PCR)
    GLACE:  u_j    -> P_j^W  (2D pixel -> 3D world, scene coordinates)

The adapter never discards `scene_coordinates_B3HW`: every 8x8 cell centre of
the output becomes one camera correspondence (u_j, P_j^W) via the vendor pixel
grid (u = OUTPUT_SUBSAMPLE * (x + 0.5)), and the same tensor feeds the pose
solver (OpenCV PnP-RANSAC + LM in v1; DSAC* C++ stays untouched per the
first-version rule). Pose conventions follow the fusion module:

    T_WC  camera -> world  (GLACE/DSAC* `pose` convention)
    T_WB  body   -> world  = T_WC @ inv(T_BC),  T_BC: camera -> body
"""
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class GLACEOutput:
    T_WC: Optional[np.ndarray]
    T_WB: Optional[np.ndarray]
    uv: np.ndarray
    xyz_world: np.ndarray
    K: np.ndarray
    inlier_count: Optional[int]
    inlier_mask: Optional[np.ndarray]
    image_size_hw: Tuple[int, int]
    diagnostics: dict = field(default_factory=dict)


def pixel_grid_uv(subsample: int, height_cells: int, width_cells: int) -> np.ndarray:
    """Cell-centre pixel coordinates, matching vendor `get_pixel_grid`:
    u = subsample * (x + 0.5), v = subsample * (y + 0.5). Returns [Hc*Wc, 2] (u, v)."""
    u = subsample * (np.arange(width_cells, dtype=np.float64) + 0.5)
    v = subsample * (np.arange(height_cells, dtype=np.float64) + 0.5)
    grid = np.stack(np.meshgrid(u, v), axis=-1)
    return grid.reshape(-1, 2)


def glace_to_leader_frame(T_WC: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """T_WB = T_WC @ inv(T_BC) with T_BC: camera -> body (fusion convention)."""
    T_WC = np.asarray(T_WC, dtype=float)
    T_BC = np.asarray(T_BC, dtype=float)
    if T_WC.shape != (4, 4) or T_BC.shape != (4, 4):
        raise ValueError("Expected two 4x4 poses")
    return T_WC @ np.linalg.inv(T_BC)


def solve_pose_pnp(uv, xyz_world, K, *, reproj_error_px=4.0, iterations=1000,
                   confidence=0.999, min_inliers=8, seed=2089, refine_lm=True):
    """OpenCV PnP-RANSAC (EPNP) + LM refine on scene coordinates.

    Returns (T_WC, inlier_mask_over_full_cells, inlier_count, diagnostics).
    T_WC is None when no accepted solution exists."""
    import cv2

    uv = np.asarray(uv, dtype=np.float64)
    xyz = np.asarray(xyz_world, dtype=np.float64)
    good = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
    pix, pts = uv[good], xyz[good]
    if len(pix) < max(6, min_inliers):
        return None, None, 0, {"solved": False, "reason": "too_few_finite_cells"}
    cv2.setRNGSeed(seed)
    try:
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(pts), np.ascontiguousarray(pix),
            np.asarray(K, dtype=np.float64), None,
            iterationsCount=iterations, reprojectionError=reproj_error_px,
            confidence=confidence, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error as exc:
        return None, None, 0, {"solved": False, "reason": f"cv2_error: {exc}"}
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, None, 0, {"solved": False, "reason": "ransac_no_inliers"}
    ix = inliers.ravel()
    if refine_lm and len(ix) >= 6:
        rv, tv = cv2.solvePnPRefineLM(pts[ix], pix[ix], K, None, rv, tv)
    T_CW = np.eye(4)  # cv2 returns world -> camera
    T_CW[:3, :3] = cv2.Rodrigues(rv)[0]
    T_CW[:3, 3] = tv.ravel()
    T_WC = np.linalg.inv(T_CW)
    mask = np.zeros(uv.shape[0], dtype=bool)
    mask[np.flatnonzero(good)[ix]] = True
    return T_WC, mask, int(len(ix)), {"solved": True, "ransac_inliers": int(len(ix))}


class GLACEAdapter:
    """Wraps the vendor ACE/GLACE Regressor. Images must already be resized to the
    training resolution; K must match that same pixel size."""

    IMAGE_SUBSAMPLE = 8

    def __init__(self, vendor_dir, head_path, encoder_path=None, T_BC=None,
                 device="cuda", global_feature_fn: Optional[Callable] = None):
        import json
        config_path = Path(head_path).parent / 'config.json'
        if config_path.exists():
            protocol = json.loads(config_path.read_text()).get('global_feature_protocol')
            if protocol == 'official_rgb_r2former_480x640':
                if getattr(global_feature_fn, 'protocol', None) != protocol:
                    raise ValueError('This head requires RGB global features; the legacy grayscale extractor is incompatible')
        vendor_dir = Path(vendor_dir)
        sys.path.insert(0, str(vendor_dir))
        import torch
        from ace_network import Regressor

        self.torch = torch
        encoder = Path(encoder_path) if encoder_path else vendor_dir / "ace_encoder_pretrained.pt"
        encoder_state = torch.load(encoder, map_location="cpu")
        head_state = torch.load(head_path, map_location="cpu")
        self.regressor = Regressor.create_from_split_state_dict(encoder_state, head_state)
        self.regressor.to(device).eval()
        self.device = device
        self.use_global = self.regressor.feature_dim != self.regressor.decoder_dim
        self.global_feature_fn = global_feature_fn
        self.T_BC = None if T_BC is None else np.asarray(T_BC, dtype=float)
        if self.use_global and self.global_feature_fn is None:
            raise ValueError(
                "Head requires global features; provide global_feature_fn "
                "(DeiT rerank backbone, see run_fusion_eval._deit_feature_fn)")

    def infer(self, image_gray01: np.ndarray, K: np.ndarray) -> GLACEOutput:
        """image_gray01: HxW float array in [0, 1]. Returns the full correspondence
        set (uv, xyz_world) plus the PnP pose in both camera and body frames."""
        import torch

        if image_gray01.ndim != 2:
            raise ValueError("Expected a single-channel HxW image")
        h, w = image_gray01.shape
        image = torch.from_numpy(((image_gray01.astype(np.float32) - 0.4) / 0.25)[None, None])
        if self.use_global:
            feats = torch.from_numpy(np.asarray(self.global_feature_fn(image_gray01), dtype=np.float32))[None]
        else:
            feats = torch.zeros((1, 0), dtype=torch.float32)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            coords = self.regressor(image.to(self.device), feats.to(self.device))
        coords = coords.float().cpu().numpy()[0]  # [3, Hc, Wc]
        hc, wc = coords.shape[1], coords.shape[2]
        uv = pixel_grid_uv(self.IMAGE_SUBSAMPLE, hc, wc)
        xyz_world = coords.reshape(3, -1).T.astype(np.float64)
        K = np.asarray(K, dtype=float)
        T_WC, mask, inliers, diag = solve_pose_pnp(uv, xyz_world, K)
        T_WB = None
        if T_WC is not None and self.T_BC is not None:
            T_WB = glace_to_leader_frame(T_WC, self.T_BC)
        diag.update(cells=(hc, wc), image_hw=(int(h), int(w)))
        return GLACEOutput(T_WC=T_WC, T_WB=T_WB, uv=uv, xyz_world=xyz_world, K=K,
                           inlier_count=inliers, inlier_mask=mask,
                           image_size_hw=(int(h), int(w)), diagnostics=diag)


def deit_global_feature_fn(vendor_dir, checkpoint_path, image_size_hw=(480, 640)):
    """Build the DeiT rerank global-feature extractor used by GLACE-style heads."""
    from functools import partial
    import torch
    from torch import nn
    from torchvision import transforms
    vendor = Path(vendor_dir)
    sys.path.insert(0, str(vendor / "datasets"))
    from extract_features import DistilledVisionTransformer

    model = DistilledVisionTransformer(
        img_size=[image_size_hw[0], image_size_hw[1]], patch_size=16, embed_dim=384,
        depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=256)
    saved = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k.replace("module.backbone.", ""): v
                           for k, v in saved["model_state_dict"].items()
                           if k.startswith("module.backbone")})
    model.cuda().eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
        transforms.Resize([image_size_hw[0], image_size_hw[1]], antialias=False),
    ])

    def batch_fn(images_gray01):
        from PIL import Image
        import numpy as np
        tensors = []
        for gray in images_gray01:
            rgb = np.stack([gray] * 3, axis=-1)
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            tensors.append(transform(Image.fromarray(rgb)))
        with torch.inference_mode():
            return model(torch.stack(tensors).cuda()).cpu().numpy()

    def fn(image_gray01):
        return batch_fn([image_gray01])[0]

    fn.batch = batch_fn
    return fn
