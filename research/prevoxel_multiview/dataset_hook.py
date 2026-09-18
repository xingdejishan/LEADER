"""Dataset-side preparation ONLY (fixed observations, no trainable modules).

Produces per frame, aligned to raw points:
  img_feat   (N, 6, 128)  PCA128'd DeDoDe features (0 for invalid views)
  quality    (N, 6, 9)    explicit quality cues
  valid      (N, 6)       bool, after hard filter + occlusion
  voxel      dict for the model forward:
    coords_q (M, 3) int    quantized polar coords (LEADER's own cell space)
    inverse  (N,) int      raw point -> quantized row  (ME's own mapping)
    index     (M,) int64   representative raw point selected by ME per row
    inverse   (N,) int64   raw point -> quantized row (diagnostics only)

NO ViewAdapter / Transformer here — those live in fusion.py and run inside
the training loop's model.forward so scene-coordinate loss gradients reach
them (gradient chain: loss -> W_V -> representative-point gather -> v_i ->
Transformer/Adapter). DeDoDe, projection, occlusion and quality cues are
computed under no_grad (frozen / non-differentiable by design).

Mapping semantics verified empirically on this build (ME 0.14, egonn118 env):
  coords_q = floor(coords / quantization_size) rows in order of first
  occurrence; `inverse` maps raw->row; `index` maps row->representative raw
  point; the quantized feature is exactly the representative point's feature.
  The visual branch must use the same `index`, rather than mean-pooling over
  `inverse`.
"""
import json
import os

import numpy as np
import torch

from projector import SixCameraProjector, load_config, load_vel_lb3
from visibility import VisibilityChecker, black_border_mask
from quality import quality_vector


class PerFrameObservation:
    """Compute fixed per-frame observations. Holds NO trainable parameters."""

    def __init__(self, config=None, device="cuda"):
        self.cfg = config or load_config()
        self.projector = SixCameraProjector(self.cfg)
        self.vis = VisibilityChecker(self.cfg)
        self.device = device
        self._dedode = None  # lazy load

    def _get_dedode(self):
        if self.cfg["image_feature"].get("dim") != 256 and not self._pca_available():
            pass  # dim config governs; PCA presence checked in pca128()
        if self._dedode is None:
            try:
                from kornia.feature.dedode.dedode_models import get_descriptor
                m = get_descriptor(kind="B")
                weights = os.path.join(
                    self.cfg["paths"]["dedode_weights_root"],
                    "dedode_descriptor_B.pth")
                if os.path.exists(weights):
                    state = torch.load(weights, map_location="cpu", weights_only=True)
                    m.load_state_dict(state, strict=True)
            except ImportError:
                self._dedode = False  # mark unavailable; features stay zero
                return None
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._dedode = m
        return self._dedode or None

    def _pca_available(self):
        import os
        p = self.cfg["image_feature"].get("pca_weights")
        return p and os.path.exists(p)

    @torch.no_grad()
    def observe(self, frame_id, scan_lb3, compute_image_features=True,
                compute_quality=True):
        """scan_lb3: (N,3) raw lb3-frame scan. Returns dict of numpy arrays."""
        n = scan_lb3.shape[0]
        proj = self.projector.project(scan_lb3.astype(np.float64))
        uv, depth, valid = proj["uv"], proj["depth"], proj["valid"]

        images = self.load_images(frame_id)
        # black border: pixel is border-black only when ALL channels near 0
        # (max>thresh == at least one channel lit; min>thresh wrongly kills
        # saturated single-channel pixels like [200,5,5])
        bb = np.zeros((n, 6), dtype=bool)
        for cam in range(6):
            keep = black_border_mask(images[cam])
            sel = valid[:, cam]
            u = np.clip(uv[sel, cam, 0].astype(int), 0, self.cfg["image"]["width"] - 1)
            v = np.clip(uv[sel, cam, 1].astype(int), 0, self.cfg["image"]["height"] - 1)
            ok = keep[v, u]
            idx = np.where(sel)[0]
            bb[idx[~ok], cam] = True
        valid = valid & ~bb

        occluded = np.zeros((n, 6), dtype=bool)
        dmargin = np.zeros((n, 6), dtype=np.float32)
        devid = np.zeros((n, 6), dtype=bool)
        for cam in range(6):
            grid, gw, gh = self.vis.build_zbuffer(uv[:, cam], depth[:, cam], valid[:, cam])
            occ, mg, ev = self.vis.occlusion_query(grid, gw, gh, uv[:, cam], depth[:, cam], valid[:, cam])
            occluded[:, cam] = occ
            dmargin[:, cam] = mg
            devid[:, cam] = ev
        valid = valid & ~occluded

        dim = self.cfg["image_feature"]["dim"]
        img_feat = np.zeros((n, 6, dim), dtype=np.float32)
        if compute_image_features:
            model = self._get_dedode()
            from PIL import Image
            for cam in range(6):
                if model is None:
                    break
                sel = np.where(valid[:, cam])[0]
                if sel.size == 0:
                    continue
                im = Image.open(self.projector.resolve_image_path(frame_id, cam)).convert("RGB")
                x = torch.from_numpy(np.asarray(im).copy()).to(self.device).float().permute(2, 0, 1)[None] / 255.0
                desc = model(x).float()  # (1,C,h,w), C=256 before PCA
                desc = pca128(desc, self.cfg)  # exact rscore-assets口径: conv2d PCA
                _, C, h, w = desc.shape
                H, W = self.cfg["image"]["height"], self.cfg["image"]["width"]
                gx = torch.from_numpy(uv[sel, cam, 0] * w / W / (w - 1) * 2 - 1).float()
                gy = torch.from_numpy(uv[sel, cam, 1] * h / H / (h - 1) * 2 - 1).float()
                grid = torch.stack([gx, gy], -1).view(1, -1, 1, 2).to(self.device)
                out = torch.nn.functional.grid_sample(desc, grid, mode="bilinear", align_corners=True)
                img_feat[sel, cam] = out[0, :, :, 0].T.cpu().numpy()

        quality = (quality_vector(proj["cam_xyz"], uv, valid, dmargin, devid,
                                  images, img_feat, self.cfg)
                   if compute_quality else np.zeros((n, 6, 9), dtype=np.float32))
        return dict(img_feat=img_feat, quality=quality, valid=valid,
                    occluded=occluded, black_border=bb)

    def load_images(self, frame_id):
        from PIL import Image
        return [np.asarray(Image.open(self.projector.resolve_image_path(frame_id, cam)).convert("RGB"))
                for cam in range(6)]


_PCA_CACHE = {}


def pca128(desc, cfg):
    """Apply the exact rscore-assets PCA128: conv2d with pcad3LB_128.pth weights.

    multicamera.py line 93: dense = F.conv2d(dense.float(), pca['weight'], pca['bias']).
    Cache file lives on the server path by default; falls back to no-PCA with a
    loud warning if absent locally (config must then pin dim accordingly).
    """
    if not cfg["image_feature"].get("pca_enabled", True):
        return desc
    path = cfg["image_feature"].get("pca_weights")
    if path is None or not os.path.exists(path):
        if not _PCA_CACHE.get("warned"):
            print("[pca128] WARNING: PCA weights not found (%s); "
                  "returning raw 256D descriptors. Config dim must be 256!" % path)
            _PCA_CACHE["warned"] = True
        return desc
    key = path
    if key not in _PCA_CACHE:
        pca = torch.load(path, map_location="cpu", weights_only=True)
        _PCA_CACHE[key] = (pca["weight"].float(), pca["bias"].float())
    w, b = _PCA_CACHE[key]
    return torch.nn.functional.conv2d(desc.float(), w, b)


def voxel_mapping(pl_coords, voxel_size):
    """Call ME sparse_quantize to obtain THE mapping used by LEADER.

    pl_coords: (N,3) polar-expanded coords (same array LEADER quantizes).
    Returns (coords_q int32 (M,3), index int64 (M,), inverse int64 (N,)).

    Semantics verified on this build (ME 0.14, real 65036-pt frame):
      coords_q rows = first-occurrence order of floor(coords/voxel_size);
      index[r] = representative raw point of row r and ME's quantized
      features satisfy feats_q == feats[index] EXACTLY (representative
      selection, NOT mean; diag_mapping2). Visual features must therefore
      be gathered with the same index — never re-derived, never averaged.
    """
    import MinkowskiEngine as ME
    dummy = np.zeros((pl_coords.shape[0], 1), dtype=np.float32)
    coords_q, _, index, inverse = ME.utils.sparse_quantize(
        coordinates=pl_coords, features=dummy,
        quantization_size=voxel_size, return_index=True, return_inverse=True)
    coords_q = np.asarray(coords_q).astype(np.int32)
    index = np.asarray(index).astype(np.int64)
    inverse = np.asarray(inverse).astype(np.int64)
    return coords_q, index, inverse

