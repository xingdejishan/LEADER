"""Frozen DeDoDe descriptor extraction + bilinear per-point sampling (spec section 6).

Uses the exact same assets as the existing rscore-assets pipeline
(dedode_descriptor_B.pth). The encoder runs once per (frame, camera) image;
per-point features come from `grid_sample` on the descriptor map with
image->feature-map scale handling. Backbone is frozen and eval-mode.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDeDoDeSampler(nn.Module):
    """Extract dense DeDoDe descriptor map per image, sample at arbitrary (u,v)."""

    def __init__(self, config, device="cuda"):
        super().__init__()
        self.device = device
        from kornia.feature.dedode.dedode_models import get_descriptor  # bufferx env
        weights = config["paths"]["dedode_weights_root"]
        self.model = get_descriptor(
            descriptor_model="D",
            weights=os_path_join(weights, "dedode_descriptor_B.pth"),
        )
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = config["image_feature"]["dim"]

    @torch.no_grad()
    def dense_map(self, image_tensor):
        """image_tensor: (1,1,H,W) or (1,3,H,W) float in [0,1]. Returns (1,C,h,w)."""
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
        desc = self.model(image_tensor)
        return desc

    @torch.no_grad()
    def sample(self, desc_map, uv, hw):
        """desc_map: (1,C,h,w); uv: (N,2) pixel coords; hw: (H,W) original image.

        Returns (N,C) features. Coordinates outside are zero (caller masks them).
        """
        _, C, h, w = desc_map.shape
        H, W = hw
        scale_x = w / float(W)
        scale_y = h / float(H)
        # grid_sample expects normalized coords in [-1,1], x last dim order (x,y)
        gx = uv[:, 0] * scale_x / (w - 1) * 2 - 1
        gy = uv[:, 1] * scale_y / (h - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2).to(desc_map.dtype)
        out = F.grid_sample(desc_map, grid, mode="bilinear", align_corners=True)
        return out[0, :, :, 0].T.contiguous()  # (N,C)


def os_path_join(*parts):
    import os
    return os.path.join(*parts)


