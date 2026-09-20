"""Trainable visual residual wrapper around the frozen upstream LEADER model."""
import os
import sys

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.model_mink import LEADER
from fusion import PreVoxelMultiViewFusion


class VisualProjection(nn.Module):
    def __init__(self, in_dim=65, hidden_dim=128, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v, reliability):
        return self.net(torch.cat([v, reliability[:, None]], dim=1)) * reliability[:, None]


class PreVoxelMultiViewLEADER(nn.Module):
    """Frozen LEADER plus a trainable raw-point visual residual branch.

    The residual is injected after the original LiDAR input stem's first block.
    This keeps the original LiDAR mapping intact and makes force-NULL exactly
    equivalent to the frozen baseline forward.
    """

    def __init__(self, config, checkpoint_dir, width=1024):
        super().__init__()
        self.config = config
        self.base = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=width)
        state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
        self.base.load_state_dict(state, strict=True)
        self.base.requires_grad_(False)
        self.base.eval()
        self.fusion = PreVoxelMultiViewFusion(config)
        stem_out = self.base.encoder.stem[0].linear.out_features
        self.visual_projection = VisualProjection(65, 128, stem_out)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.fusion.train(mode)
        self.visual_projection.train(mode)
        return self

    def forward(self, obs, lidar_feats, coords, force_null=None,
                all_camera_dropout_prob=0.0, rng=None):
        device = lidar_feats.device
        if force_null is None and all_camera_dropout_prob > 0 and rng is not None:
            force_null = torch.from_numpy(
                rng.random(lidar_feats.shape[0]) < all_camera_dropout_prob).to(device)
        alpha_null, alpha_views, v = self.fusion(
            obs["img_feat"].to(device), obs["quality"].to(device),
            obs["valid"].to(device), force_null=force_null)
        index = obs["index"].to(device)
        v_voxel = v[index]
        reliability = (1.0 - alpha_null)[index]
        h_visual = self.visual_projection(v_voxel, reliability)

        sparse = ME.SparseTensor(features=lidar_feats[index], coordinates=coords)
        h_lidar = self.base.encoder.stem[0](sparse)
        mixed = ME.SparseTensor(
            features=h_lidar.F + h_visual,
            coordinate_manager=h_lidar.coordinate_manager,
            coordinate_map_key=h_lidar.coordinate_map_key,
            tensor_stride=h_lidar.tensor_stride,
        )
        stem = self.base.encoder.stem[1](mixed)
        encoded = self.base.encoder._unet_forward(
            stem, self.base.encoder.encoders, self.base.encoder.decoders)
        prediction = self.base.decoder(encoded.F)
        diagnostics = {
            "alpha_null": alpha_null,
            "alpha_views": alpha_views,
            "visual_residual": h_visual,
            "encoded": encoded,
        }
        return prediction, diagnostics
