from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dinov2_fusion'))
from fusion import FeatureFusion


class MultiViewFusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ['lidar', 'cam5', 'mean', 'query']:
            raise ValueError(mode)
        self.mode = mode
        self.fusion = FeatureFusion()
        self.image_norm = nn.LayerNorm(128)
        self.lidar_norm = nn.LayerNorm(512)
        self.query = nn.Linear(512, 64)
        self.key = nn.Linear(128, 64)
        self.direction = nn.Linear(3, 64, bias=False)

    def forward(self, lidar, image, mask, direction, return_weights=False):
        mask = mask.clone()
        if self.mode == 'lidar':
            mask.zero_()
        elif self.mode == 'cam5':
            mask[:, :5] = False
        image = torch.where(mask[..., None], image, torch.zeros_like(image))
        if self.mode == 'query':
            keys = self.key(self.image_norm(image)) + self.direction(direction)
            logits = (self.query(self.lidar_norm(lidar))[:, None] * keys).sum(-1) / 8
            weights = logits.masked_fill(~mask, -1e4).softmax(-1) * mask
        else:
            weights = mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weights[..., None] * image).sum(1)
        output = self.fusion(lidar, pooled, mask.any(-1))
        return (output, weights) if return_weights else output
