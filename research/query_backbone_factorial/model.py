from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dinov2_fusion'))
from fusion import FeatureFusion


class FactorialFusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.fusion = FeatureFusion()
        self.image_norm = nn.LayerNorm(128)
        self.lidar_norm = nn.LayerNorm(512)
        self.query = nn.Linear(512, 64)
        self.key = nn.Linear(128, 64)
        self.direction = nn.Linear(3, 64, bias=False)

    def forward(self, lidar, image, mask, direction):
        if self.mode.endswith('center'):
            image, mask = image[:, :, 12:13], mask[:, :, 12:13]
        image = torch.where(mask[..., None], image.float(), 0.)
        query = self.query(self.lidar_norm(lidar))
        logits = (query[:, None, None] * self.key(self.image_norm(image))).sum(-1) / 8
        weights = logits.masked_fill(~mask, -1e4).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weights[..., None] * image).sum(2)
        visible = mask.any(-1)
        logits = (query[:, None] * (self.key(self.image_norm(pooled)) + self.direction(direction))).sum(-1) / 8
        weights = logits.masked_fill(~visible, -1e4).softmax(-1) * visible
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return self.fusion(lidar, (weights[..., None] * pooled).sum(1), visible.any(-1))
