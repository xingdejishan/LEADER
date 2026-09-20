import torch
from torch import nn
from torch.nn import functional as F


def sample_patches(features, uv, image_hw):
    h, w = image_hw
    grid = (uv + .5) / uv.new_tensor([w, h]) * 2 - 1
    return F.grid_sample(features.float(), grid[None, None], align_corners=False,
                         padding_mode='border')[0, :, 0].T


class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_norm = nn.LayerNorm(128)
        self.lidar_norm = nn.LayerNorm(512)
        self.residual = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 512, bias=False))
        self.gate = nn.Sequential(nn.Linear(640, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, lidar, image, valid):
        image = self.image_norm(torch.where(valid[:, None], image, torch.zeros_like(image)))
        weight = self.gate(torch.cat([self.lidar_norm(lidar), image], -1)).sigmoid()
        return lidar + torch.where(valid[:, None], weight * self.residual(image), torch.zeros_like(lidar))
