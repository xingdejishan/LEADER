import torch
from torch import nn
from torch.nn import functional as F


def project(points, camera_to_body, intrinsics, image_hw, correction=None):
    body = points.float()
    if correction is not None:
        inverse = torch.linalg.inv(correction.float())
        body = body @ inverse[:3, :3].T + inverse[:3, 3]
    camera = (body - camera_to_body[:3, 3]) @ camera_to_body[:3, :3]
    pixels = camera @ intrinsics.T
    uv = pixels[:, :2] / pixels[:, 2:].clamp_min(1e-6)
    h, w = image_hw
    valid = torch.isfinite(camera).all(-1) & (camera[:, 2] > 0)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    return uv, camera[:, 2], valid


def sample_map(feature_map, uv, image_hw):
    h, w = image_hw
    grid = (uv + .5) / uv.new_tensor([w, h]) * 2 - 1
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    return F.grid_sample(feature_map.float(), grid[None, None], align_corners=False,
                         mode='bilinear', padding_mode='zeros')[0, :, 0].T


def sample_visible(feature_map, points, camera_to_body, intrinsics, mask,
                   surface_points, correction=None, tolerance=.5, cell_size=4):
    image_hw = mask.shape[-2:]
    uv, depth, valid = project(points, camera_to_body, intrinsics, image_hw, correction)
    su, sd, sv = project(surface_points, camera_to_body, intrinsics, image_hw, correction)
    h, w = image_hw
    gh, gw = (h + cell_size - 1) // cell_size, (w + cell_size - 1) // cell_size
    zbuffer = depth.new_full((gh * gw,), float('inf'))
    cells = (su[sv] / cell_size).long()
    zbuffer.scatter_reduce_(0, cells[:, 1] * gw + cells[:, 0], sd[sv], reduce='amin')
    query = (torch.nan_to_num(uv) / cell_size).long()
    index = query[:, 1].clamp(0, gh - 1) * gw + query[:, 0].clamp(0, gw - 1)
    surface_depth = zbuffer[index]
    valid &= torch.isfinite(surface_depth) & ((depth - surface_depth).abs() <= tolerance)
    valid &= sample_map(mask.float().reshape(1, 1, h, w), uv, image_hw)[:, 0] > .999
    sampled = sample_map(feature_map, uv, image_hw)
    return torch.where(valid[:, None], sampled, torch.zeros_like(sampled)), valid


class ImageGate(nn.Module):
    def __init__(self, bottleneck=64):
        super().__init__()
        self.image_norm = nn.LayerNorm(128)
        self.lidar_norm = nn.LayerNorm(512)
        self.residual = nn.Sequential(nn.Linear(128, bottleneck), nn.GELU(),
                                      nn.Linear(bottleneck, 512, bias=False))
        self.gate = nn.Sequential(nn.Linear(640, bottleneck), nn.GELU(), nn.Linear(bottleneck, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, lidar, image, valid):
        image = self.image_norm(torch.where(valid[:, None], image, torch.zeros_like(image)))
        weight = self.gate(torch.cat([self.lidar_norm(lidar), image], -1)).sigmoid()
        delta = weight * self.residual(image)
        return lidar + torch.where(valid[:, None], delta, torch.zeros_like(delta))
