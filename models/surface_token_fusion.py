import math

import torch
from torch import nn
from torch.nn import functional as F

from models.magic_fusion import MaGiCFusion, project_voxel_centers


@torch.no_grad()
def surface_support(raw_points, coordinates, stride, intrinsics, extrinsics, recovery,
                    bounds, image_valid_mask, voxel_size=0.2, horizontal=1024):
    device = intrinsics.device
    dtype = intrinsics.dtype
    coordinates = coordinates.to(device=device, dtype=torch.long)
    stride = torch.as_tensor(stride, device=device, dtype=dtype)
    points = torch.cat([torch.as_tensor(p, device=device, dtype=dtype) for p in raw_points])
    batch = torch.cat([torch.full((len(p),), b, device=device, dtype=torch.long)
                       for b, p in enumerate(raw_points)])
    angle = torch.atan2(points[:, 1], points[:, 0]).clamp(-math.pi, math.pi - 1e-6)
    polar = torch.stack((angle * horizontal / (2 * math.pi),
                         points[:, :2].norm(dim=1) / voxel_size,
                         points[:, 2] / voxel_size), dim=1)
    cell = torch.floor(polar / stride).long() * stride.long()
    keys = torch.cat((batch[:, None], cell), dim=1)
    groups, inverse = torch.unique(torch.cat((coordinates, keys)), dim=0, return_inverse=True)
    lookup = torch.full((len(groups),), -1, device=device, dtype=torch.long)
    lookup[inverse[:len(coordinates)]] = torch.arange(len(coordinates), device=device)
    rows = lookup[inverse[len(coordinates):]]
    pixels, valid = project_voxel_centers(points, batch, intrinsics, extrinsics, recovery, bounds)
    if image_valid_mask is not None:
        for b in range(len(raw_points)):
            chosen = torch.where(batch == b)[0]
            grid = ((pixels[chosen] + 0.5) * (2 / 1024) - 1).reshape(1, -1, 1, 2)
            mask = F.grid_sample(image_valid_mask[b:b + 1].to(dtype), grid,
                                 align_corners=False).reshape(-1)
            valid[chosen] &= mask > 0.5
    valid &= rows >= 0
    local = (polar - cell) / stride
    octant = torch.floor(local * 2).long().clamp(0, 1)
    slot = octant[:, 0] * 4 + octant[:, 1] * 2 + octant[:, 2]
    distance = (local - (octant.to(dtype) + 0.5) / 2).square().sum(dim=1)
    candidates = torch.where(valid)[0]
    bins = rows[candidates] * 8 + slot[candidates]
    best = torch.full((len(coordinates) * 8,), torch.inf, device=device, dtype=dtype)
    best.scatter_reduce_(0, bins, distance[candidates], reduce='amin', include_self=True)
    winners = distance[candidates] == best[bins]
    selected = torch.full((len(coordinates) * 8,), len(points), device=device, dtype=torch.long)
    selected.scatter_reduce_(0, bins[winners], candidates[winners], reduce='amin', include_self=True)
    supported = selected < len(points)
    padded_pixels = torch.cat((pixels, torch.zeros((1, 2), device=device, dtype=dtype)))
    padded_local = torch.cat((local - 0.5, torch.zeros((1, 3), device=device, dtype=dtype)))
    return (padded_pixels[selected].reshape(-1, 8, 2),
            padded_local[selected].reshape(-1, 8, 3), supported.reshape(-1, 8))


class SurfaceTokenAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels, attention_channels=64):
        super().__init__()
        self.query = nn.Linear(lidar_channels, attention_channels)
        self.key = nn.Conv2d(image_channels, attention_channels, 1)
        self.value = nn.Conv2d(image_channels, attention_channels, 1)
        self.fuse = nn.Linear(lidar_channels + attention_channels, lidar_channels)
        self.relative_key = nn.Linear(3, attention_channels, bias=False)
        self.relative_value = nn.Linear(3, attention_channels, bias=False)
        nn.init.zeros_(self.relative_key.weight)
        nn.init.zeros_(self.relative_value.weight)

    def forward(self, lidar, image, pixels, offsets, supported, batch_index,
                image_bounds, image_valid_mask):
        keys, values = self.key(image), self.value(image)
        visual = lidar.new_zeros((len(lidar), keys.shape[1]))
        valid = torch.zeros(len(lidar), device=lidar.device, dtype=torch.bool)
        for b in range(len(image)):
            selected = torch.where((batch_index == b) & supported.any(dim=1))[0]
            if not len(selected):
                continue
            scale = image.shape[-1] / 1024
            x = torch.arange(image.shape[-1], device=image.device, dtype=image.dtype)
            y = torch.arange(image.shape[-2], device=image.device, dtype=image.dtype)
            mask = ((image_bounds[b, 1] * scale - y).clamp(0, 1)[:, None] *
                    (image_bounds[b, 0] * scale - x).clamp(0, 1)[None, :])[None, None]
            if image_valid_mask is not None:
                mask = mask * F.avg_pool2d(image_valid_mask[b:b + 1], 64 // image.shape[-1])
            grid = ((pixels[selected] + 0.5) * (2 / 1024) - 1)[None]
            mass = F.grid_sample(mask, grid, align_corners=False)
            sampled_k = (F.grid_sample(keys[b:b + 1] * mask, grid, align_corners=False) /
                         mass.clamp_min(1e-6))[0].permute(1, 2, 0)
            sampled_v = (F.grid_sample(values[b:b + 1] * mask, grid, align_corners=False) /
                         mass.clamp_min(1e-6))[0].permute(1, 2, 0)
            usable = supported[selected] & (mass[0, 0] > 1e-6)
            relative = offsets[selected]
            logits = ((sampled_k + self.relative_key(relative)) *
                      self.query(lidar[selected])[:, None]).sum(-1) / math.sqrt(keys.shape[1])
            weights = logits.masked_fill(~usable, -1e4).softmax(dim=1) * usable
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            visual[selected] = (weights[..., None] *
                                (sampled_v + self.relative_value(relative))).sum(dim=1)
            valid[selected] = usable.any(dim=1)
        fused = self.fuse(torch.cat((lidar, visual), dim=1))
        return torch.where(valid[:, None], fused, lidar), valid


class SurfaceMaGiCFusion(MaGiCFusion):
    requires_raw_points = True

    def __init__(self, lidar_channels=512, image_channels=64):
        super().__init__(lidar_channels, image_channels)
        self.attention = nn.ModuleList([
            SurfaceTokenAttention(lidar_channels, image_channels) for _ in range(3)
        ])

    def forward(self, lidar, points, coordinates, stride, sam_features, intrinsics,
                camera_from_lidar, recovery, image_bounds, stages=None,
                voxel_size=0.2, horizontal=1024, image_valid_mask=None,
                return_validity=False, raw_points=None):
        if raw_points is None or stages is None or len(stages) != 3:
            raise ValueError('Surface fusion requires raw LiDAR points and three RPGE stages')
        images = self.image_encoder(sam_features, image_bounds, image_valid_mask)
        fused, supports = [], []
        for stage, projection, attention, image in zip(
                stages, self.stage_projections, self.attention, images):
            stage_coords = stage.C.to(lidar.device)
            stage_stride = torch.as_tensor(stage.tensor_stride, device=lidar.device)
            pixels, relative, supported = surface_support(
                raw_points, stage_coords, stage_stride, intrinsics, camera_from_lidar,
                recovery, image_bounds, image_valid_mask, voxel_size, horizontal)
            features, valid = attention(projection(stage.F), image, pixels, relative,
                                        supported, stage_coords[:, 0].long(),
                                        image_bounds, image_valid_mask)
            fused.append(self._align_stage(features, stage_coords, stage_stride, coordinates, stride))
            supports.append(self._align_stage(valid[:, None].to(lidar.dtype), stage_coords,
                                              stage_stride, coordinates, stride)[:, 0] > 0)
        _, fine_valid = project_voxel_centers(
            points, coordinates[:, 0].long(), intrinsics, camera_from_lidar, recovery, image_bounds)
        if image_valid_mask is not None:
            pixels, _, supported = surface_support(
                raw_points, coordinates, stride, intrinsics, camera_from_lidar,
                recovery, image_bounds, image_valid_mask, voxel_size, horizontal)
            fine_valid &= supported.any(dim=1)
        valid = fine_valid & torch.stack(supports).any(dim=0)
        output = lidar + self.aggregate(*fused) * valid[:, None]
        return (output, valid) if return_validity else output
