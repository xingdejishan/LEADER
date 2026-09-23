import math

import torch
from torch import nn
from torch.nn import functional as F


class ImagePyramid(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.stages = nn.ModuleList([
            self._stage(3, channels // 2),
            self._stage(channels // 2, channels),
            self._stage(channels, channels),
            self._stage(channels, channels),
            self._stage(channels, channels),
        ])

    @staticmethod
    def _stage(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        features = image
        pyramid = []
        for index, stage in enumerate(self.stages):
            features = stage(features)
            if index >= 2:
                pyramid.append(features)
        return pyramid


def project_voxel_centers(points, batch_index, intrinsics, camera_from_lidar, recovery, image_bounds):
    restored = torch.bmm(recovery[batch_index, :3, :3], points.unsqueeze(-1)).squeeze(-1)
    restored = restored + recovery[batch_index, :3, 3]
    camera = torch.bmm(camera_from_lidar[batch_index, :3, :3], restored.unsqueeze(-1)).squeeze(-1)
    camera = camera + camera_from_lidar[batch_index, :3, 3]
    depth = camera[:, 2]
    projected = torch.bmm(intrinsics[batch_index], camera.unsqueeze(-1)).squeeze(-1)
    pixels = projected[:, :2] / depth.clamp_min(1e-6).unsqueeze(-1)
    bounds = image_bounds[batch_index]
    valid = (depth > 1e-6) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
    valid &= (pixels[:, 0] < bounds[:, 0]) & (pixels[:, 1] < bounds[:, 1])
    valid &= torch.isfinite(pixels).all(dim=1)
    return pixels, valid


class VoxelRegionAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels, attention_channels=64, region_size=3):
        super().__init__()
        self.region_size = region_size
        self.query = nn.Linear(lidar_channels, attention_channels)
        self.key = nn.Conv2d(image_channels, attention_channels, 1)
        self.value = nn.Conv2d(image_channels, attention_channels, 1)
        self.fuse = nn.Linear(lidar_channels + attention_channels, lidar_channels)

    def forward(self, lidar, image, pixels, batch_index, valid, input_size, image_bounds):
        keys = self.key(image)
        values = self.value(image)
        visual = lidar.new_zeros((lidar.shape[0], keys.shape[1]))
        offsets = torch.arange(self.region_size, device=lidar.device, dtype=lidar.dtype)
        offsets = offsets - (self.region_size - 1) / 2
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        region = torch.stack((ox, oy), dim=-1).reshape(1, -1, 2)
        for batch in range(image.shape[0]):
            selection = torch.where(valid & (batch_index == batch))[0]
            if selection.numel() == 0:
                continue
            feature_pixels = (pixels[selection] + 0.5) * (image.shape[-1] / input_size) - 0.5
            locations = feature_pixels[:, None, :] + region
            feature_bounds = image.shape[-1] / input_size
            bounds = image_bounds[batch] * feature_bounds
            within = (locations[..., 0] >= 0) & (locations[..., 1] >= 0)
            within &= (locations[..., 0] < bounds[0]) & (locations[..., 1] < bounds[1])
            normalized = (locations + 0.5) * (2.0 / image.shape[-1]) - 1.0
            normalized = normalized.reshape(1, -1, self.region_size ** 2, 2)
            sampled_keys = F.grid_sample(keys[batch:batch + 1], normalized, align_corners=False)
            sampled_values = F.grid_sample(values[batch:batch + 1], normalized, align_corners=False)
            sampled_keys = sampled_keys[0].permute(1, 2, 0)
            sampled_values = sampled_values[0].permute(1, 2, 0)
            query = self.query(lidar[selection])
            scores = (sampled_keys * query[:, None, :]).sum(-1) / math.sqrt(query.shape[-1])
            scores = scores.masked_fill(~within, -1e4)
            weights = scores.softmax(dim=1)
            visual.index_copy_(0, selection, (weights[..., None] * sampled_values).sum(1))
        fused = self.fuse(torch.cat((lidar, visual), dim=1))
        return torch.where(valid[:, None], fused, lidar)


class MultiScaleAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_gate = nn.Linear(channels, channels)
        self.mid_gate = nn.Linear(channels, channels)
        self.high_gate = nn.Linear(channels, channels * 2)
        self.pair_gate = nn.Linear(channels * 2, channels * 2)
        self.output = nn.Linear(channels * 3, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, low, mid, high):
        gate = torch.sigmoid(self.low_gate(low) + self.mid_gate(mid))
        pair = torch.cat((gate * low, gate * mid), dim=1)
        final_gate = torch.sigmoid(self.pair_gate(pair) + self.high_gate(high))
        return self.output(torch.cat((final_gate * pair, high), dim=1))


class MaGiCFusion(nn.Module):
    def __init__(self, lidar_channels=512, image_channels=64, region_size=3):
        super().__init__()
        self.image_encoder = ImagePyramid(image_channels)
        self.attention = nn.ModuleList([
            VoxelRegionAttention(lidar_channels, image_channels, region_size=region_size)
            for _ in range(3)
        ])
        self.aggregate = MultiScaleAggregation(lidar_channels)

    @staticmethod
    def _pool_voxels(lidar, points, coordinates, stride, factor):
        group_coordinates = torch.cat((
            coordinates[:, :1].long(),
            torch.floor_divide(coordinates[:, 1:].long(), stride.long() * factor),
        ), dim=1)
        groups, inverse = torch.unique(group_coordinates, dim=0, return_inverse=True)
        counts = torch.bincount(inverse, minlength=groups.shape[0]).to(lidar.dtype).unsqueeze(1)
        pooled_lidar = lidar.new_zeros((groups.shape[0], lidar.shape[1])).index_add(0, inverse, lidar)
        pooled_points = points.new_zeros((groups.shape[0], 3)).index_add(0, inverse, points)
        return pooled_lidar / counts, pooled_points / counts, groups[:, 0], inverse

    def forward(self, lidar, points, coordinates, stride, images, intrinsics, camera_from_lidar, recovery, image_bounds):
        points = points.to(lidar.device)
        coordinates = coordinates.to(lidar.device)
        stride = stride.to(lidar.device)
        batch_index = coordinates[:, 0].long()
        _, fine_valid = project_voxel_centers(
            points, batch_index, intrinsics, camera_from_lidar, recovery, image_bounds
        )
        image_features = self.image_encoder(images)
        fused = []
        for factor, attention, feature in zip((1, 2, 4), self.attention, image_features):
            pooled_lidar, pooled_points, pooled_batch, inverse = self._pool_voxels(
                lidar, points, coordinates, stride, factor
            )
            pixels, valid = project_voxel_centers(
                pooled_points, pooled_batch, intrinsics, camera_from_lidar, recovery, image_bounds
            )
            fused.append(attention(
                pooled_lidar, feature, pixels, pooled_batch, valid, images.shape[-1], image_bounds
            )[inverse])
        delta = self.aggregate(*fused)
        return lidar + delta * fine_valid[:, None]
