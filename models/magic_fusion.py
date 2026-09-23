import math

import torch
from torch import nn
from torch.nn import functional as F


class SAMFeaturePyramid(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(256, channels, 1) for _ in range(3)])

    def forward(self, embedding, image_bounds=None):
        if embedding.ndim != 4 or embedding.shape[1:] != (256, 64, 64):
            raise ValueError('Expected SAM ViT-L image embeddings shaped [B, 256, 64, 64]')
        if image_bounds is None:
            mask = embedding.new_ones((embedding.shape[0], 1, 64, 64))
        else:
            x = torch.arange(64, device=embedding.device, dtype=embedding.dtype)[None, :]
            y = x.transpose(0, 1)
            width = (image_bounds[:, 0] * (64.0 / 1024.0))[:, None, None]
            height = (image_bounds[:, 1] * (64.0 / 1024.0))[:, None, None]
            mask = ((width - x).clamp(0, 1) * (height - y).clamp(0, 1))[:, None]
        return [
            projection(F.avg_pool2d(embedding * mask, factor) /
                       F.avg_pool2d(mask, factor).clamp_min(1e-6))
            for factor, projection in zip((1, 2, 4), self.projections)
        ]


def project_voxel_centers(points, batch_index, intrinsics, camera_from_lidar, recovery, image_bounds,
                          require_bounds=True):
    restored = torch.bmm(recovery[batch_index, :3, :3], points.unsqueeze(-1)).squeeze(-1)
    restored = restored + recovery[batch_index, :3, 3]
    camera = torch.bmm(camera_from_lidar[batch_index, :3, :3], restored.unsqueeze(-1)).squeeze(-1)
    camera = camera + camera_from_lidar[batch_index, :3, 3]
    depth = camera[:, 2]
    projected = torch.bmm(intrinsics[batch_index], camera.unsqueeze(-1)).squeeze(-1)
    pixels = projected[:, :2] / depth.clamp_min(1e-6).unsqueeze(-1)
    bounds = image_bounds[batch_index]
    valid = depth > 1e-6
    if require_bounds:
        valid &= (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
        valid &= (pixels[:, 0] < bounds[:, 0]) & (pixels[:, 1] < bounds[:, 1])
    valid &= torch.isfinite(pixels).all(dim=1)
    return pixels, valid


def polar_voxel_centers(coordinates, stride, voxel_size, horizontal):
    polar = (coordinates[:, 1:].to(stride.dtype) + stride / 2) * voxel_size
    angle = polar[:, 0] * (2 * math.pi / (horizontal * voxel_size))
    return torch.stack((polar[:, 1] * angle.cos(), polar[:, 1] * angle.sin(), polar[:, 2]), dim=1)


def project_polar_voxel_regions(coordinates, stride, voxel_size, horizontal, intrinsics,
                                camera_from_lidar, recovery, image_bounds):
    corners = torch.tensor([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)],
                           device=coordinates.device, dtype=stride.dtype)
    polar = (coordinates[:, None, 1:].to(stride.dtype) + corners[None] * stride) * voxel_size
    angle = polar[..., 0] * (2 * math.pi / (horizontal * voxel_size))
    points = torch.stack((polar[..., 1] * angle.cos(), polar[..., 1] * angle.sin(), polar[..., 2]), dim=-1)
    batch_index = coordinates[:, 0].long().repeat_interleave(8)
    pixels, valid = project_voxel_centers(points.reshape(-1, 3), batch_index, intrinsics,
                                          camera_from_lidar, recovery, image_bounds,
                                          require_bounds=False)
    pixels = pixels.reshape(-1, 8, 2)
    valid = valid.reshape(-1, 8).all(dim=1)
    region = torch.stack((pixels.amin(dim=1), pixels.amax(dim=1)), dim=1)
    return region, valid


class VoxelRegionAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels, attention_channels=64, region_size=3):
        super().__init__()
        self.region_size = region_size
        self.query = nn.Linear(lidar_channels, attention_channels)
        self.key = nn.Conv2d(image_channels, attention_channels, 1)
        self.value = nn.Conv2d(image_channels, attention_channels, 1)
        self.fuse = nn.Linear(lidar_channels + attention_channels, lidar_channels)

    def forward(self, lidar, image, pixels, batch_index, valid, input_size, image_bounds,
                region_bounds=None):
        keys = self.key(image)
        values = self.value(image)
        visual = lidar.new_zeros((lidar.shape[0], keys.shape[1]))
        fractions = torch.linspace(0, 1, self.region_size, device=lidar.device, dtype=lidar.dtype)
        fy, fx = torch.meshgrid(fractions, fractions, indexing="ij")
        samples = torch.stack((fx, fy), dim=-1).reshape(1, -1, 2)
        offsets = (samples - 0.5) * (self.region_size - 1)
        has_support = torch.zeros_like(valid)
        for batch in range(image.shape[0]):
            selection = torch.where(valid & (batch_index == batch))[0]
            if selection.numel() == 0:
                continue
            feature_scale = image.shape[-1] / input_size
            if region_bounds is None:
                feature_pixels = (pixels[selection] + 0.5) * feature_scale - 0.5
                locations = feature_pixels[:, None, :] + offsets
            else:
                bounds_pixels = region_bounds[selection]
                locations = (bounds_pixels[:, 0, None, :] * (1 - samples) +
                             bounds_pixels[:, 1, None, :] * samples + 0.5) * feature_scale - 0.5
            x = torch.arange(image.shape[-1], device=image.device, dtype=image.dtype)
            y = torch.arange(image.shape[-2], device=image.device, dtype=image.dtype)
            x_weight = (image_bounds[batch, 0] * feature_scale - x).clamp(0, 1)
            y_weight = (image_bounds[batch, 1] * (image.shape[-2] / input_size) - y).clamp(0, 1)
            mask = (y_weight[:, None] * x_weight[None, :])[None, None]
            normalized = (locations + 0.5) * (2.0 / image.shape[-1]) - 1.0
            normalized = normalized.reshape(1, -1, self.region_size ** 2, 2)
            sampled_mask = F.grid_sample(mask, normalized, align_corners=False)
            sampled_keys = F.grid_sample(keys[batch:batch + 1] * mask, normalized,
                                         align_corners=False) / sampled_mask.clamp_min(1e-6)
            sampled_values = F.grid_sample(values[batch:batch + 1] * mask, normalized,
                                           align_corners=False) / sampled_mask.clamp_min(1e-6)
            sampled_keys = sampled_keys[0].permute(1, 2, 0)
            sampled_values = sampled_values[0].permute(1, 2, 0)
            within = sampled_mask[0, 0] > 1e-6
            has_support[selection] = within.any(dim=1)
            query = self.query(lidar[selection])
            scores = (sampled_keys * query[:, None, :]).sum(-1) / math.sqrt(query.shape[-1])
            scores = scores.masked_fill(~within, -1e4)
            weights = scores.softmax(dim=1)
            visual.index_copy_(0, selection, (weights[..., None] * sampled_values).sum(1))
        fused = self.fuse(torch.cat((lidar, visual), dim=1))
        return torch.where((valid & has_support)[:, None], fused, lidar)


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
        self.image_encoder = SAMFeaturePyramid(image_channels)
        self.stage_projections = nn.ModuleList([nn.Linear(channels, lidar_channels)
                                                for channels in (32, 128, 384)])
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

    @staticmethod
    def _align_stage(features, source_coords, source_stride, target_coords, target_stride):
        group_stride = torch.maximum(source_stride.long(), target_stride.long())
        source_keys = torch.cat((source_coords[:, :1].long(),
                                 torch.floor_divide(source_coords[:, 1:].long(), group_stride)), dim=1)
        target_keys = torch.cat((target_coords[:, :1].long(),
                                 torch.floor_divide(target_coords[:, 1:].long(), group_stride)), dim=1)
        groups, inverse = torch.unique(torch.cat((source_keys, target_keys)), dim=0,
                                       return_inverse=True)
        source_index = inverse[:source_keys.shape[0]]
        target_index = inverse[source_keys.shape[0]:]
        pooled = features.new_zeros((groups.shape[0], features.shape[1]))
        pooled.index_add_(0, source_index, features)
        counts = torch.bincount(source_index, minlength=groups.shape[0]).to(features.dtype)
        return (pooled / counts.clamp_min(1)[:, None])[target_index]

    def forward(self, lidar, points, coordinates, stride, sam_features, intrinsics, camera_from_lidar,
                recovery, image_bounds, stages=None, voxel_size=0.2, horizontal=1024):
        points = points.to(lidar.device)
        coordinates = coordinates.to(lidar.device)
        stride = stride.to(lidar.device)
        batch_index = coordinates[:, 0].long()
        _, fine_valid = project_voxel_centers(
            points, batch_index, intrinsics, camera_from_lidar, recovery, image_bounds
        )
        image_features = self.image_encoder(sam_features, image_bounds)
        fused = []
        if stages is None:
            for factor, attention, feature in zip((1, 2, 4), self.attention, image_features):
                pooled_lidar, pooled_points, pooled_batch, inverse = self._pool_voxels(
                    lidar, points, coordinates, stride, factor
                )
                pixels, valid = project_voxel_centers(
                    pooled_points, pooled_batch, intrinsics, camera_from_lidar, recovery, image_bounds
                )
                fused.append(attention(
                    pooled_lidar, feature, pixels, pooled_batch, valid, 1024, image_bounds
                )[inverse])
        else:
            if len(stages) != 3:
                raise ValueError('Expected three RPGE stage tensors')
            for stage, projection, attention, feature in zip(
                    stages, self.stage_projections, self.attention, image_features):
                stage_coords = stage.C.to(lidar.device)
                stage_stride = torch.as_tensor(stage.tensor_stride, device=lidar.device, dtype=lidar.dtype)
                stage_lidar = projection(stage.F)
                stage_points = polar_voxel_centers(stage_coords, stage_stride, voxel_size, horizontal)
                stage_batch = stage_coords[:, 0].long()
                pixels, valid = project_voxel_centers(
                    stage_points, stage_batch, intrinsics, camera_from_lidar, recovery, image_bounds
                )
                regions, region_valid = project_polar_voxel_regions(
                    stage_coords, stage_stride, voxel_size, horizontal, intrinsics,
                    camera_from_lidar, recovery, image_bounds
                )
                attended = attention(stage_lidar, feature, pixels, stage_batch, valid & region_valid,
                                     1024, image_bounds, region_bounds=regions)
                fused.append(self._align_stage(attended, stage_coords, stage_stride, coordinates, stride))
        delta = self.aggregate(*fused)
        return lidar + delta * fine_valid[:, None]
