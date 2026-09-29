import MinkowskiEngine as ME
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.magic_fusion import project_polar_voxel_regions


def _as_stride(value, device=None):
    if isinstance(value, torch.Tensor):
        stride = value.to(device=device, dtype=torch.long).reshape(-1)
    else:
        stride = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
    if stride.numel() == 1:
        stride = stride.repeat(3)
    if stride.numel() != 3 or torch.any(stride <= 0):
        raise ValueError('A positive three-dimensional sparse tensor stride is required')
    return stride


def _cell_keys(coordinates, common_stride):
    stride = _as_stride(common_stride, coordinates.device)
    coords = coordinates.long()
    return torch.cat((coords[:, :1], torch.div(coords[:, 1:], stride, rounding_mode='floor')), dim=1)


def align_sparse_features(features, source_coordinates, target_coordinates, common_stride):
    if features.ndim != 2 or source_coordinates.ndim != 2 or target_coordinates.ndim != 2:
        raise ValueError('Sparse feature alignment expects rank-two features and coordinates')
    if features.shape[0] != source_coordinates.shape[0]:
        raise ValueError('Sparse source features and coordinates differ in row count')
    source_keys = _cell_keys(source_coordinates, common_stride)
    target_keys = _cell_keys(target_coordinates, common_stride)
    joined = torch.cat((source_keys, target_keys), dim=0)
    unique, inverse = torch.unique(joined, dim=0, sorted=True, return_inverse=True)
    source_inverse = inverse[:source_keys.shape[0]]
    target_inverse = inverse[source_keys.shape[0]:]
    accumulated = features.new_zeros((unique.shape[0], features.shape[1]))
    accumulated.index_add_(0, source_inverse, features)
    counts = features.new_zeros((unique.shape[0], 1))
    counts.index_add_(0, source_inverse, features.new_ones((source_inverse.numel(), 1)))
    aligned = accumulated[target_inverse] / counts[target_inverse].clamp_min(1)
    present = counts[target_inverse, 0] > 0
    return aligned, present


def _replace_sparse_features(sparse, features):
    if features.shape[0] != sparse.F.shape[0]:
        raise ValueError('Replacement features must preserve the sparse coordinate map')
    return ME.SparseTensor(features=features,
                           coordinate_map_key=sparse.coordinate_map_key,
                           coordinate_manager=sparse.coordinate_manager)


def _valid_coverage(image_bounds, image_valid_mask, dtype, device):
    if image_bounds.ndim != 2 or image_bounds.shape[1] != 2:
        raise ValueError('Image bounds must contain resized width and height per batch item')
    rows = torch.arange(64, device=device, dtype=dtype)
    cols = torch.arange(64, device=device, dtype=dtype)
    widths = (image_bounds[:, 0].to(device=device, dtype=dtype) / 16).unsqueeze(1)
    heights = (image_bounds[:, 1].to(device=device, dtype=dtype) / 16).unsqueeze(1)
    x_coverage = (widths - cols).clamp(0, 1)
    y_coverage = (heights - rows).clamp(0, 1)
    coverage = (y_coverage[:, None, :, None] * x_coverage[:, None, None, :])
    if image_valid_mask is not None:
        mask = image_valid_mask.to(device=device, dtype=dtype)
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape != coverage.shape:
            raise ValueError('Image-valid mask must be [B,1,64,64] or [B,64,64]')
        coverage = coverage * mask.clamp(0, 1)
    return coverage


def masked_sam_pyramid(embedding, image_bounds, image_valid_mask=None):
    if embedding.ndim != 4 or tuple(embedding.shape[1:]) != (256, 64, 64):
        raise ValueError('Expected frozen SAM-L embeddings shaped [B,256,64,64]')
    coverage = _valid_coverage(image_bounds, image_valid_mask, embedding.dtype, embedding.device)
    result = {}
    for level, factor in ((64, 1), (32, 2), (16, 4)):
        if factor == 1:
            pooled_coverage = coverage
            pooled = embedding
        else:
            pooled_coverage = F.avg_pool2d(coverage, factor, stride=factor)
            numerator = F.avg_pool2d(embedding * coverage, factor, stride=factor)
            pooled = numerator / pooled_coverage.clamp_min(1e-6)
        result[level] = (pooled, pooled_coverage)
    return result


def region_token_mask(region_bounds, query_batch, coverage, level):
    if level not in (16, 32, 64):
        raise ValueError('SAM feature level must be 16, 32, or 64')
    if region_bounds.ndim != 3 or tuple(region_bounds.shape[1:]) != (2, 2):
        raise ValueError('Projected regions must have shape [N,2,2]')
    if region_bounds.shape[0] != query_batch.numel():
        raise ValueError('Projected regions and batch indices differ in row count')
    if coverage.ndim != 4 or coverage.shape[1] != 1:
        raise ValueError('Coverage must have shape [B,1,H,W]')
    height, width = coverage.shape[-2:]
    if height != level or width != level:
        raise ValueError('Coverage shape does not match the requested SAM level')
    centers = (torch.arange(level, dtype=region_bounds.dtype, device=region_bounds.device) + 0.5) * (1024.0 / level)
    y = centers.view(1, level, 1)
    x = centers.view(1, 1, level)
    bounds = region_bounds
    xmin, ymin = bounds[:, 0, 0], bounds[:, 0, 1]
    xmax, ymax = bounds[:, 1, 0], bounds[:, 1, 1]
    inside = ((x >= xmin[:, None, None]) & (x <= xmax[:, None, None]) &
              (y >= ymin[:, None, None]) & (y <= ymax[:, None, None]))
    valid = coverage[:, 0].index_select(0, query_batch.long()) > 0
    return inside & valid


def region_token_candidates(region_bounds, query_batch, coverage, level):
    mask = region_token_mask(region_bounds, query_batch, coverage, level)
    weights = mask.to(torch.float32)
    empty = ~mask.flatten(1).any(dim=1)
    finite = torch.isfinite(region_bounds).flatten(1).all(dim=1)
    ordered = (region_bounds[:, 1, 0] >= region_bounds[:, 0, 0]) & (
        region_bounds[:, 1, 1] >= region_bounds[:, 0, 1])
    cell_size = 1024.0 / level
    extent = region_bounds[:, 1] - region_bounds[:, 0]
    small = (extent[:, 0] <= cell_size) & (extent[:, 1] <= cell_size)
    fallback_rows = torch.where(empty & finite & ordered & small)[0]
    if fallback_rows.numel() == 0:
        return mask, weights
    center = (region_bounds[fallback_rows, 0].float() + region_bounds[fallback_rows, 1].float()) * 0.5
    grid = center / cell_size - 0.5
    x0 = torch.floor(grid[:, 0]).long()
    y0 = torch.floor(grid[:, 1]).long()
    dx = grid[:, 0] - x0.to(grid.dtype)
    dy = grid[:, 1] - y0.to(grid.dtype)
    xs = torch.stack((x0, x0 + 1, x0, x0 + 1), dim=1)
    ys = torch.stack((y0, y0, y0 + 1, y0 + 1), dim=1)
    bilinear = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy),
                            (1 - dx) * dy, dx * dy), dim=1)
    in_bounds = (xs >= 0) & (xs < level) & (ys >= 0) & (ys < level)
    safe_x = xs.clamp(0, level - 1)
    safe_y = ys.clamp(0, level - 1)
    frame_index = query_batch[fallback_rows].long()[:, None].expand_as(safe_x)
    coverage_values = coverage[frame_index, 0, safe_y, safe_x]
    bilinear = bilinear * in_bounds.to(bilinear.dtype) * coverage_values.float().clamp(0, 1)
    normalizer = bilinear.sum(dim=1, keepdim=True)
    bilinear = bilinear / normalizer.clamp_min(1e-20)
    flat_indices = safe_y * level + safe_x
    weight_rows = weights[fallback_rows].flatten(1)
    weight_rows.scatter_add_(1, flat_indices, bilinear)
    weights[fallback_rows] = weight_rows.view(-1, level, level)
    mask[fallback_rows] = weight_rows.view(-1, level, level) > 0
    return mask, weights


def unscaled_dot_logits(queries, keys):
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError('Query and key channel dimensions differ')
    return (queries * keys).sum(dim=-1)


class VoxelRegionAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels=256, attention_channels=64, query_chunk_size=64):
        super().__init__()
        self.query = nn.Linear(lidar_channels, attention_channels)
        self.key = nn.Conv2d(image_channels, attention_channels, 1)
        self.value = nn.Conv2d(image_channels, attention_channels, 1)
        self.output = nn.Linear(lidar_channels + attention_channels, lidar_channels)
        self.query_chunk_size = query_chunk_size

    @staticmethod
    def _segment_softmax_context(queries, keys, values, query_indices, token_indices,
                                 token_weights, batch_indices):
        count = queries.shape[0]
        channels = queries.shape[1]
        context = torch.zeros((count, channels), dtype=torch.float32, device=queries.device)
        present = torch.zeros(count, dtype=torch.bool, device=queries.device)
        if query_indices.numel() == 0:
            return context, present
        pair_batches = batch_indices[query_indices]
        pair_query = queries.index_select(0, query_indices).float()
        pair_keys = keys[pair_batches, token_indices].float()
        scores = unscaled_dot_logits(pair_query, pair_keys) + token_weights.float().clamp_min(1e-20).log()
        maxima = scores.new_full((count,), -torch.inf)
        maxima.scatter_reduce_(0, query_indices, scores, reduce='amax', include_self=True)
        exponentials = torch.exp(scores - maxima.index_select(0, query_indices))
        normalizers = scores.new_zeros((count,))
        normalizers.index_add_(0, query_indices, exponentials)
        weights = exponentials / normalizers.index_select(0, query_indices).clamp_min(1e-20)
        pair_values = values[pair_batches, token_indices].float()
        context.index_add_(0, query_indices, pair_values * weights[:, None])
        present[query_indices] = True
        return context.to(dtype=queries.dtype), present

    def forward(self, lidar, image, region_bounds, batch_index, region_valid, coverage, level):
        if image.ndim != 4 or image.shape[1] != self.key.in_channels:
            raise ValueError('SAM feature channels do not match VRA input')
        if lidar.ndim != 2 or lidar.shape[1] != self.query.in_features:
            raise ValueError('LiDAR feature channels do not match VRA input')
        if lidar.shape[0] != batch_index.numel() or lidar.shape[0] != region_bounds.shape[0]:
            raise ValueError('VRA query metadata row counts differ')
        keys_map = self.key(image)
        values_map = self.value(image)
        batch, channels, height, width = keys_map.shape
        keys = keys_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        values = values_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        queries = self.query(lidar)
        context = queries.new_zeros((queries.shape[0], channels))
        supported = torch.zeros(queries.shape[0], dtype=torch.bool, device=queries.device)
        for start in range(0, queries.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, queries.shape[0])
            query_rows = torch.arange(start, stop, device=queries.device)
            finite = torch.isfinite(region_bounds[start:stop]).flatten(1).all(dim=1)
            active = region_valid[start:stop] & finite
            masks, token_weights = region_token_candidates(
                region_bounds[start:stop], batch_index[start:stop], coverage, level)
            masks = masks.reshape(stop - start, height * width)
            token_weights = token_weights.reshape(stop - start, height * width)
            masks &= active[:, None]
            query_indices, token_indices = masks.nonzero(as_tuple=True)
            chunk_context, chunk_supported = self._segment_softmax_context(
                queries[start:stop], keys, values, query_indices, token_indices,
                token_weights[query_indices, token_indices],
                batch_index[start:stop].long())
            context[start:stop] = chunk_context
            supported[start:stop] = chunk_supported
        fused = self.output(torch.cat((lidar, context), dim=1))
        return torch.where(supported[:, None], fused, lidar), supported


class PolarPeriodicSparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, width=1024, kernel_size=3, dilation=1):
        super().__init__()
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels,
                                            kernel_size=self.kernel_size,
                                            dilation=self.dilation,
                                            dimension=3, bias=False)

    def forward(self, sparse):
        padding = ((self.kernel_size - 1) * self.dilation + 1) // 2
        if padding == 0:
            return self.conv(sparse)
        coordinates = sparse.C
        features = sparse.F
        stride = int(sparse.tensor_stride[0])
        offset = torch.tensor([0, self.width, 0, 0], dtype=coordinates.dtype,
                              device=coordinates.device)
        left = coordinates[:, 1] >= self.width // 2 - padding * stride
        right = coordinates[:, 1] < -(self.width // 2 - padding * stride)
        extended_coordinates = torch.cat((coordinates[left] - offset,
                                          coordinates,
                                          coordinates[right] + offset), dim=0)
        extended_features = torch.cat((features[left], features, features[right]), dim=0)
        batch_order = np.argsort(extended_coordinates[:, 0].detach().cpu().numpy(), kind='stable')
        coordinate_order = torch.as_tensor(batch_order, device=coordinates.device)
        feature_order = coordinate_order.to(device=features.device)
        extended = ME.SparseTensor(coordinates=extended_coordinates[coordinate_order],
                                   features=extended_features[feature_order],
                                   coordinate_manager=sparse.coordinate_manager,
                                   tensor_stride=sparse.tensor_stride)
        output = self.conv(extended)
        keep = (output.C[:, 1] >= -self.width // 2) & (output.C[:, 1] < self.width // 2)
        return ME.SparseTensor(coordinates=output.C[keep], features=output.F[keep],
                               coordinate_manager=output.coordinate_manager,
                               tensor_stride=output.tensor_stride)


def _sparse_gate_block(in_channels, width):
    return nn.Sequential(
        PolarPeriodicSparseConv(in_channels, 1, width=width, kernel_size=3),
        ME.MinkowskiBatchNorm(1),
        ME.MinkowskiReLU(inplace=False),
    )


class EquationMMA(nn.Module):
    def __init__(self, channels=(32, 128, 384), output_channels=512, horizontal_res=1024):
        super().__init__()
        if tuple(channels) != (32, 128, 384):
            raise ValueError('The frozen LEADER RPGE taps must have channels (32,128,384)')
        self.pool_f1 = ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=3)
        self.pool_f2 = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.eq4_f1 = _sparse_gate_block(channels[0], horizontal_res)
        self.eq4_f2 = _sparse_gate_block(channels[1], horizontal_res)
        self.eq6_f12 = _sparse_gate_block(channels[0] + channels[1], horizontal_res)
        self.eq6_f3 = _sparse_gate_block(channels[2], horizontal_res)
        self.output_projection = nn.Linear(channels[0] + channels[1] + channels[2], output_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def shared_gate(first_map, second_map):
        return torch.sigmoid(first_map + second_map)

    @staticmethod
    def apply_shared_gate(first, second, gate):
        if gate.shape[0] != first.shape[0] or gate.shape[0] != second.shape[0]:
            raise ValueError('Shared spatial gate rows must match both aligned branches')
        return torch.cat((first * gate, second * gate), dim=1)

    @staticmethod
    def _check_stage_strides(stages):
        expected = ((2, 2, 2), (8, 8, 8), (16, 16, 16))
        actual = tuple(tuple(int(value) for value in stage.tensor_stride) for stage in stages)
        if actual != expected:
            raise ValueError(f'Unexpected RPGE stage strides: {actual}; expected {expected}')

    def forward(self, stages, stage_features, stage_visual_valid, target_coordinates, target_stride, lidar_features):
        if len(stages) != 3 or len(stage_features) != 3 or len(stage_visual_valid) != 3:
            raise ValueError('EquationMMA requires three native RPGE stages')
        self._check_stage_strides(stages)
        sparse_features = [_replace_sparse_features(stage, feature)
                           for stage, feature in zip(stages, stage_features)]
        f1, f2, f3 = sparse_features
        common_stride = _as_stride(f3.tensor_stride, f3.C.device)
        requested_stride = _as_stride(target_stride, f3.C.device)
        if not torch.equal(common_stride, requested_stride):
            raise ValueError('LEADER decoder input and deepest RPGE tap must share a sparse stride')
        f3_keys = torch.unique(_cell_keys(f3.C, common_stride), dim=0, sorted=True)
        target_keys = torch.unique(_cell_keys(target_coordinates, common_stride), dim=0, sorted=True)
        if not torch.equal(f3_keys, target_keys):
            raise ValueError('LEADER decoder and deepest RPGE tap have different sparse supports')
        down1 = self.pool_f1(f1)
        down2 = self.pool_f2(f2)
        gate1 = self.eq4_f1(down1)
        gate2 = self.eq4_f2(down2)
        gate1_on3, _ = align_sparse_features(gate1.F, gate1.C, f3.C, common_stride)
        gate2_on3, _ = align_sparse_features(gate2.F, gate2.C, f3.C, common_stride)
        gate_eq4 = self.shared_gate(gate1_on3, gate2_on3)
        gate1_native, _ = align_sparse_features(gate_eq4, f3.C, f1.C, common_stride)
        gate2_native, _ = align_sparse_features(gate_eq4, f3.C, f2.C, common_stride)
        gated1 = _replace_sparse_features(f1, f1.F * gate1_native)
        gated2 = _replace_sparse_features(f2, f2.F * gate2_native)
        pooled1 = self.pool_f1(gated1)
        pooled2 = self.pool_f2(gated2)
        pooled1_on3, _ = align_sparse_features(pooled1.F, pooled1.C, f3.C, common_stride)
        pooled2_on3, _ = align_sparse_features(pooled2.F, pooled2.C, f3.C, common_stride)
        f12 = torch.cat((pooled1_on3, pooled2_on3), dim=1)
        f12_sparse = _replace_sparse_features(f3, f12)
        gate12 = self.eq6_f12(f12_sparse)
        gate3 = self.eq6_f3(f3)
        gate12_on3, _ = align_sparse_features(gate12.F, gate12.C, f3.C, common_stride)
        gate3_on3, _ = align_sparse_features(gate3.F, gate3.C, f3.C, common_stride)
        gate_eq6 = self.shared_gate(gate12_on3, gate3_on3)
        joined = torch.cat((f12 * gate_eq6, f3.F), dim=1)
        residual_on3 = self.output_projection(joined)
        valid_on3 = torch.zeros((f3.F.shape[0], 1), dtype=f3.F.dtype, device=f3.F.device)
        for valid, stage in zip(stage_visual_valid, stages):
            aligned, _ = align_sparse_features(valid.to(f3.F.dtype).unsqueeze(1), stage.C,
                                               f3.C, common_stride)
            valid_on3 = torch.maximum(valid_on3, aligned)
        residual_on_target, _ = align_sparse_features(residual_on3, f3.C,
                                                       target_coordinates, common_stride)
        valid_on_target, _ = align_sparse_features(valid_on3, f3.C,
                                                    target_coordinates, common_stride)
        if residual_on_target.shape != lidar_features.shape:
            raise ValueError('EquationMMA residual does not align with the LEADER decoder input')
        return lidar_features + residual_on_target * (valid_on_target > 0).to(lidar_features.dtype)


class MaGiCPaperFusion(nn.Module):
    def __init__(self, lidar_channels=512, attention_channels=64, horizontal_res=1024):
        super().__init__()
        self.attention = nn.ModuleList([
            VoxelRegionAttention(32, attention_channels=attention_channels),
            VoxelRegionAttention(128, attention_channels=attention_channels),
            VoxelRegionAttention(384, attention_channels=attention_channels),
        ])
        self.mma = EquationMMA(output_channels=lidar_channels, horizontal_res=horizontal_res)

    def forward(self, lidar, points, coordinates, stride, sam_features, intrinsics,
                camera_from_lidar, recovery, image_bounds, stages=None, voxel_size=0.2,
                horizontal=1024, image_valid_mask=None, return_validity=False):
        if stages is None or len(stages) != 3:
            raise ValueError('Paper MaGiC fusion requires the three RPGE stages')
        if coordinates.shape[0] != lidar.shape[0] or points.shape != (lidar.shape[0], 3):
            raise ValueError('LEADER output features, points, and coordinates are misaligned')
        if sam_features.shape[0] != image_bounds.shape[0]:
            raise ValueError('SAM features and image calibration batch sizes differ')
        pyramid = masked_sam_pyramid(sam_features, image_bounds, image_valid_mask)
        stage_levels = (64, 32, 16)
        stage_visual = []
        stage_outputs = []
        for stage, attention, level in zip(stages, self.attention, stage_levels):
            stage_stride = torch.as_tensor(stage.tensor_stride, device=stage.F.device, dtype=stage.F.dtype)
            batch_index = stage.C[:, 0].long()
            regions, region_valid = project_polar_voxel_regions(
                stage.C, stage_stride, voxel_size, horizontal, intrinsics,
                camera_from_lidar, recovery, image_bounds
            )
            image, coverage = pyramid[level]
            attended, supported = attention(stage.F, image, regions, batch_index,
                                            region_valid, coverage, level)
            stage_outputs.append(attended)
            stage_visual.append(supported)
        output = self.mma(stages, stage_outputs, stage_visual, coordinates, stride, lidar)
        if return_validity:
            target_valid = torch.zeros((coordinates.shape[0],), dtype=torch.bool, device=coordinates.device)
            for valid, stage in zip(stage_visual, stages):
                aligned, _ = align_sparse_features(valid.to(lidar.dtype).unsqueeze(1), stage.C,
                                                   coordinates, stages[2].tensor_stride)
                target_valid |= aligned[:, 0] > 0
            return output, target_valid
        return output
