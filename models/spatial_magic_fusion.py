import MinkowskiEngine as ME
import torch
from torch import nn

from models.magic_fusion import MaGiCFusion


class SpatialFusionDecoder(nn.Module):
    def __init__(self, channels=512, hidden=128, horizontal=1024):
        super().__init__()
        from models.model_mink import SparseResBlock

        self.projections = nn.ModuleList([nn.Linear(channels, hidden) for _ in range(3)])
        self.native = nn.ModuleList([
            SparseResBlock(hidden, hidden, 3, width=horizontal) for _ in range(3)
        ])
        self.decode = nn.ModuleList([
            SparseResBlock(2 * hidden, hidden, 3, width=horizontal) for _ in range(2)
        ])
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def with_features(stage, features):
        return ME.SparseTensor(features=features, coordinate_map_key=stage.coordinate_map_key,
                               coordinate_manager=stage.coordinate_manager)

    @staticmethod
    def align(source, target_coords, target_stride):
        return MaGiCFusion._align_stage(
            source.F, source.C.to(source.F.device),
            torch.as_tensor(source.tensor_stride, device=source.F.device),
            target_coords.to(source.F.device),
            torch.as_tensor(target_stride, device=source.F.device))

    def forward(self, features, stages, coordinates, stride):
        if len(stages) != 3 or len(features) != 3:
            raise ValueError('Expected three native RPGE grids')
        for fine, coarse in zip(stages, stages[1:]):
            if any(c < f or c % f for f, c in zip(fine.tensor_stride, coarse.tensor_stride)):
                raise ValueError('Expected nested fine-to-coarse strides')
        native = [block(self.with_features(stage, projection(feature)))
                  for feature, stage, projection, block in
                  zip(features, stages, self.projections, self.native)]
        decoded = native[2]
        for level in (1, 0):
            target = native[level]
            coarse = self.align(decoded, target.C, target.tensor_stride)
            decoded = self.decode[level](self.with_features(
                target, torch.cat((target.F, coarse), dim=1)))
        return self.output(self.align(decoded, coordinates, stride))


class SpatialMaGiCFusion(MaGiCFusion):
    def __init__(self, lidar_channels=512, image_channels=64, region_size=3,
                 hidden=128, horizontal=1024):
        super().__init__(lidar_channels, image_channels, region_size)
        self.horizontal = horizontal
        self.aggregate = SpatialFusionDecoder(lidar_channels, hidden, horizontal)

    def aggregate_stages(self, features, stages, coordinates, stride):
        return self.aggregate(features, stages, coordinates, stride)

    def forward(self, lidar, points, coordinates, stride, sam_features, intrinsics, camera_from_lidar,
                recovery, image_bounds, stages=None, voxel_size=0.2, horizontal=1024,
                image_valid_mask=None, return_validity=False):
        if stages is None or horizontal != self.horizontal:
            raise ValueError('Native RPGE stages and matching angular width are required')
        return super().forward(
            lidar, points, coordinates, stride, sam_features, intrinsics, camera_from_lidar,
            recovery, image_bounds, stages=stages, voxel_size=voxel_size, horizontal=horizontal,
            image_valid_mask=image_valid_mask, return_validity=return_validity)
