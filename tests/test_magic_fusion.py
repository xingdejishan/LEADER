import json
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from data.magic_data import CalibratedImageDataset, load_lidar_center
from models.magic_fusion import (MaGiCFusion, VoxelRegionAttention,
                                 project_polar_voxel_regions, project_voxel_centers)


class FakeLidarDataset:
    def __init__(self, scan_path):
        self.pcs = [scan_path]

    def __getitem__(self, index):
        return (None, None, None, None, None)

    def __len__(self):
        return len(self.pcs)

    def get_center_t(self):
        return np.zeros(3)


class FakeStage:
    def __init__(self, features, coordinates, stride):
        self.F = features
        self.C = coordinates
        self.tensor_stride = (stride, stride, stride)


class MaGiCFusionTests(unittest.TestCase):
    def test_projection_undoes_lidar_correction(self):
        points = torch.tensor([[2.0, 0.0, 4.0], [2.0, 0.0, -1.0]])
        recovery = torch.eye(4).unsqueeze(0)
        recovery[0, 0, 3] = -1.0
        intrinsic = torch.tensor([[[100.0, 0.0, 10.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]])
        pixels, valid = project_voxel_centers(
            points, torch.zeros(2, dtype=torch.long), intrinsic,
            torch.eye(4).unsqueeze(0), recovery, torch.tensor([[100.0, 100.0]])
        )
        torch.testing.assert_close(pixels[0], torch.tensor([35.0, 20.0]))
        self.assertEqual(valid.tolist(), [True, False])

    def test_initial_fusion_preserves_pretrained_lidar_features(self):
        torch.manual_seed(3)
        model = MaGiCFusion(lidar_channels=8, image_channels=8)
        lidar = torch.randn(3, 8)
        points = torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.0, 4.0], [0.0, 0.2, 4.0]])
        intrinsic = torch.tensor([[[20.0, 0.0, 512.0], [0.0, 20.0, 512.0], [0.0, 0.0, 1.0]]])
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        result = model(
            lidar, points, coordinates, torch.ones(3), torch.randn(1, 256, 64, 64),
            intrinsic, torch.eye(4).unsqueeze(0), torch.eye(4).unsqueeze(0),
            torch.tensor([[1024.0, 1024.0]])
        )
        torch.testing.assert_close(result, lidar)
        result.sum().backward()
        self.assertIsNotNone(model.aggregate.output.weight.grad)
        with self.assertRaises(ValueError):
            model.image_encoder(torch.randn(1, 3, 64, 64))

    def test_multiscale_groups_pool_lidar_and_geometry_together(self):
        lidar = torch.tensor([[1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
        points = torch.tensor([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0], [7.0, 0.0, 2.0]])
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        pooled, centers, _, inverse = MaGiCFusion._pool_voxels(
            lidar, points, coordinates, torch.ones(3), 2
        )
        torch.testing.assert_close(pooled[inverse][:, 0], torch.tensor([2.0, 2.0, 7.0]))
        torch.testing.assert_close(centers[inverse][:, 0], torch.tensor([2.0, 2.0, 7.0]))

    def test_stage_alignment_uses_coordinates(self):
        features = torch.tensor([[10.0], [20.0], [30.0]])
        source = torch.tensor([[0, 2, 0, 0], [0, 0, 0, 0], [0, 8, 0, 0]])
        target = torch.tensor([[0, 8, 0, 0], [0, 0, 0, 0], [0, 16, 0, 0]])
        result = MaGiCFusion._align_stage(features, source, torch.tensor([2, 2, 2]),
                                           target, torch.tensor([8, 8, 8]))
        torch.testing.assert_close(result[:, 0], torch.tensor([30.0, 15.0, 0.0]))

    def test_polar_region_depends_on_range(self):
        coordinates = torch.tensor([[0, 0, 1, 4], [0, 0, 4, 4]])
        identity = torch.eye(4).unsqueeze(0)
        intrinsic = torch.tensor([[[100.0, 0.0, 512.0], [0.0, 100.0, 512.0], [0.0, 0.0, 1.0]]])
        regions, valid = project_polar_voxel_regions(
            coordinates, torch.ones(3), 1.0, 8, intrinsic, identity, identity,
            torch.tensor([[1024.0, 1024.0]])
        )
        self.assertTrue(valid.all())
        near_width = regions[0, 1, 1] - regions[0, 0, 1]
        far_width = regions[1, 1, 1] - regions[1, 0, 1]
        self.assertGreater(far_width.item(), near_width.item())

    def test_real_stage_path_preserves_initial_features(self):
        torch.manual_seed(7)
        model = MaGiCFusion(lidar_channels=8, image_channels=8)
        lidar = torch.randn(2, 8)
        coordinates = torch.tensor([[0, 0, 1, 8], [0, 8, 2, 8]])
        points = torch.tensor([[1.5, 0.0, 8.5], [2.5, 0.0, 8.5]])
        stages = (
            FakeStage(torch.randn(2, 32), coordinates, 1),
            FakeStage(torch.randn(2, 128), coordinates, 1),
            FakeStage(torch.randn(2, 384), coordinates, 1),
        )
        identity = torch.eye(4).unsqueeze(0)
        intrinsic = torch.tensor([[[20.0, 0.0, 512.0], [0.0, 20.0, 512.0], [0.0, 0.0, 1.0]]])
        output = model(lidar, points, coordinates, torch.ones(3),
                       torch.randn(1, 256, 64, 64), intrinsic, identity, identity,
                       torch.tensor([[1024.0, 1024.0]]), stages=stages, voxel_size=1.0,
                       horizontal=8)
        torch.testing.assert_close(output, lidar)
        output.sum().backward()
        self.assertIsNotNone(model.aggregate.output.weight.grad)

    def test_lidar_center_requires_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = os.path.join(directory, 'pytorch_model.bin')
            with self.assertRaises(FileNotFoundError):
                load_lidar_center(weights)
            with open(os.path.join(directory, 'extra.json'), 'w', encoding='utf-8') as stream:
                json.dump({'center_t': [1, 2, 3]}, stream)
            np.testing.assert_array_equal(load_lidar_center(weights), [1, 2, 3])
            with open(os.path.join(directory, 'extra.json'), 'w', encoding='utf-8') as stream:
                json.dump({'center_t': [1, 'nan', 3]}, stream)
            with self.assertRaises(ValueError):
                load_lidar_center(weights)

    def test_padding_cannot_enter_interpolated_key_value(self):
        attention = VoxelRegionAttention(1, 1, attention_channels=1)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            attention.value.weight.fill_(1)
            attention.fuse.weight[0, 1] = 1
        image = torch.zeros(1, 1, 64, 64)
        changed = image.clone()
        changed[:, :, :, 32:] = 100
        args = (torch.zeros(1, 1), torch.tensor([[507.5, 327.5]]),
                torch.tensor([0]), torch.tensor([True]), 1024,
                torch.tensor([[512.0, 1024.0]]))
        baseline = attention(args[0], image, *args[1:])
        altered = attention(args[0], changed, *args[1:])
        torch.testing.assert_close(altered, baseline)

    def test_manifest_scales_intrinsics_and_requires_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = os.path.join(directory, "NCLT", "scan.bin")
            os.makedirs(os.path.dirname(scan))
            Image.new("RGB", (80, 40), (255, 0, 0)).save(os.path.join(directory, "image.png"))
            record = {
                "image": "image.png",
                "K": [[40, 0, 20], [0, 40, 10], [0, 0, 1]],
                "T_camera_lidar": np.eye(4).tolist(),
                "sam_features": "features.npy",
                "sam_checkpoint_sha256": "a" * 64,
                "resized_size": [1024, 512],
            }
            np.save(os.path.join(directory, "features.npy"), np.zeros((256, 64, 64), dtype=np.float16))
            manifest = os.path.join(directory, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as stream:
                json.dump({"sam_model": "vit_l", "sam_checkpoint_sha256": "a" * 64,
                           "frames": {"NCLT/scan.bin": record}}, stream)
            dataset = CalibratedImageDataset(FakeLidarDataset(scan), directory, manifest)
            sample = dataset[0]
            self.assertEqual(sample[5].shape, (256, 64, 64))
            self.assertEqual(sample[8].tolist(), [1024.0, 512.0])
            torch.testing.assert_close(sample[6][0, 0], torch.tensor(512.0))
            with open(manifest, "w", encoding="utf-8") as stream:
                json.dump({"sam_model": "vit_l", "sam_checkpoint_sha256": "a" * 64,
                           "frames": {}}, stream)
            with self.assertRaises(ValueError):
                CalibratedImageDataset(FakeLidarDataset(scan), directory, manifest)


if __name__ == "__main__":
    unittest.main()
