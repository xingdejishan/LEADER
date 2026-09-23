import json
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from data.magic_data import CalibratedImageDataset
from models.magic_fusion import MaGiCFusion, project_voxel_centers


class FakeLidarDataset:
    def __init__(self, scan_path):
        self.pcs = [scan_path]

    def __getitem__(self, index):
        return (None, None, None, None, None)

    def __len__(self):
        return len(self.pcs)

    def get_center_t(self):
        return np.zeros(3)


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
        intrinsic = torch.tensor([[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]]])
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        result = model(
            lidar, points, coordinates, torch.ones(3), torch.randn(1, 3, 64, 64),
            intrinsic, torch.eye(4).unsqueeze(0), torch.eye(4).unsqueeze(0),
            torch.tensor([[64.0, 64.0]])
        )
        torch.testing.assert_close(result, lidar)
        result.sum().backward()
        self.assertIsNotNone(model.aggregate.output.weight.grad)

    def test_multiscale_groups_pool_lidar_and_geometry_together(self):
        lidar = torch.tensor([[1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
        points = torch.tensor([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0], [7.0, 0.0, 2.0]])
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        pooled, centers, _, inverse = MaGiCFusion._pool_voxels(
            lidar, points, coordinates, torch.ones(3), 2
        )
        torch.testing.assert_close(pooled[inverse][:, 0], torch.tensor([2.0, 2.0, 7.0]))
        torch.testing.assert_close(centers[inverse][:, 0], torch.tensor([2.0, 2.0, 7.0]))

    def test_manifest_scales_intrinsics_and_requires_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = os.path.join(directory, "NCLT", "scan.bin")
            os.makedirs(os.path.dirname(scan))
            Image.new("RGB", (80, 40), (255, 0, 0)).save(os.path.join(directory, "image.png"))
            record = {
                "image": "image.png",
                "K": [[40, 0, 20], [0, 40, 10], [0, 0, 1]],
                "T_camera_lidar": np.eye(4).tolist(),
            }
            manifest = os.path.join(directory, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as stream:
                json.dump({"frames": {"NCLT/scan.bin": record}}, stream)
            dataset = CalibratedImageDataset(FakeLidarDataset(scan), directory, manifest, 64)
            sample = dataset[0]
            self.assertEqual(sample[5].shape, (3, 64, 64))
            self.assertEqual(sample[8].tolist(), [64.0, 32.0])
            torch.testing.assert_close(sample[6][0, 0], torch.tensor(32.0))
            with open(manifest, "w", encoding="utf-8") as stream:
                json.dump({"frames": {}}, stream)
            with self.assertRaises(ValueError):
                CalibratedImageDataset(FakeLidarDataset(scan), directory, manifest, 64)


if __name__ == "__main__":
    unittest.main()
