"""Unit tests for the full-scene LiDAR target storage and patch chain.

Run from the repository root:
    python -m unittest research.glace_fusion.test_fullscene_targets -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.glace_fusion.fullscene_lidar_targets import (SCAN_DTYPE, load_scan_world,
                                                           voxel_downsample)
from research.glace_fusion.lidar_supervision import load_camera_targets_rel


def random_se3(rng, max_rot_deg=20, max_trans=3):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rng.uniform(-max_rot_deg, max_rot_deg, 3),
                                    degrees=True).as_matrix()
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def write_scan(path, body_points):
    data = np.zeros(len(body_points), dtype=SCAN_DTYPE)
    for k in ('x', 'y', 'z'):
        data[k] = np.round((body_points[:, 'xyz'.index(k)] + 100.0) / 0.005).astype(np.int64)
    data['x'] = np.clip(data['x'], 0, 65535)
    data['y'] = np.clip(data['y'], 0, 65535)
    data['z'] = np.clip(data['z'], 0, 65535)
    data.tofile(path)


class TestScanLoading(unittest.TestCase):
    def test_decode_and_frames(self):
        rng = np.random.default_rng(0)
        T_BC = random_se3(rng)
        T_WC = random_se3(rng)
        # points placed directly in the camera frame within the FOV band
        cam_pts = np.column_stack([rng.uniform(-5, 5, (200, 2)),
                                   rng.uniform(3, 60, 200)])
        body_pts = cam_pts @ T_BC[:3, :3].T + T_BC[:3, 3]  # camera -> body
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / 's.bin'
            write_scan(scan, body_pts)
            camera, world = load_scan_world(scan, T_WC, T_BC)
        # quantisation to the uint16 grid: 0.005 m steps
        np.testing.assert_allclose(camera, cam_pts, atol=0.01)
        world_expected = cam_pts @ T_WC[:3, :3].T + T_WC[:3, 3]
        np.testing.assert_allclose(world, world_expected, atol=0.01)

    def test_depth_band_filter(self):
        rng = np.random.default_rng(1)
        T_BC = random_se3(rng)
        T_WC = random_se3(rng)
        cam_pts = np.column_stack([rng.uniform(-5, 5, (60, 2)),
                                   np.concatenate([rng.uniform(0.1, 1.9, 20),
                                                   rng.uniform(3, 60, 20),
                                                   rng.uniform(81, 120, 20)])])
        body_pts = cam_pts @ T_BC[:3, :3].T + T_BC[:3, 3]
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / 's.bin'
            write_scan(scan, body_pts)
            camera, _ = load_scan_world(scan, T_WC, T_BC)
        self.assertTrue(((camera[:, 2] > 2) & (camera[:, 2] < 80)).all())


class TestVoxelDownsample(unittest.TestCase):
    def test_deterministic_and_capped(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(-50, 50, (5000, 3))
        a = voxel_downsample(pts, 0.5, 500)
        b = voxel_downsample(pts, 0.5, 500)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(len(a), 500)
        self.assertGreater(len(a), 100)  # a real reduction, not everything
        # every kept point is an original point
        members = (pts[:, None, :] == a[None, :, :]).all(-1).any(1)
        self.assertGreaterEqual(members.sum(), len(a))

    def test_empty(self):
        self.assertEqual(len(voxel_downsample(np.zeros((0, 3)), 0.1, 10)), 0)


class TestRelativeStorage(unittest.TestCase):
    def test_float16_roundtrip_precision(self):
        rng = np.random.default_rng(3)
        cam = np.column_stack([rng.uniform(-8, 8, (1000, 2)), rng.uniform(3, 64, 1000)])
        err = np.abs(cam - cam.astype(np.float16).astype(np.float64)).max()
        self.assertLess(err, 0.05)  # <= ~2^-11 * 64 m

    def test_loader_reconstructs_world(self):
        rng = np.random.default_rng(4)
        T_WC = random_se3(rng)
        T_BC = random_se3(rng)
        cam_pts = np.column_stack([rng.uniform(-4, 4, (400, 2)),
                                   rng.uniform(3, 40, 400)])
        world = cam_pts @ T_WC[:3, :3].T + T_WC[:3, 3]
        K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        uv = np.column_stack([rng.uniform(0, 640, 50), rng.uniform(0, 480, 50)])
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / 'img.npy', cam_pts.astype(np.float16))
            np.save(Path(tmp) / 'img_twc.npy', T_WC.astype(np.float32))
            targets_rel, valid_rel = load_camera_targets_rel(
                tmp, 'img.jpg', uv, K, np.linalg.inv(T_WC), 480, 640)
            # direct world layout must agree with the relative reconstruction
            from research.glace_fusion.lidar_supervision import load_camera_targets
            np.save(Path(tmp) / 'imgw.npy', world)
            targets_direct, valid_direct = load_camera_targets(
                tmp, 'imgw.jpg', uv, K, np.linalg.inv(T_WC), 480, 640)
        np.testing.assert_allclose(valid_rel, valid_direct)
        np.testing.assert_allclose(targets_rel, targets_direct, atol=0.05)

    def test_augmented_pose_projection(self):
        rng = np.random.default_rng(5)
        T_WC = random_se3(rng)
        T_BC = random_se3(rng)
        # dense fronto-parallel wall so 3-NN depths are stable under the gate
        cam_pts = np.column_stack([rng.uniform(-0.5, 0.5, (600, 2)),
                                   np.full(600, 10.0) + rng.uniform(-0.05, 0.05, 600)])
        world = cam_pts @ T_WC[:3, :3].T + T_WC[:3, 3]
        K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        # an "augmented" pose slightly perturbed from the nominal one
        T_CW_aug = np.linalg.inv(T_WC) @ random_se3(rng, max_rot_deg=2, max_trans=0.2)
        # sample uv where the LiDAR wall actually projects (as the buffer does)
        q = world @ T_CW_aug[:3, :3].T + T_CW_aug[:3, 3]
        proj = q @ K.T
        px = proj[:, :2] / proj[:, 2:]
        inside = ((px[:, 0] > 4) & (px[:, 0] < 636) & (px[:, 1] > 4) & (px[:, 1] < 476)
                  & (q[:, 2] > 2) & (q[:, 2] < 80))
        picked = np.flatnonzero(inside)[:40]
        uv = px[picked] + rng.uniform(-1, 1, (len(picked), 2))
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / 'img.npy', cam_pts.astype(np.float16))
            np.save(Path(tmp) / 'img_twc.npy', T_WC.astype(np.float32))
            targets, valid = load_camera_targets_rel(tmp, 'img.jpg', uv, K, T_CW_aug, 480, 640)
        hit = valid > 0
        self.assertGreater(hit.sum(), 0)
        # targets are CAMERA-frame ray targets: pinhole projection alone must
        # return the sampling pixel
        proj = targets[hit] @ K.T
        pixels = proj[:, :2] / proj[:, 2:]
        np.testing.assert_allclose(pixels, uv[hit], atol=0.6)


@unittest.skipUnless(Path('/root/rivermind-data/glace_nclt_rgb_large_20260912/vendor').exists(),
                     'stage-1 vendor not available')
class TestPatchChain(unittest.TestCase):
    def test_chain_applies_cleanly(self):
        from research.glace_fusion.patch_lidar_supervision import patch_lidar_supervision
        from research.glace_fusion.patch_valid_region import patch_valid_region
        with tempfile.TemporaryDirectory() as tmp:
            vendor = Path(tmp) / 'vendor'
            vendor.mkdir()
            src = Path('/root/rivermind-data/glace_nclt_rgb_large_20260912/vendor')
            for path in src.glob('*.py'):
                (vendor / path.name).write_text(path.read_text())
            patch_valid_region(vendor)
            patch_lidar_supervision(vendor, weight=0.5, relative_storage=True)
            trainer = (vendor / 'ace_trainer.py').read_text()
            for marker in ('lidar_folder', 'load_camera_targets_rel', 'lidar_loss',
                           'grid_valid_region'):
                self.assertIn(marker, trainer)
            self.assertIn('+ 0.5 * lidar_loss', trainer)
            self.assertIn('valid_mask.npy', (vendor / 'dataset.py').read_text())
            # patching twice must fail loudly
            with self.assertRaises(ValueError):
                patch_lidar_supervision(vendor, weight=1.0, relative_storage=True)


if __name__ == '__main__':
    unittest.main()
