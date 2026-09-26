import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import camera_reprojection as reprojection


def make_transform(rotvec, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, 3] = translation
    return transform


class CameraReprojectionTests(unittest.TestCase):
    def setUp(self):
        self.camera_from_body = make_transform([0.08, -0.04, 0.12], [0.21, -0.09, 0.31])
        self.pose = make_transform([-0.16, 0.11, 0.24], [2.5, -4.0, 1.2])
        source_intrinsics = np.array([[410.0, 0.0, 413.0], [0.0, 402.0, 310.0], [0.0, 0.0, 1.0]])
        pixel_scale = np.diag([0.5, 0.5, 1.0])
        self.intrinsics = pixel_scale @ source_intrinsics
        rng = np.random.default_rng(7301)
        body_points = np.column_stack((rng.uniform(-5, 5, 160), rng.uniform(-3, 3, 160), rng.uniform(8, 30, 160)))
        self.world_points = reprojection.transform_points(self.pose, body_points)
        self.query_uv, valid = reprojection.project_world_points(
            self.pose, self.camera_from_body, self.intrinsics, self.world_points)
        self.assertTrue(valid.all())

    def test_nonunit_extrinsic_pixel_scale_and_jacobian(self):
        finite = reprojection.normalized_pixel_jacobian(
            self.pose, self.camera_from_body, self.intrinsics, self.world_points, self.query_uv)
        analytic = reprojection.analytic_normalized_pixel_jacobian(
            self.pose, self.camera_from_body, self.intrinsics, self.world_points)
        np.testing.assert_allclose(finite, analytic, atol=2e-5, rtol=2e-5)
        self.assertGreater(np.linalg.norm(self.camera_from_body[:3, 3]), 0.1)
        self.assertEqual(self.intrinsics[0, 0], 205.0)

    def test_synthetic_pose_recovery(self):
        initial_delta = np.array([0.08, -0.08, 0.08, 0.01, -0.012, 0.008], dtype=np.float64)
        initial = self.pose @ reprojection.se3_exp(initial_delta)
        refined, report = reprojection.refine_pose(
            initial, self.camera_from_body, self.intrinsics, self.world_points, self.query_uv)
        initial_uv, _ = reprojection.project_world_points(initial, self.camera_from_body,
                                                          self.intrinsics, self.world_points)
        refined_uv, _ = reprojection.project_world_points(refined, self.camera_from_body,
                                                          self.intrinsics, self.world_points)
        initial_reprojection = np.median(np.linalg.norm(initial_uv - self.query_uv, axis=1))
        refined_reprojection = np.median(np.linalg.norm(refined_uv - self.query_uv, axis=1))
        self.assertEqual(report['status'], 'refined')
        self.assertLess(refined_reprojection, initial_reprojection)
        self.assertLess(np.linalg.norm(refined[:3, 3] - self.pose[:3, 3]),
                        np.linalg.norm(initial[:3, 3] - self.pose[:3, 3]))
        rotation_before = Rotation.from_matrix(self.pose[:3, :3].T @ initial[:3, :3]).magnitude()
        rotation_after = Rotation.from_matrix(self.pose[:3, :3].T @ refined[:3, :3]).magnitude()
        self.assertLess(rotation_after, rotation_before)
        self.assertLessEqual(np.max(np.abs(report['normalized_delta'])), 1.0 + 1e-8)

    def test_optimizer_failure_falls_back_even_if_candidate_reduces_objective(self):
        initial_delta = np.array([0.08, -0.08, 0.08, 0.01, -0.012, 0.008], dtype=np.float64)
        initial = self.pose @ reprojection.se3_exp(initial_delta)
        _, successful_report = reprojection.refine_pose(
            initial, self.camera_from_body, self.intrinsics, self.world_points, self.query_uv)
        self.assertEqual(successful_report['status'], 'refined')
        with patch.object(reprojection, 'minimize', return_value=SimpleNamespace(
                x=np.asarray(successful_report['normalized_delta']), success=False,
                status=2, message='simulated optimizer failure', nit=5)):
            result, report = reprojection.refine_pose(
                initial, self.camera_from_body, self.intrinsics, self.world_points, self.query_uv)
        self.assertEqual(report['reason'], 'optimizer_failed')
        self.assertLess(report['objective_after'], report['objective_before'])
        self.assertTrue(np.array_equal(result, initial))

    def test_no_correspondence_returns_pose_bitwise(self):
        original = self.pose.copy()
        result, report = reprojection.refine_pose(
            original, self.camera_from_body, self.intrinsics,
            np.empty((0, 3)), np.empty((0, 2)))
        self.assertEqual(report['reason'], 'fewer_than_6_correspondences')
        self.assertTrue(np.array_equal(result, original))

    def test_reference_keypoint_uses_nearest_depth(self):
        intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
        body_points = np.array([[0.0, 0.0, 5.0], [0.01, 0.0, 4.0]])
        feature_rows, point_rows, _ = reprojection.associate_keypoints(
            np.array([[10.0, 10.0]]), body_points, np.array([7, 8]), np.eye(4), intrinsics,
            (20, 20), radius_px=2.0)
        np.testing.assert_array_equal(feature_rows, [0])
        np.testing.assert_array_equal(point_rows, [1])

    def test_online_runner_never_needs_query_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / 'data'
            data_root.mkdir()
            image = np.random.default_rng(91).integers(0, 256, (80, 120), dtype=np.uint8)
            image_path = root / 'query.png'
            cv2.imwrite(str(image_path), image)
            query_key = 'scans/2012-02-02/velodyne_sync/query.bin'
            split_path = root / 'split.json'
            split_path.write_text(json.dumps({'splits': {'train': ['train_key'], 'val': [query_key], 'test': []}}), encoding='utf-8')
            split_sha = reprojection.sha256_file(split_path)
            old_split_sha = reprojection.SPLIT_SHA256
            old_prediction_sha = reprojection.FROZEN_PREDICTION_SHA256
            reprojection.SPLIT_SHA256 = split_sha
            try:
                manifest_path = root / 'raw_manifest.json'
                manifest_path.write_text(json.dumps({'frames': {query_key: {
                    'image': 'query.png',
                    'K': [[80.0, 0.0, 60.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]],
                    'T_camera_lidar': np.eye(4).tolist(),
                }}}), encoding='utf-8')
                baseline_path = root / 'baseline_predictions.json'
                baseline_pose = np.eye(4).tolist()
                baseline_path.write_text(json.dumps({
                    'protocol': reprojection.ONLINE_PROTOCOL,
                    'checkpoint_sha256': reprojection.CHECKPOINT_SHA256,
                    'split_sha256': split_sha, 'subset': 'val', 'expected_frames': 1,
                    'predictions': [{'scan': query_key, 'status': 'ok', 'T_world_body': baseline_pose}],
                }), encoding='utf-8')
                baseline_sha = reprojection.sha256_file(baseline_path)
                baseline_path.with_suffix('.sha256').write_text(baseline_sha + '\n', encoding='ascii')
                reprojection.FROZEN_PREDICTION_SHA256 = {'val': baseline_sha, 'test': 'unused'}
                map_path = root / 'train_map.npz'
                arrays = {
                    'reference_keys': np.asarray(['train_key'], dtype=np.str_),
                    'reference_centers_world': np.zeros((1, 3), dtype=np.float64),
                    'reference_world_body': np.eye(4, dtype=np.float64)[None],
                    'reference_camera_from_body': np.eye(4, dtype=np.float64)[None],
                    'descriptors': np.empty((0, 128), dtype=np.float32),
                    'keypoints_xy': np.empty((0, 2), dtype=np.float32),
                    'world_points': np.empty((0, 3), dtype=np.float64),
                    'body_points': np.empty((0, 3), dtype=np.float64),
                    'raw_point_indices': np.empty((0,), dtype=np.int32),
                    'reference_ids': np.empty((0,), dtype=np.int32),
                }
                np.savez(map_path, **arrays)
                map_metadata_path = root / 'train_map.json'
                reprojection.write_json(map_metadata_path, {
                    'protocol': reprojection.MAP_PROTOCOL, 'split_sha256': split_sha,
                    'raw_manifest_sha256': reprojection.sha256_file(manifest_path),
                    'training_keys': ['train_key'], 'map_sha256': reprojection.sha256_file(map_path),
                })
                output_dir = root / 'online_output'
                reprojection.run_subset(data_root, split_path, manifest_path, map_path,
                                        map_metadata_path, baseline_path, output_dir, 'val')
                output = reprojection.load_json(output_dir / 'predictions.json')
                self.assertEqual(output['predictions'][0]['T_world_body'], baseline_pose)
                query_pose_path = data_root / 'train_scene' / 'train' / 'poses' / 'query.txt'
                self.assertFalse(query_pose_path.exists())
            finally:
                reprojection.SPLIT_SHA256 = old_split_sha
                reprojection.FROZEN_PREDICTION_SHA256 = old_prediction_sha


if __name__ == '__main__':
    unittest.main()
