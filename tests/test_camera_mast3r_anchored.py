import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

import camera_mast3r_anchored as anchored
import camera_reprojection as camera


class CameraMast3rAnchoredTests(unittest.TestCase):
    def test_z_buffer_retains_nearest_anchor_and_raw_identity(self):
        points = np.array([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.02, 0.0, 2.0],
        ])
        raw_indices = np.array([8, 3, 9], dtype=np.int32)
        anchors = anchored.project_lidar_anchors(
            points, raw_indices, np.eye(4), np.eye(4),
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            (100, 100))
        by_pixel = {tuple(pixel): index for pixel, index in zip(anchors['pixel_xy'], anchors['raw_point_indices'])}
        self.assertEqual(by_pixel[(50, 50)], 3)
        self.assertEqual(by_pixel[(51, 50)], 9)
        self.assertEqual(anchors['counts']['visible_pixels'], 2)
        self.assertLessEqual(np.max(np.abs(anchors['quantization_error_px'])), 0.5)

    def test_pixel_center_transform_round_trip(self):
        transform = np.array([[1.5, 0.0, 4.0], [0.0, 1.5, -3.0], [0.0, 0.0, 1.0]])
        points = np.array([[2.25, 7.75], [100.0, 60.0]])
        mapped = anchored.apply_pixel_transform(transform, points, pixel_centers=True)
        restored = anchored.apply_pixel_transform(np.linalg.inv(transform), mapped, pixel_centers=True)
        np.testing.assert_allclose(restored, points, atol=1e-12, rtol=0)

    def test_geometry_filter_deduplicates_each_endpoint(self):
        intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])

        def candidate(query_uv, confidence, train_scan, point_index):
            return {
                'query_uv': query_uv,
                'world_xyz': [0.0, 0.0, 10.0],
                'confidence': confidence,
                'reference_id': 0,
                'raw_point_index': point_index,
                'train_scan': train_scan,
                'reference_pixel_xy': [5, 5],
            }

        rows = [
            candidate([1.0, 1.0], 0.9, 'scan_a', 1),
            candidate([1.0, 1.0], 0.8, 'scan_b', 2),
            candidate([2.0, 1.0], 0.7, 'scan_a', 1),
            candidate([2.0, 1.0], 0.6, 'scan_c', 3),
        ]
        unique, summary = anchored.geometry_filter_then_deduplicate(
            rows, np.eye(4), np.eye(4), intrinsics, (100, 100))
        self.assertEqual([(item['train_scan'], item['raw_point_index']) for item in unique], [
            ('scan_a', 1), ('scan_c', 3)])
        self.assertEqual(summary['duplicate_query_pixel'], 1)
        self.assertEqual(summary['duplicate_landmark'], 1)
        self.assertEqual(summary['deduplicated'], 2)

    def test_geometry_failure_does_not_suppress_valid_duplicate(self):
        intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
        rows = [
            {'query_uv': [1.0, 1.0], 'world_xyz': [2.0, 0.0, 10.0],
             'confidence': 0.99, 'reference_id': 0, 'raw_point_index': 5,
             'train_scan': 'scan_a', 'reference_pixel_xy': [4, 4]},
            {'query_uv': [1.0, 1.0], 'world_xyz': [0.0, 0.0, 10.0],
             'confidence': 0.3, 'reference_id': 1, 'raw_point_index': 5,
             'train_scan': 'scan_a', 'reference_pixel_xy': [4, 4]},
        ]
        unique, summary = anchored.geometry_filter_then_deduplicate(
            rows, np.eye(4), np.eye(4), intrinsics, (100, 100))
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]['confidence'], 0.3)
        self.assertEqual(summary['geometry_rejections']['initial_reprojection_over_12px'], 1)
        self.assertEqual(summary['deduplicated'], 0)

    def test_objective_is_sum_of_huber_terms_plus_normalized_prior(self):
        intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
        value = anchored.summed_refinement_objective(
            np.zeros(6), np.eye(4), np.eye(4), intrinsics,
            np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]),
            np.array([[5.0, 1.0], [9.0, 1.0]]))
        self.assertAlmostEqual(value, 2.0, places=12)

    def test_failed_optimizer_returns_exact_baseline_pose(self):
        intrinsics = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
        points = np.array([
            [-2.0, -1.0, 8.0], [-1.0, 2.0, 9.0], [0.5, -2.0, 10.0],
            [2.0, 1.0, 11.0], [3.0, -0.5, 13.0], [-3.0, 1.5, 15.0],
            [1.5, 2.0, 7.0], [-0.5, 0.5, 12.0],
        ])
        true_uv, _ = camera.project_camera(points, intrinsics)
        query_uv = true_uv + np.array([3.0, -2.0])
        baseline = camera.se3_exp(np.array([0.01, -0.005, 0.004, 0.002, -0.003, 0.001]))
        failed = type('Result', (), {
            'x': np.full(6, 0.25), 'success': False, 'nit': 3,
            'status': 1, 'message': 'iteration limit',
        })()
        with patch.object(anchored, 'minimize', return_value=failed):
            result, summary = anchored.refine_pose_sum(
                baseline, np.eye(4), intrinsics, points, query_uv)
        self.assertEqual(summary['status'], 'fallback')
        self.assertEqual(summary['reason'], 'optimizer_failed')
        np.testing.assert_array_equal(result, baseline)

    def test_synthetic_bounded_refinement_reduces_summed_objective(self):
        intrinsics = np.array([[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]])
        rng = np.random.default_rng(42)
        points = np.column_stack((rng.uniform(-3, 3, 48), rng.uniform(-2, 2, 48), rng.uniform(7, 16, 48)))
        query_uv, _ = camera.project_camera(points, intrinsics)
        baseline = camera.se3_exp(np.array([0.025, -0.018, 0.012, 0.004, -0.003, 0.002]))
        result, summary = anchored.refine_pose_sum(baseline, np.eye(4), intrinsics, points, query_uv)
        self.assertEqual(summary['status'], 'refined')
        self.assertLess(summary['objective_after'], summary['objective_before'])
        self.assertLess(np.linalg.norm(result[:3, 3]), np.linalg.norm(baseline[:3, 3]))
        self.assertTrue(np.all(np.abs(summary['normalized_delta']) <= 1.0))

    def test_nonidentity_extrinsics_and_pixel_scale_jacobian(self):
        intrinsics = np.array([[380.0, 0.0, 320.0], [0.0, 390.0, 240.0], [0.0, 0.0, 1.0]])
        camera_from_body = camera.se3_exp(np.array([0.12, -0.04, 0.08, 0.05, -0.08, 0.03]))
        pose = camera.se3_exp(np.array([1.2, -0.7, 0.4, -0.04, 0.06, 0.02]))
        body_points = np.array([
            [-2.0, -1.0, 8.0], [-1.0, 2.0, 9.0], [0.5, -2.0, 10.0],
            [2.0, 1.0, 11.0], [3.0, -0.5, 13.0], [-3.0, 1.5, 15.0],
            [1.5, 2.0, 7.0], [-0.5, 0.5, 12.0],
        ])
        world_points = camera.transform_points(pose, body_points)
        camera_points = camera.transform_points(camera_from_body, body_points)
        pixels, positive = camera.project_camera(camera_points, intrinsics)
        self.assertTrue(positive.all())
        pixel_scale = np.array([[1.25, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 0.0, 1.0]])
        scaled_pixels = camera.project_camera(camera_points, pixel_scale @ intrinsics)[0]
        scaled_intrinsics = pixel_scale @ intrinsics
        finite_jacobian = camera.normalized_pixel_jacobian(
            pose, camera_from_body, scaled_intrinsics, world_points, scaled_pixels)
        analytic_jacobian = camera.analytic_normalized_pixel_jacobian(
            pose, camera_from_body, scaled_intrinsics, world_points)
        np.testing.assert_allclose(finite_jacobian, analytic_jacobian, atol=2e-5, rtol=2e-5)

    def test_online_runner_does_not_require_query_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'query.png'
            Image.new('RGB', (32, 24), (20, 30, 40)).save(image_path)
            raw_path = root / 'raw_manifest.json'
            raw_path.write_text(json.dumps({'frames': {
                'scans/query.bin': {
                    'image': 'query.png', 'K': [[20, 0, 16], [0, 20, 12], [0, 0, 1]],
                    'T_camera_lidar': np.eye(4).tolist(),
                },
            }}), encoding='utf-8')
            split_path = root / 'split.json'
            split_path.write_text(json.dumps({'splits': {
                'train': [], 'val': ['scans/query.bin'], 'test': [],
            }}), encoding='utf-8')
            baseline_path = root / 'baseline.json'
            baseline_path.write_text(json.dumps({
                'protocol': camera.ONLINE_PROTOCOL,
                'checkpoint_sha256': camera.CHECKPOINT_SHA256,
                'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                'subset': 'val', 'expected_frames': 1,
                'predictions': [{'scan': 'scans/query.bin', 'status': 'ok',
                                 'T_world_body': np.eye(4).tolist()}],
            }), encoding='utf-8')
            baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            baseline_path.with_suffix('.sha256').write_text(baseline_sha + '\n', encoding='ascii')
            map_metadata_path = root / 'map_metadata.json'
            map_metadata_path.write_text('{}\n', encoding='utf-8')
            pose_path = root / 'data' / 'train_scene' / 'val' / 'poses' / 'query.txt'
            self.assertFalse(pose_path.exists())

            class Matcher:
                def bind_data(self, *args):
                    return None

                def match_pair(self, *args):
                    raise AssertionError('No references should be matched')

            fake_map = {'reference_centers_world': np.empty((0, 3))}
            fake_metadata = {'map_sha256': 'synthetic-map'}
            output_dir = root / 'out'
            with patch.object(anchored, 'SPLIT_SHA256', hashlib.sha256(split_path.read_bytes()).hexdigest()), \
                    patch.object(anchored, 'FROZEN_PREDICTION_SHA256', {'val': baseline_sha, 'test': 'unused'}), \
                    patch.object(anchored, 'load_map', return_value=(fake_map, fake_metadata)), \
                    patch.object(camera, 'read_camera_pose', side_effect=AssertionError('query GT read')):
                anchored.run_subset(
                    root / 'data', split_path, raw_path, root / 'map.npz', map_metadata_path,
                    baseline_path, output_dir, 'val', matcher=Matcher())
            prediction = json.loads((output_dir / 'predictions.json').read_text(encoding='utf-8'))
            self.assertEqual(prediction['query_ground_truth_access'],
                             'none; runner inputs contain no query-pose path')
            self.assertEqual(prediction['predictions'][0]['T_world_body'], np.eye(4).tolist())
            self.assertFalse(pose_path.exists())


if __name__ == '__main__':
    unittest.main()
