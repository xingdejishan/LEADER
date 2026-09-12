import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from .replay_geometry import (accept_on_holdout, camera_scores, candidates_from_pool, grid_cells,
    lidar_arrays, lidar_scores, matched_subset, point_errors, sample_pixels, spatial_holdout)
from .correspondence_replay import reference_input


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        self.shape = (100, 100)

    def test_candidates_do_not_need_ground_truth(self):
        leader = np.eye(4)
        leader[:3, 3] = [2, 3, 4]
        seeds = np.repeat(leader[None], 2, axis=0)
        seeds[1, :3, :3] = Rotation.from_euler('x', 5, degrees=True).as_matrix()
        poses, names = candidates_from_pool(dict(v1_two_stage=np.eye(4), leader=leader, candidate_T_WB=seeds))
        np.testing.assert_array_equal(poses[1, :3, 3], leader[:3, 3])
        np.testing.assert_allclose(poses[2:], seeds)
        self.assertEqual(names, ['v1_two_stage', 'leader', 'sc2_0', 'sc2_1'])

    def test_negative_depth_keeps_denominator(self):
        xyz = np.array([[0., 0, 10], [0, 0, -10]])
        uv = np.array([[50., 50], [50, 50]])
        scores = camera_scores(np.eye(4)[None], xyz, uv, self.K, np.eye(4), self.shape, strict=True)
        self.assertAlmostEqual(scores[0], .5)

    def test_outside_raster_full_penalty_even_when_near_observation(self):
        xyz = np.array([[-5.1, 0, 10]])
        uv = np.array([[1., 50]])
        legacy = camera_scores(np.eye(4)[None], xyz, uv, self.K, np.eye(4), self.shape)
        strict = camera_scores(np.eye(4)[None], xyz, uv, self.K, np.eye(4), self.shape, strict=True)
        self.assertAlmostEqual(legacy[0], .04)
        self.assertEqual(strict[0], 1)

    def test_extrinsic_and_world_frame_invariance(self):
        E = np.eye(4)
        E[:3, :3] = Rotation.from_euler('z', 20, degrees=True).as_matrix()
        E[:3, 3] = [1, -2, .5]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
        T[:3, 3] = [-4, 7, 1]
        camera = np.array([[0., 0, 10], [1, 2, 20]])
        world = camera @ (T @ E)[:3, :3].T + (T @ E)[:3, 3]
        projection = camera @ self.K.T
        uv = projection[:, :2] / projection[:, 2:]
        score = camera_scores(T[None], world, uv, self.K, E, self.shape, strict=True)
        self.assertLess(score[0], 1e-24)

    def test_direction_and_range_errors_are_separate(self):
        targets = np.array([[0., 0, 10], [0, 0, 10]])
        prediction = np.array([[0., 0, 20], [1, 0, 10]])
        uv = np.tile([50., 50], (2, 1))
        errors = point_errors(prediction, uv, self.K, np.eye(4), self.shape, targets)
        self.assertEqual(errors['angle_deg'][0], 0)
        self.assertEqual(errors['along_error_m'][0], 10)
        self.assertEqual(errors['perpendicular_m'][1], 1)
        self.assertGreater(errors['angle_deg'][1], 5)

    def test_sampling_and_holdout_are_fixed_and_disjoint(self):
        uv = np.array([(x, y) for x in range(5, 100, 10) for y in range(5, 100, 10)], dtype=float)
        selected = sample_pixels(uv, self.shape, 48, 2089)
        np.testing.assert_array_equal(selected, sample_pixels(uv, self.shape, 48, 2089))
        a, b = spatial_holdout(uv[selected], self.shape)
        self.assertFalse((a & b).any())
        self.assertTrue((a | b).all())
        self.assertFalse(set(grid_cells(uv[selected][a], self.shape)) & set(grid_cells(uv[selected][b], self.shape)))

    def test_oracle_control_matches_block_counts(self):
        uv = np.array([(x, y) for x in range(5, 100, 10) for y in range(5, 100, 10)], dtype=float)
        oracle = np.arange(len(uv)) % 3 == 0
        matched = matched_subset(uv, oracle, self.shape, 1)
        np.testing.assert_array_equal(np.bincount(grid_cells(uv[oracle], self.shape), minlength=16),
                                      np.bincount(grid_cells(uv[matched], self.shape), minlength=16))

    def test_empty_visual_evidence_rejects_update(self):
        self.assertEqual(len(sample_pixels(np.empty((0, 2)), self.shape, 256, 1)), 0)
        accepted, reason = accept_on_holdout(np.eye(4), np.eye(4), {}, np.empty((0, 3)), np.empty((0, 2)), self.K, np.eye(4), self.shape)
        self.assertFalse(accepted)
        self.assertEqual(reason, 'insufficient_heldout_coverage')

    def test_lidar_ground_correction_and_center_are_applied_once(self):
        Q = np.eye(4)
        Q[:3, :3] = Rotation.from_euler('x', 15, degrees=True).as_matrix()
        Q[:3, 3] = [1, 2, 3]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('y', 20, degrees=True).as_matrix()
        T[:3, 3] = [100, -50, 3]
        body = np.array([[1., 2, 3], [2, 4, 5], [-1, 2, 1]])
        world = body @ T[:3, :3].T + T[:3, 3]
        center = np.array([80, -40, 0])
        pool = dict(T_corr=Q, c_local_all=body @ Q[:3, :3].T + Q[:3, 3],
                    c_pred_all=world - center, center_t=center, u_pred_all=np.zeros(3))
        restored, targets, _ = lidar_arrays(pool)
        np.testing.assert_allclose(restored, body, atol=1e-12)
        np.testing.assert_allclose(targets, world, atol=1e-12)
        self.assertLess(lidar_scores(T[None], pool)[0], 1e-24)

    def test_dense_block_does_not_gain_extra_weight(self):
        uv = np.array([[10., 10], [90, 90]])
        rays = np.column_stack([uv, np.ones(2)]) @ np.linalg.inv(self.K).T
        xyz = rays * 10
        xyz[1, 0] += 1
        score = camera_scores(np.eye(4)[None], xyz, uv, self.K, np.eye(4), self.shape, True, True)
        duplicated_xyz = np.concatenate([np.repeat(xyz[:1], 100, axis=0), xyz[1:]])
        duplicated_uv = np.concatenate([np.repeat(uv[:1], 100, axis=0), uv[1:]])
        duplicate_score = camera_scores(np.eye(4)[None], duplicated_xyz, duplicated_uv, self.K, np.eye(4), self.shape, True, True)
        np.testing.assert_allclose(score, duplicate_score)

    def test_reference_must_match_pixel_identity(self):
        with TemporaryDirectory() as folder:
            uv = np.array([[10., 20], [30, 40]])
            xyz = np.ones((2, 3))
            np.savez(Path(folder) / 'frame.npz', uv=uv[::-1], xyz=xyz, valid=np.ones(2, bool))
            with self.assertRaises(ValueError):
                reference_input(Path(folder), 'frame', uv)
            np.savez(Path(folder) / 'frame.npz', uv=uv, xyz=xyz, valid=np.ones(2, bool))
            np.testing.assert_array_equal(reference_input(Path(folder), 'frame', uv)[0], xyz)


if __name__ == '__main__':
    unittest.main()
