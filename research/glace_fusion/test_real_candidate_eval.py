import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_separability import camera_evidence
from .real_candidate_eval import dominance_counts, pose_errors, score_camera
from .balanced_candidate_eval import lidar_scores
from .joint_solver import JointProblem, JointSolverConfig, lidar_reliability_weights


class RealCandidateTests(unittest.TestCase):
    def test_camera_batch_matches_reference_with_nontrivial_extrinsic(self):
        E = np.eye(4)
        E[:3, :3] = Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix()
        E[:3, 3] = [.1, -.2, 1.3]
        body = np.eye(4)
        body[:3, :3] = Rotation.from_euler('xyz', [.1, .3, -.2]).as_matrix()
        body[:3, 3] = [20., 100., -5.]
        K = np.array([[200., 0, 320], [0, 200., 240.], [0, 0, 1.]])
        uv = np.array([[100., 120.], [300., 220.], [400., 300.]])
        camera = np.column_stack([uv, np.ones(3)]) @ np.linalg.inv(K).T * 10
        world_camera = body @ E
        world = camera @ world_camera[:3, :3].T + world_camera[:3, 3]
        wrong = body.copy()
        wrong[:3, 3] += [1., -.2, .4]
        poses = np.stack([body, wrong])
        actual = score_camera(poses, E, world, uv, K)
        expected = [camera_evidence(world, uv, K, p @ E)['S_C'] for p in poses]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertLess(actual[0], actual[1])

    def test_dominance_excludes_translation_rotation_tradeoffs(self):
        errors = np.array([[.1, .2], [.4, .6], [.05, 2.]])
        result = dominance_counts(errors, np.array([.2, .7, .1]))
        self.assertEqual(result, dict(pairs=1, correct=1, ties=0))

    def test_pose_error_units(self):
        p = np.eye(4)
        p[:3, 3] = [3., 4., 0]
        p[:3, :3] = Rotation.from_euler('z', 5, degrees=True).as_matrix()
        np.testing.assert_allclose(pose_errors(p[None], np.eye(4)), [[5., 5.]], atol=1e-10)

    def test_lidar_batch_matches_existing_joint_score(self):
        body = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
        pose[:3, 3] = [.1, -.2, .3]
        world = body @ pose[:3, :3].T + pose[:3, 3]
        u = np.array([-1., .1, .5, 2.])
        shifted = pose.copy()
        shifted[0, 3] += .15
        candidates = np.stack([pose, shifted])
        problem = JointProblem(body, world, u, np.array([[0., 0.]]), np.array([[0., 0., 10.]]),
            np.eye(3), np.eye(4), JointSolverConfig(camera_scale_px=10.))
        actual = lidar_scores(candidates, body, world, lidar_reliability_weights(u))
        np.testing.assert_allclose(actual, [problem.score(p)['lidar_score'] for p in candidates], atol=1e-12)

    def test_rotation_metric_rejects_scale_roundoff_bias(self):
        p = np.eye(4)
        p[:3, :3] = Rotation.from_euler('z', 1., degrees=True).as_matrix() * (1 + 1e-6)
        np.testing.assert_allclose(pose_errors(p[None], np.eye(4)), [[0., 1.]], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
