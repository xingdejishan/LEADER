import unittest

import numpy as np

from .fixed_origin_refine import oracle_mask, run_refinement, summarize
from .joint_solver import JointProblem, JointSolverConfig


class FixedOriginRefineTests(unittest.TestCase):
    def test_joint_oracle_rejects_wrong_depth_and_wrong_direction(self):
        target = np.array([[0., 0, 10]] * 4)
        prediction = np.array([[0., 0, 10.5], [0, 0, 30], [0, 0, 10], [np.nan, 0, 10]])
        mask = np.array([True, True, False, True])
        np.testing.assert_array_equal(oracle_mask(mask, prediction, target, 'joint3d'), [True, False, False, False])

    def setUp(self):
        self.points = np.random.default_rng(4).uniform([-2, -2, 8], [2, 2, 12], (40, 3))
        self.target = self.points + [.08, -.03, .02]
        self.K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        self.cfg = JointSolverConfig(camera_scale_px=10.)

    def test_zero_visual_actually_optimizes(self):
        problem = JointProblem(self.points, self.target, np.zeros(40), np.empty((0, 2)),
            np.empty((0, 3)), self.K, np.eye(4), self.cfg)
        pose, info = problem.refine(np.eye(4), camera_weight=0)
        self.assertTrue(info['solver_called'])
        self.assertGreater(info['nfev'], 0)
        self.assertGreater(info['residual_calls'], 0)
        self.assertEqual(info['camera_support'], 0)
        np.testing.assert_allclose(pose[:3, 3], [.08, -.03, .02], atol=1e-5)

    def test_enabled_visual_without_support_is_honestly_skipped(self):
        problem = JointProblem(self.points, self.target, np.zeros(40), np.empty((0, 2)),
            np.empty((0, 3)), self.K, np.eye(4), self.cfg)
        pose, info = problem.refine(np.eye(4))
        self.assertFalse(info['solver_called'])
        self.assertEqual(info['nfev'], 0)
        np.testing.assert_array_equal(pose, np.eye(4))

    def test_ablation_with_no_camera_support_matches_lidar_only(self):
        problem = JointProblem(self.points, self.target, np.zeros(40), np.empty((0, 2)),
            np.empty((0, 3)), self.K, np.eye(4), self.cfg)
        control, _ = problem.refine(np.eye(4), camera_weight=0)
        pose, info = problem.refine(np.eye(4), require_camera_support=False)
        self.assertTrue(info['solver_called'])
        np.testing.assert_array_equal(pose, control)

    def test_fixed_support_excludes_bad_lidar_points(self):
        target = self.target.copy()
        target[:4] += 10
        problem = JointProblem(self.points, target, np.zeros(40), np.empty((0, 2)),
            np.empty((0, 3)), self.K, np.eye(4), self.cfg)
        mask = problem.support(np.eye(4))['lidar_inlier_mask']
        pose, info = problem.refine(np.eye(4), camera_weight=0, lidar_support=mask)
        self.assertEqual(info['lidar_support'], 36)
        np.testing.assert_allclose(pose[:3, 3], [.08, -.03, .02], atol=1e-5)

    def test_arms_share_origin_and_lidar_support_without_gt(self):
        projection = self.points @ self.K.T
        uv = projection[:, :2] / projection[:, 2:]
        pool = dict(c_local_all=self.points, c_pred_all=self.target, u_pred_all=np.zeros(40),
            T_corr=np.eye(4), center_t=np.zeros(3))
        arms = dict(lidar_only=(np.empty((0, 3)), np.empty((0, 2))), prediction_full=(self.target, uv))
        poses, info, support, _ = run_refinement(pool, np.eye(4), np.eye(4), self.K, arms, self.cfg)
        self.assertTrue(all(i['solver_called'] for i in info.values()))
        self.assertTrue(all(i['lidar_support'] == int(support.sum()) for i in info.values()))
        self.assertAlmostEqual(info['lidar_only']['fixed_objective_before']['lidar'],
                               info['prediction_full']['fixed_objective_before']['lidar'])


if __name__ == '__main__':
    unittest.main()
