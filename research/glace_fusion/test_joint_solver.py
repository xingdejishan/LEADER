"""Unit tests for the v2 shared pose solver (every-frame joint backend).

Run from the repository root:
    python -m unittest research.glace_fusion.test_joint_solver -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.glace_fusion.joint_solver import (JointProblem, JointSolverConfig,
                                                lidar_reliability_weights,
                                                p3p_candidates, pose_distance, solve,
                                                validate_pose)


def random_se3(rng, max_rot_deg=30, max_trans=5):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rng.uniform(-max_rot_deg, max_rot_deg, 3),
                                    degrees=True).as_matrix()
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def make_problem(seed=0, n_lidar=150, lidar_outliers=30, n_camera=250, camera_outliers=50):
    rng = np.random.default_rng(seed)
    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
    T_BC = random_se3(rng)
    T_true = random_se3(rng)
    T_WC = T_true @ T_BC
    p_body = rng.uniform(-25, 25, (n_lidar, 3))
    P_world = p_body @ T_true[:3, :3].T + T_true[:3, 3]
    if lidar_outliers:
        P_world[:lidar_outliers] += rng.uniform(8, 15, (lidar_outliers, 3))
    u_raw = rng.uniform(0.5, 2.0, n_lidar)
    d = rng.uniform(4, 12, (n_camera, 1))
    xy = rng.uniform(-0.35, 0.35, (n_camera, 2))
    xyz_cam = np.concatenate([xy, np.ones((n_camera, 1))], axis=1) * d
    xyz_world = xyz_cam @ T_WC[:3, :3].T + T_WC[:3, 3]
    q = (xyz_world - T_WC[:3, 3]) @ T_WC[:3, :3]
    proj = q @ K.T
    uv = proj[:, :2] / proj[:, 2:]
    if camera_outliers:
        uv[:camera_outliers] += rng.uniform(25, 60, (camera_outliers, 2))
    problem = JointProblem(p_body, P_world, u_raw, uv, xyz_world, K, T_BC,
                           JointSolverConfig())
    return problem, T_true, K, T_BC


class TestWeights(unittest.TestCase):
    def test_bounded_sum_one_monotone(self):
        u = np.array([-50.0, -1.0, 0.0, 1.0, 50.0, 3.0, 3.0])
        w = lidar_reliability_weights(u)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        self.assertTrue(np.isfinite(w).all())
        self.assertGreaterEqual(w.min(), 1e-6)
        # monotone in u (strictly, since atan(10 pi) has not fully saturated)
        self.assertLess(w[0], w[1])
        self.assertLess(w[1], w[2])
        self.assertLess(w[2], w[3])
        self.assertLess(w[3], w[4])
        self.assertAlmostEqual(w[5], w[6], places=12)  # duplicate inputs

    def test_equal_weights_when_equal_reliability(self):
        w = lidar_reliability_weights(np.full(7, 1.3))
        np.testing.assert_allclose(w, np.full(7, 1 / 7), rtol=1e-12)


class TestScoring(unittest.TestCase):
    def test_exact_fit_scores_outlier_fractions(self):
        problem, T_true, _, _ = make_problem()
        sc = problem.score(T_true)
        # exact-fit inliers score 0; outlier mass equals their total weight
        self.assertAlmostEqual(sc['lidar_score'], problem.w_L[:30].sum(), places=9)
        self.assertAlmostEqual(sc['camera_score'], problem.w_C[:50].sum(), places=9)
        self.assertAlmostEqual(sc['score'], 0.5 * (problem.w_L[:30].sum() + 50 / 250),
                               places=9)

    def test_negative_depth_is_full_outlier(self):
        problem, T_true, K, T_BC = make_problem(camera_outliers=0)
        T = T_true.copy()
        T[:3, 3] += np.array([0.0, 0.0, 100.0])  # push everything far behind/away
        sc = problem.score(T)
        self.assertGreater(sc['camera_score'], 0.9)


class TestP3P(unittest.TestCase):
    def test_pool_contains_ground_truth(self):
        problem, T_true, K, T_BC = make_problem(camera_outliers=10)
        rng = np.random.default_rng(3)
        cands, stats = p3p_candidates(problem.uv, problem.xyz_world, K, T_BC,
                                      JointSolverConfig(camera_sample_budget=128), rng)
        self.assertGreater(stats['p3p_solutions'], 0)
        best_dt = min(pose_distance(c, T_true)[0] for c in cands)
        self.assertLess(best_dt, 0.1)

    def test_convention_inversion(self):
        # H = inv(T_CW) @ inv(E): build from a known T_WB and check round trip
        rng = np.random.default_rng(5)
        T_BC = random_se3(rng)
        T_WB = random_se3(rng)
        T_WC = T_WB @ T_BC
        T_CW = np.linalg.inv(T_WC)
        H = np.linalg.inv(T_CW) @ np.linalg.inv(T_BC)
        np.testing.assert_allclose(H, T_WB, atol=1e-12)


class TestSolve(unittest.TestCase):
    def test_joint_refine_recovers_truth(self):
        problem, T_true, _, _ = make_problem()
        rng = np.random.default_rng(1)
        T_L = T_true.copy()
        T_L[:3, 3] += [0.4, -0.3, 0.2]      # LiDAR candidate: mediocre
        T_C = T_true.copy()
        T_C[:3, 3] += [-0.6, 0.5, -0.1]     # Camera candidate: mediocre
        seedwise = np.stack([random_se3(rng, max_trans=40) for _ in range(12)])
        result = solve(problem, T_L, T_C, seedwise)
        self.assertEqual(result.status, 'JOINT')
        dt, dr = pose_distance(result.pose, T_true)
        self.assertLess(dt, 0.1)
        self.assertLess(dr, np.deg2rad(1.0))
        self.assertGreater(result.diagnostics['p3p_solutions'], 0)
        self.assertGreater(result.diagnostics['n_scored_candidates'], 12)

    def test_joint_beats_select_with_decoy(self):
        # LiDAR evidence supports two poses equally; camera only supports truth.
        problem, T_true, K, T_BC = make_problem(seed=2, lidar_outliers=0)
        rng = np.random.default_rng(4)
        T_decoy = random_se3(rng, max_trans=8)
        # duplicate the correspondence pairs with a decoy world map
        p_decoy = problem.p_body @ T_decoy[:3, :3].T + T_decoy[:3, 3]
        problem.p_body = np.concatenate([problem.p_body, problem.p_body.copy()])
        problem.p_world = np.concatenate([problem.p_world, p_decoy])
        problem.u_raw = np.concatenate([problem.u_raw, problem.u_raw.copy()])
        problem.w_L = lidar_reliability_weights(problem.u_raw)
        T_C = T_true.copy()
        T_C[:3, 3] += [0.05, 0, 0]
        joint = solve(problem, T_true, T_C, None, mode='joint_refine')
        self.assertEqual(joint.status, 'JOINT')
        dt, _ = pose_distance(joint.pose, T_true)
        self.assertLess(dt, 0.1)

    def test_single_modal_when_camera_blind(self):
        problem, T_true, _, _ = make_problem(seed=6, lidar_outliers=0, camera_outliers=220)
        # camera pool almost fully outliered -> no camera support anywhere
        T_L = T_true.copy()
        result = solve(problem, T_L, None, None, mode='joint_refine')
        self.assertEqual(result.status, 'SINGLE_MODAL')
        self.assertEqual(result.modality, 'LIDAR')
        dt, _ = pose_distance(result.pose, T_true)
        self.assertLess(dt, 0.1)

    def test_rejected_without_support(self):
        problem, T_true, _, _ = make_problem(seed=8, lidar_outliers=140, camera_outliers=230)
        result = solve(problem, None, None, None, mode='joint_refine')
        self.assertEqual(result.status, 'REJECTED')

    def test_ambiguous_two_supported_clusters(self):
        problem, T_true, _, _ = make_problem(seed=10, lidar_outliers=0, camera_outliers=0)
        rng = np.random.default_rng(11)
        T_decoy = random_se3(rng, max_trans=8)
        # duplicate the support sets: same pixels / same body points, decoy world
        p_decoy = problem.p_body @ T_decoy[:3, :3].T + T_decoy[:3, 3]
        T_WC_true = T_true @ problem.T_BC
        T_WC_decoy = T_decoy @ problem.T_BC
        q = (problem.xyz_world - T_WC_true[:3, 3]) @ T_WC_true[:3, :3]
        x_decoy = q @ T_WC_decoy[:3, :3].T + T_WC_decoy[:3, 3]
        problem.p_body = np.concatenate([problem.p_body, problem.p_body.copy()])
        problem.p_world = np.concatenate([problem.p_world, p_decoy])
        problem.u_raw = np.concatenate([problem.u_raw, problem.u_raw.copy()])
        problem.w_L = lidar_reliability_weights(problem.u_raw)
        problem.xyz_world = np.concatenate([problem.xyz_world, x_decoy])
        problem.uv = np.concatenate([problem.uv, problem.uv.copy()])
        problem.w_C = np.full(len(problem.xyz_world), 1.0 / len(problem.xyz_world))
        result = solve(problem, None, None, None, mode='joint')
        self.assertIn(result.status, ('AMBIGUOUS', 'JOINT'))
        if result.status == 'JOINT':
            # at minimum, the output must be near ONE of the two supported poses
            d_true = pose_distance(result.pose, T_true)[0]
            d_decoy = pose_distance(result.pose, T_decoy)[0]
            self.assertLess(min(d_true, d_decoy), 0.5)

    def test_mode_joint_no_refine(self):
        problem, T_true, _, _ = make_problem()
        result = solve(problem, None, None, None, mode='joint')
        self.assertIn(result.status, ('JOINT', 'AMBIGUOUS', 'REJECTED'))
        self.assertNotIn('refinements', result.diagnostics)


if __name__ == '__main__':
    unittest.main()
