"""Unit tests for the GLACE-independent-branch fusion wiring.

Run from the repository root:
    python -m unittest research.glace_fusion.test_glace_fusion -v

Geometry and interface tests on synthetic data; they do not exercise the
LEADER network or require a GPU (except the optional SC2-PCR hypothesis test,
skipped when torch is unavailable).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.glace_fusion.glace_adapter import (glace_to_leader_frame, pixel_grid_uv,
                                                 solve_pose_pnp)
from research.glace_fusion.packet import (IsotonicCalibrator, camera_support_rate,
                                          diverse_poses, lidar_pool_from_export,
                                          lidar_support_rate, make_evidence_stamps,
                                          make_fusion_evidence)
from research.glace_fusion import lidar_camera_fusion as fcf


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def random_se3(rng, max_rot_deg=30, max_trans=5):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rng.uniform(-max_rot_deg, max_rot_deg, 3),
                                    degrees=True).as_matrix()
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def project(T_WC, K, xyz_world):
    # world -> camera: subtract the world-frame camera centre, then apply R^T
    q = (xyz_world - T_WC[:3, 3]) @ T_WC[:3, :3]
    proj = q @ K.T
    uv = proj[:, :2] / proj[:, 2:]
    return uv, q[:, 2]


class TestPixelGrid(unittest.TestCase):
    def test_cell_centres(self):
        uv = pixel_grid_uv(8, 4, 6)
        self.assertEqual(uv.shape, (24, 2))
        # row-major over (y, x): first row is y=0, u = 8*(x+0.5)
        np.testing.assert_allclose(uv[:6, 0], 8 * (np.arange(6) + 0.5))
        np.testing.assert_allclose(uv[:6, 1], 4.0)
        self.assertEqual(uv[-1, 0], 8 * (5 + 0.5))
        self.assertEqual(uv[-1, 1], 8 * (3 + 0.5))


class TestLidarPoolFrames(unittest.TestCase):
    def test_body_world_recovery(self):
        rng = np.random.default_rng(0)
        T_corr = random_se3(rng)
        T_corr[3] = [0, 0, 0, 1]
        T_WB = random_se3(rng)
        center_t = rng.uniform(-100, 100, 3)
        p_body = rng.uniform(-30, 30, (200, 3))
        p_world = p_body @ T_WB[:3, :3].T + T_WB[:3, 3]
        export = {
            'c_local_all': p_body @ T_corr[:3, :3].T + T_corr[:3, 3],  # leveled frame
            'c_pred_all': p_world - center_t,
            'T_corr': T_corr,
            'center_t': center_t,
            'u_pred_all': rng.uniform(0, 1, 200),
        }
        pool = lidar_pool_from_export(export)
        np.testing.assert_allclose(pool['p_body'], p_body, atol=1e-9)
        np.testing.assert_allclose(pool['p_world'], p_world, atol=1e-9)
        q = lidar_support_rate(T_WB, pool['p_body'], pool['p_world'], 0.3)
        self.assertEqual(q, 1.0)
        T_shifted = T_WB.copy()
        T_shifted[:3, 3] += np.array([3.0, 0.0, 0.0])
        q_bad = lidar_support_rate(T_shifted, pool['p_body'], pool['p_world'], 0.3)
        self.assertLess(q_bad, 0.5)

    def test_support_rate_threshold_semantics(self):
        T = np.eye(4)
        p = np.zeros((4, 3))
        P = np.array([[0, 0, 0], [0.29, 0, 0], [0.31, 0, 0], [np.inf, 0, 0]])
        # strict inequality, NaN handled as outlier
        q = lidar_support_rate(T, p, P, 0.3)
        self.assertEqual(q, 0.5)


class TestCameraSupport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        self.T_BC = random_se3(rng)
        self.T_WB = random_se3(rng)
        self.T_WC = self.T_WB @ self.T_BC
        d = rng.uniform(4, 20, (300, 1))
        xy = rng.uniform(-0.3, 0.3, (300, 2))
        xyz_cam = np.concatenate([xy, np.ones((300, 1))], axis=1) * d
        self.xyz_world = xyz_cam @ self.T_WC[:3, :3].T + self.T_WC[:3, 3]
        self.uv, _ = project(self.T_WC, self.K, self.xyz_world)

    def test_full_support_at_true_pose(self):
        q = camera_support_rate(self.T_WB, self.T_BC, self.K, self.uv, self.xyz_world, 4.0)
        self.assertEqual(q, 1.0)

    def test_negative_depth_counts_as_outlier(self):
        xyz = self.xyz_world.copy()
        T_CW = np.linalg.inv(self.T_WC)
        p = (xyz - T_CW[:3, 3]) @ T_CW[:3, :3]  # camera coords
        p[0, 2] = -5.0  # one point behind the camera
        xyz[0] = p[0] @ self.T_WC[:3, :3].T + self.T_WC[:3, 3]
        q = camera_support_rate(self.T_WB, self.T_BC, self.K, self.uv, xyz, 4.0)
        self.assertAlmostEqual(q, 299 / 300, places=6)


class TestCalibrator(unittest.TestCase):
    def test_identity_when_unfitted(self):
        cal = IsotonicCalibrator()
        self.assertFalse(cal.calibrated)
        self.assertEqual(float(cal(0.7)), 0.7)

    def test_monotone_fit_and_roundtrip(self):
        rng = np.random.default_rng(2)
        q = np.linspace(0, 1, 200)
        success = rng.uniform() < np.clip(q ** 2, 0, 1)
        cal = IsotonicCalibrator().fit(q, success.astype(float))
        self.assertTrue(cal.calibrated)
        out = cal(q)
        self.assertTrue(np.all(np.diff(out) >= -1e-12))
        self.assertTrue(np.all((out >= 0) & (out <= 1)))
        path = Path(__file__).parent / '_cal_roundtrip.json'
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)
        np.testing.assert_allclose(loaded(q), out)
        path.unlink()


class TestFusionExtraHypotheses(unittest.TestCase):
    def _scene(self, seed=3):
        rng = np.random.default_rng(seed)
        K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        T_BC = random_se3(rng)
        T_true = random_se3(rng)
        T_WC = T_true @ T_BC
        # LiDAR pool: 120 inliers + 30 gross outliers
        p_body = rng.uniform(-25, 25, (150, 3))
        P_world = p_body @ T_true[:3, :3].T + T_true[:3, 3]
        P_world[:30] += rng.uniform(8, 15, (30, 3))
        # Camera pool: scene coords on a plane, pixels from true pose
        d = rng.uniform(4, 12, (250, 1))
        xy = rng.uniform(-0.35, 0.35, (250, 2))
        xyz_cam = np.concatenate([xy, np.ones((250, 1))], axis=1) * d
        xyz_world = xyz_cam @ T_WC[:3, :3].T + T_WC[:3, 3]
        uv, _ = project(T_WC, K, xyz_world)
        uv[:50] += rng.uniform(25, 60, (50, 2))  # outliers
        return K, T_BC, T_true, p_body, P_world, uv, xyz_world

    def test_extra_hypothesis_reaches_verified(self):
        K, T_BC, T_true, p_body, P_world, uv, xyz_world = self._scene()
        lidar_stamp, camera_stamp = make_evidence_stamps('t', 1.0, 1.0)
        evidence = make_fusion_evidence(p_body, P_world, uv, xyz_world, K, T_BC,
                                        lidar_stamp, camera_stamp)
        cfg = fcf.FusionConfig(sample_budget=8, seed=5)
        result = fcf.localize(None, None, None, None, evidence=evidence, config=cfg,
                              extra_hypotheses=[T_true.copy()])
        self.assertEqual(result.status, 'EVIDENCE_VERIFIED')
        self.assertTrue(str(result.source).startswith('EXTRA'))
        dt, dr = fcf.pose_distance(result.pose, T_true)
        self.assertLess(dt, 0.1)
        self.assertEqual(result.diagnostics.get('n_extra_hypotheses'), 1)

    def test_none_keeps_fixed_behaviour(self):
        K, T_BC, T_true, p_body, P_world, uv, xyz_world = self._scene(seed=4)
        lidar_stamp, camera_stamp = make_evidence_stamps('t', 1.0, 1.0)
        evidence = make_fusion_evidence(p_body, P_world, uv, xyz_world, K, T_BC,
                                        lidar_stamp, camera_stamp)
        cfg = fcf.FusionConfig(sample_budget=64, seed=5)
        result = fcf.localize(None, None, None, None, evidence=evidence, config=cfg)
        self.assertEqual(result.diagnostics.get('n_extra_hypotheses', 0), 0)
        self.assertEqual(result.status, 'EVIDENCE_VERIFIED')


class TestFrameConversion(unittest.TestCase):
    def test_glace_to_leader(self):
        rng = np.random.default_rng(7)
        T_BC = random_se3(rng)
        T_WB = random_se3(rng)
        T_WC = T_WB @ T_BC
        back = glace_to_leader_frame(T_WC, T_BC)
        np.testing.assert_allclose(back, T_WB, atol=1e-12)

    def test_pnp_recovers_pose(self):
        rng = np.random.default_rng(11)
        K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        T_WC = random_se3(rng)
        d = rng.uniform(4, 15, (400, 1))
        xy = rng.uniform(-0.35, 0.35, (400, 2))
        xyz_cam = np.concatenate([xy, np.ones((400, 1))], axis=1) * d
        xyz_world = xyz_cam @ T_WC[:3, :3].T + T_WC[:3, 3]
        uv, depth = project(T_WC, K, xyz_world)
        uv = uv + rng.normal(0, 0.3, uv.shape)  # subpixel noise
        sol, mask, n, diag = solve_pose_pnp(uv, xyz_world, K)
        self.assertTrue(diag['solved'])
        dt, dr = fcf.pose_distance(sol, T_WC)
        self.assertLess(dt, 0.05)
        self.assertLess(dr, np.deg2rad(0.5))
        self.assertGreater(n, 300)


class TestDiverseHypotheses(unittest.TestCase):
    def test_spread_and_cap(self):
        rng = np.random.default_rng(13)
        poses = [random_se3(rng, max_trans=50) for _ in range(40)]
        picked = diverse_poses(poses, 5)
        self.assertLessEqual(len(picked), 5)
        dts = [fcf.pose_distance(a, b)[0] for i, a in enumerate(picked) for b in picked[i + 1:]]
        self.assertGreater(min(dts), 0.5)


class TestEvidenceValidation(unittest.TestCase):
    def test_stamps_must_match(self):
        K = np.eye(3)
        with self.assertRaises(ValueError):
            fcf._validate_evidence(make_fusion_evidence(
                np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 2)), np.zeros((5, 3)),
                K, np.eye(4), make_evidence_stamps('a', 1.0, 1.0)[0],
                make_evidence_stamps('b', 2.0, 2.0)[1]), fcf.FusionConfig())
        e = make_fusion_evidence(np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 2)),
                                 np.zeros((5, 3)), K, np.eye(4),
                                 *make_evidence_stamps('f', 1.0, 1.0))
        fcf._validate_evidence(e, fcf.FusionConfig())


@unittest.skipUnless(_torch_available(), 'torch unavailable')
class TestSC2Seedwise(unittest.TestCase):
    def test_estimator_returns_pool(self):
        import torch
        from models.sc2pcr import Matcher

        rng = np.random.default_rng(17)
        T = random_se3(rng)
        src = rng.uniform(-20, 20, (1, 600, 3)).astype(np.float32)
        tgt = src @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        tgt[0, :100] += 5.0  # outliers
        src_t, tgt_t = torch.from_numpy(src), torch.from_numpy(tgt)
        m = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10,
                    ratio=0.15, nms_radius=0.1, max_points=3000, k1=30)
        final = m.estimator(src_t, tgt_t)
        final2, seedwise, fitness = m.estimator(src_t, tgt_t, return_hypotheses=True)
        self.assertEqual(final.shape, (1, 4, 4))
        self.assertEqual(final2.shape, (1, 4, 4))
        np.testing.assert_allclose(final.numpy(), final2.numpy(), atol=1e-5)
        self.assertEqual(seedwise.shape[0], 1)
        self.assertEqual(seedwise.shape[2:], (4, 4))
        self.assertEqual(fitness.shape[1], seedwise.shape[1])
        dt = np.linalg.norm(final.numpy()[0, :3, 3] - T[:3, 3])
        self.assertLess(dt, 0.5)


if __name__ == '__main__':
    unittest.main()
