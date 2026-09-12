import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .rotation_complementarity import evidence, rotation_candidates


class RotationComplementarityTests(unittest.TestCase):
    def test_fixed_translation_and_baseline(self):
        initial = np.eye(4)
        initial[:3, 3] = [10, -20, 3]
        initial[:3, :3] = Rotation.from_euler('z', 60, degrees=True).as_matrix()
        poses = rotation_candidates(initial)
        self.assertEqual(len(poses), 125)
        np.testing.assert_array_equal(poses[0], initial)
        np.testing.assert_array_equal(poses[:, :3, 3], np.tile(initial[:3, 3], (125, 1)))
        np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-12)

    def test_camera_resolves_lidar_rotation_ambiguity_without_gt(self):
        rng = np.random.default_rng(2089)
        xyz = rng.uniform([-4, -3, 10], [4, 3, 30], (100, 3))
        K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])
        projected = xyz @ K.T
        camera = dict(xyz=xyz, uv=projected[:, :2] / projected[:, 2:], K=K)
        initial = np.eye(4)
        initial[:3, :3] = Rotation.from_euler('z', 1, degrees=True).as_matrix()
        pool = dict(T_corr=np.eye(4), c_local_all=np.zeros((6, 3)),
                    c_pred_all=np.zeros((6, 3)), center_t=np.zeros(3), u_pred_all=np.zeros(6))
        poses = rotation_candidates(initial)
        scores, far_count = evidence(poses, pool, camera, np.eye(4))
        self.assertEqual(int(np.argmin(scores['lidar'])), 0)
        self.assertEqual(far_count, 50)
        best = poses[np.argmin(scores['joint'])]
        np.testing.assert_allclose(best, np.eye(4), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
