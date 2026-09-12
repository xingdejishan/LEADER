import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .lidar_supervision import camera_targets


class LidarSupervisionTests(unittest.TestCase):
    def test_world_to_augmented_camera_targets(self):
        K = np.array([[100., 0, 30.], [0, 100., 20.], [0, 0, 1.]])
        y, x = np.mgrid[:48, :64]
        pixels = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
        camera = (pixels @ np.linalg.inv(K).T) * 10
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
        T[:3, 3] = [100., -200., 5.]
        world = camera @ T[:3, :3].T + T[:3, 3]
        uv = np.array([[12., 12.], [28., 28.], [44., 36.]])
        target, valid = camera_targets(world, uv, K, np.linalg.inv(T), 48, 64, 3)
        np.testing.assert_array_equal(valid, np.ones(3))
        expected = np.column_stack([uv, np.ones(3)]) @ np.linalg.inv(K).T * 10
        np.testing.assert_allclose(target, expected, atol=1e-6)

    def test_missing_support_does_not_create_depth_label(self):
        target, valid = camera_targets(np.empty((0, 3)), np.array([[4., 4.]]), np.eye(3), np.eye(4), 16, 16, 3)
        self.assertEqual(valid.sum(), 0)
        self.assertEqual(target.sum(), 0)

    def test_hidden_far_surface_uses_front_depth(self):
        K = np.array([[100., 0, 30.], [0, 100., 20.], [0, 0, 1.]])
        y, x = np.mgrid[10:15, 10:15]
        rays = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)]) @ np.linalg.inv(K).T
        world = np.concatenate([rays * 10, rays * 30])
        target, valid = camera_targets(world, np.array([[12., 12.]]), K, np.eye(4), 48, 64, 3)
        self.assertEqual(valid[0], 1)
        self.assertAlmostEqual(float(target[0, 2]), 10., places=5)


if __name__ == '__main__':
    unittest.main()
