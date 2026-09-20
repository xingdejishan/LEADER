import unittest

import numpy as np

from lscr_v2_offset_head import OffsetHead, pixel_geometry_features


class LSCRV2Test(unittest.TestCase):
    def test_pixel_geometry_features_are_normalized(self):
        pixels = np.array([[12., 14.]])
        lidar = np.array([[10., 20.]])
        covariance = np.array([[[4., 1.], [1., 9.]]])
        radius = np.array([[4., 6.]])
        values = pixel_geometry_features(pixels, lidar, covariance, radius)
        np.testing.assert_allclose(values[0, :2], [.5, -1.])
        np.testing.assert_allclose(values[0, 2:4], [np.log(.5), np.log(.5)])
        np.testing.assert_allclose(values[0, 4], 1 / 6)

    def test_offset_head_learns_a_small_synthetic_offset(self):
        generator = np.random.default_rng(7)
        correlation = generator.normal(size=(128, 9)).astype(np.float32)
        geometry = generator.normal(size=(128, 5)).astype(np.float32)
        target = np.column_stack((.5 * geometry[:, 0] - .25 * correlation[:, 0],
                                  -.75 * geometry[:, 1] + .1 * correlation[:, 1])).astype(np.float32)
        head = OffsetHead(9, 5, 32, "cpu")
        head.fit(correlation, geometry, target, epochs=80, batch_size=32, learning_rate=3e-3, seed=3)
        prediction, covariance = head.predict(correlation, geometry, batch_size=64)
        self.assertLess(float(np.mean((prediction - target) ** 2)), .02)
        self.assertTrue(np.isfinite(covariance).all())
        self.assertTrue((np.diagonal(covariance, axis1=1, axis2=2) > 0).all())


if __name__ == "__main__":
    unittest.main()
