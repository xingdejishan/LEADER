import unittest

import numpy as np

from lscr_refinement import quadratic_subpixel_peak, stable_inverse


class LSCRPrimitiveTest(unittest.TestCase):
    def test_quadratic_peak_and_curvature_covariance(self):
        y, x = np.mgrid[-1:2, -1:2]
        score = 7. - (x - .25) ** 2 - 2 * (y + .4) ** 2
        offset, covariance = quadratic_subpixel_peak(score)
        np.testing.assert_allclose(offset, [.25, -.4], atol=1e-10)
        np.testing.assert_allclose(covariance, [[.5, 0.], [0., .25]], atol=1e-10)

    def test_nonconcave_or_outside_peak_falls_back_to_integer(self):
        y, x = np.mgrid[-1:2, -1:2]
        for score in (x ** 2 + y ** 2, 4. - (x - 2.) ** 2 - y ** 2):
            offset, covariance = quadratic_subpixel_peak(score)
            np.testing.assert_allclose(offset, [0., 0.])
            self.assertIsNone(covariance)

    def test_stable_inverse_regularizes_a_singular_covariance(self):
        inverse = stable_inverse(np.array([[1., 0.], [0., 0.]]), floor=.25)
        np.testing.assert_allclose(inverse, [[1., 0.], [0., 4.]])


if __name__ == "__main__":
    unittest.main()
