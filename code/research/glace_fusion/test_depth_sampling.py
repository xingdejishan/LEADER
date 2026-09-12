import unittest

import torch

from .depth_sampling import depth_sampling_weights


class DepthSamplingTests(unittest.TestCase):
    def test_mass_and_invalid_pixels(self):
        mask = torch.tensor([1, 1, 1, 1, 0.])
        support = torch.tensor([1, 0, 0, 0, 1.])
        weights = depth_sampling_weights(mask, support)
        self.assertAlmostEqual(weights[0].item(), .5)
        self.assertAlmostEqual(weights[1:4].sum().item(), .5)
        self.assertEqual(weights[4].item(), 0)

    def test_missing_or_full_support(self):
        mask = torch.tensor([1, 1, 0.])
        for support in [torch.zeros(3), torch.ones(3)]:
            self.assertTrue(torch.equal(depth_sampling_weights(mask, support), mask))

    def test_empty_fov_rejected(self):
        with self.assertRaises(ValueError):
            depth_sampling_weights(torch.zeros(3), torch.ones(3))


if __name__ == '__main__':
    unittest.main()
