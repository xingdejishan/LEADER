import unittest

import numpy as np

from .valid_region import make_valid_mask, resize_valid_mask, sample_valid_region, grid_valid_region


class ValidRegionTests(unittest.TestCase):
    def test_mask_comes_from_map_not_image_brightness(self):
        y, x = np.mgrid[:16, :16].astype(np.float32)
        x[:, :4] = -100
        mask = make_valid_mask(x, y, (16, 16), (16, 16))
        self.assertTrue(np.all(mask[:, :4] == 0))
        self.assertTrue(np.all(mask[:, 4:] == 1))

    def test_sampling_uses_pixel_centers_and_rejects_padding(self):
        import torch
        mask = np.ones((16, 17), np.float32)
        mask[4, 4] = 0
        uv = np.array([[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]])
        expected = np.array([False, True, False, True, True, False])
        np.testing.assert_array_equal(sample_valid_region(mask, uv), expected)
        actual = grid_valid_region(torch.from_numpy(mask)[None, None], 2, 3)
        np.testing.assert_array_equal(actual.numpy().reshape(-1), expected)

    def test_fractional_boundary_is_excluded(self):
        mask = np.ones((16, 16), np.float32)
        mask[:, :8] = 0
        resized = resize_valid_mask(mask, 8, 8)
        valid = sample_valid_region(resized, np.column_stack([np.arange(8), np.full(8, 4)]))
        self.assertFalse(valid[3])
        self.assertFalse(valid[4])
        self.assertTrue(valid[6])

    def test_invalid_masks_fail(self):
        for mask in [np.zeros((8, 8)), np.full((8, 8), np.nan), np.ones((2, 2, 2))]:
            with self.assertRaises(ValueError):
                resize_valid_mask(mask, 8, 8)


if __name__ == '__main__':
    unittest.main()
