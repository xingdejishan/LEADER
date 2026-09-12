import tempfile
from pathlib import Path
import unittest

import numpy as np

from .correspondence_filter import CorrespondenceFilter
from .glace_adapter import GLACEOutput


class FilterTests(unittest.TestCase):
    def test_mask_aligns_every_output_field(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mask.npy'
            mask = np.ones((16, 16), dtype=np.float32)
            mask[:, :8] = 0
            np.save(path, mask)
            output = GLACEOutput(None, None, np.array([[4., 4], [12, 4], [4, 12], [12, 12]]),
                np.arange(12).reshape(4, 3), np.eye(3), 2, np.array([True, True, False, False]), (16, 16))
            result = CorrespondenceFilter(path, 'unused').apply(output)
            self.assertTrue(np.array_equal(result.uv, output.uv[[1, 3]]))
            self.assertTrue(np.array_equal(result.xyz_world, output.xyz_world[[1, 3]]))
            self.assertEqual(result.inlier_count, 1)
            self.assertEqual(len(output.uv), 4)


if __name__ == '__main__':
    unittest.main()
