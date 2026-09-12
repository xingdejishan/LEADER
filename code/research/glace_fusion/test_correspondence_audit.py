import unittest
from pathlib import Path
import tempfile

import numpy as np

from .correspondence_audit import metrics


class AuditTests(unittest.TestCase):
    def test_exact_projection_and_behind_camera(self):
        K = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
        uv = np.array([[4., 4], [12, 4], [4, 12], [12, 12]])
        xyz = np.column_stack([uv, np.ones(4)]) @ np.linalg.inv(K).T * 10
        with tempfile.TemporaryDirectory() as folder:
            scan = Path(folder) / 'absent.bin'
            result = metrics(xyz, uv, K, np.eye(4), np.eye(4), scan)
            self.assertEqual(result['q1'], 1)
            xyz[0] *= -1
            result = metrics(xyz, uv, K, np.eye(4), np.eye(4), scan)
            self.assertEqual(result['q10'], .75)


if __name__ == '__main__':
    unittest.main()
