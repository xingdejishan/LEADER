import unittest
import tempfile
from pathlib import Path

import numpy as np

from .correspondence_confidence import features, select, quality_labels, labels, point_probabilities


class ConfidenceTests(unittest.TestCase):
    def test_features_and_selection_need_no_pose_or_labels(self):
        uv = np.array([[4, 4], [12, 4], [4, 12], [12, 12.]])
        xyz = np.column_stack([uv, np.ones(4)])
        result = features(xyz, uv)
        self.assertEqual(result.shape, (4, 13))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.array_equal(select([.2, .8, .4, .3]), [1]))

    def test_selection_ties_are_deterministic(self):
        self.assertTrue(np.array_equal(select(np.ones(8)), [0, 1]))

    def test_depth_error_cannot_pass_with_correct_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            scan = Path(folder) / 'date/velodyne_sync/123.bin'
            scan.parent.mkdir(parents=True)
            dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
            raw = np.zeros(3, dtype=dtype)
            raw['x'], raw['y'], raw['z'] = [19999, 20000, 20001], 20000, 22000
            raw.tofile(scan)
            data = dict(xyz=np.array([[0., 0, 10]]), uv=np.array([[100., 100]]), GT=np.eye(4),
                K=np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]]))
            row = dict(sequence='date', image='123')
            correct, support = quality_labels(data, row, np.eye(4), folder)
            self.assertTrue(correct[0] and support[0])
            data['xyz'][0, 2] = 30
            self.assertTrue(labels(data)[0])
            correct, support = quality_labels(data, row, np.eye(4), folder)
            self.assertTrue(support[0])
            self.assertFalse(correct[0])

    def test_unobserved_label_regions_get_zero_confidence(self):
        class Model:
            def predict_proba(self, X):
                return np.tile([.1, .9], (len(X), 1))
        protocol = dict(supported_grid_cells=[True] + [False] * 63)
        result = point_probabilities(Model(), np.zeros((2, 3)), np.array([[4., 4], [500, 400]]), protocol)
        self.assertTrue(np.allclose(result, [.9, 0]))


if __name__ == '__main__':
    unittest.main()
