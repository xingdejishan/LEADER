import unittest
from pathlib import Path

import numpy as np

from models.server_sort import server_argsort_numpy


FIXTURES = Path(__file__).parent / "fixtures"


class ServerSortTest(unittest.TestCase):
    def test_server_boundary_fixtures(self):
        with np.load(FIXTURES / "server_torch112_sort.npz") as records:
            for key in records.files:
                if key.endswith("_input"):
                    for descending in (False, True):
                        with self.subTest(key=key, descending=descending):
                            actual = server_argsort_numpy(records[key], descending=descending)
                            expected = records[key[:-6] + "_" + str(descending)]
                            np.testing.assert_array_equal(actual, expected)

    def test_server_real_sc2_fixtures(self):
        with np.load(FIXTURES / "server_sc2_sort_trace.npz") as records:
            for key in records.files:
                if "_sort" in key and key.endswith("_input"):
                    with self.subTest(key=key):
                        actual = server_argsort_numpy(records[key])
                        np.testing.assert_array_equal(actual, records[key.replace("_input", "_index")])

    def test_nonfinal_dimension(self):
        with np.load(FIXTURES / "server_torch112_sort.npz") as records:
            actual = server_argsort_numpy(records["40_input"].T, axis=0)
            np.testing.assert_array_equal(actual, records["40_True"].T)

    def test_input_unchanged(self):
        values = np.array([[3, 3, 1, 3, 2]], dtype=np.float32)
        original = values.copy()
        server_argsort_numpy(values)
        np.testing.assert_array_equal(values, original)

    def test_nonfinite_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                server_argsort_numpy(np.array([1, value], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
