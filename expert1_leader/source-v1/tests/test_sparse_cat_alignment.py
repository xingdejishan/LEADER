import unittest

import MinkowskiEngine as ME
import torch

from models.model_mink import MinkowskiSparseTensorCat


class SparseCatAlignmentTest(unittest.TestCase):
    def test_permuted_rows_use_voxel_coordinates(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32)
        first_features = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
        second_features = torch.tensor([[30.], [20.], [10.]], requires_grad=True)
        first = ME.SparseTensor(first_features, coordinates=coords)
        second = ME.SparseTensor(second_features,
                                 coordinates=coords.flip(0))
        result = MinkowskiSparseTensorCat([first, second])
        values = {tuple(coordinate): feature for coordinate, feature in
                  zip(result.C.tolist(), result.F.tolist())}
        self.assertEqual(values[(0, 0, 0, 0)], [1.0, 10.0])
        self.assertEqual(values[(0, 1, 0, 0)], [2.0, 20.0])
        self.assertEqual(values[(0, 2, 0, 0)], [3.0, 30.0])
        result.F.sum().backward()
        self.assertTrue(torch.equal(first_features.grad, torch.ones_like(first_features)))
        self.assertTrue(torch.equal(second_features.grad, torch.ones_like(second_features)))


if __name__ == '__main__':
    unittest.main()
