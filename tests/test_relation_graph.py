import unittest

import torch

from models.relation_graph import VisualRelationTransport, neighbor_edges


class RelationGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.points = torch.randn(33, 3, dtype=torch.float64)
        self.geometry = torch.randn(33, 8, dtype=torch.float64)
        self.visual = torch.randn(33, 8, dtype=torch.float64, requires_grad=True)
        self.module = VisualRelationTransport(8).double()

    def test_edges_are_unique_undirected_without_self(self):
        edges = neighbor_edges(self.points)
        self.assertTrue(bool((edges[:, 0] < edges[:, 1]).all()))
        self.assertEqual(len(edges), len(torch.unique(edges, dim=0)))
        self.assertTrue(bool((torch.bincount(edges.flatten(), minlength=33) >= 16).all()))

    def test_zero_sum_and_constant_geometry(self):
        update = self.module(self.geometry, self.visual, self.points)
        self.assertLess(float(update.sum(0).abs().max()), 1e-12)
        constant = self.module(torch.ones_like(self.geometry), self.visual, self.points)
        self.assertEqual(float(constant.abs().max()), 0.)

    def test_permutation_equivariance(self):
        permutation = torch.randperm(33)
        original = self.module(self.geometry, self.visual, self.points)
        shuffled = self.module(self.geometry[permutation], self.visual[permutation], self.points[permutation])
        torch.testing.assert_close(original[permutation], shuffled, atol=1e-12, rtol=1e-12)

    def test_rigid_geometry_invariance_and_visual_gradient(self):
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        original = self.module(self.geometry, self.visual, self.points)
        changed = self.module(self.geometry, self.visual, self.points@rotation+3)
        torch.testing.assert_close(original, changed, atol=1e-12, rtol=1e-12)
        original.square().sum().backward()
        self.assertGreater(float(self.visual.grad.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(self.visual.grad).all()))

    def test_empty_and_singleton(self):
        for count in (0, 1):
            result = self.module(self.geometry[:count], self.visual[:count], self.points[:count])
            self.assertEqual(result.shape, (count, 8))
            self.assertEqual(float(result.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
