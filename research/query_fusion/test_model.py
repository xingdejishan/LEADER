import unittest

import torch

from model import QueryFusion


class QueryTests(unittest.TestCase):
    def inputs(self):
        y, x = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing='ij')
        return torch.randn(4, 512), torch.randn(4, 25, 128), torch.ones(4, 25, dtype=torch.bool), torch.stack([x, y], -1).reshape(25, 2).float()

    def test_gaussian_prior_prefers_center_without_content(self):
        lidar, image, mask, offsets = self.inputs()
        model = QueryFusion(gaussian=True)
        with torch.no_grad():
            model.query.weight.zero_()
            model.query.bias.zero_()
        _, attention = model(lidar, image, mask, offsets, return_attention=True)
        self.assertTrue(torch.all(attention[:, 12] > attention[:, 0]))
        expected = (-.5*offsets.square().sum(-1)).softmax(-1)
        torch.testing.assert_close(attention, expected.expand(4, -1))

    def test_missing_and_masked_values_never_change_features(self):
        lidar, image, mask, offsets = self.inputs()
        model = QueryFusion()
        with torch.no_grad():
            model.output.weight.normal_()
        mask.zero_()
        image.fill_(float('nan'))
        fused, attention = model(lidar, image, mask, offsets, return_attention=True)
        self.assertTrue(torch.equal(fused, lidar))
        self.assertEqual(float(attention.sum()), 0.)

    def test_identity_initialization_gradient_and_point_count(self):
        lidar, image, mask, offsets = self.inputs()
        model = QueryFusion()
        fused = model(lidar, image, mask, offsets)
        self.assertTrue(torch.equal(fused, lidar))
        fused.square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.norm()), 0.)
        self.assertEqual(fused.shape, lidar.shape)

    def test_invalid_neighbors_have_zero_attention(self):
        lidar, image, mask, offsets = self.inputs()
        mask[:, :12] = False
        _, attention = QueryFusion()(lidar, image, mask, offsets, return_attention=True)
        self.assertEqual(float(attention[:, :12].sum()), 0.)
        torch.testing.assert_close(attention.sum(-1), torch.ones(4))


if __name__ == '__main__':
    unittest.main()
