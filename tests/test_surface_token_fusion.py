import unittest
from types import SimpleNamespace

import torch

from models.magic_fusion import polar_voxel_centers
from models.surface_token_fusion import SurfaceMaGiCFusion, SurfaceTokenAttention, surface_support


class SurfaceTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.points = torch.tensor([[4.04, 0.006, 4.04], [4.28, 0.038, 4.28],
                                    [4.08, 0.007, 4.30], [4.31, 0.039, 4.08]])
        self.coords = torch.tensor([[0, 0, 20, 20]])
        self.intrinsic = torch.tensor([[[120., 0., 512.], [0., 120., 400.], [0., 0., 1.]]])
        self.identity = torch.eye(4)[None]
        self.bounds = torch.tensor([[1024., 781.]])

    def support(self, points=None, coords=None, mask=None):
        return surface_support([self.points if points is None else points],
                               self.coords if coords is None else coords, [2, 2, 2],
                               self.intrinsic, self.identity, self.identity, self.bounds, mask)

    def test_tokens_are_actual_projected_points(self):
        pixels, offsets, valid = self.support()
        projected = self.points @ self.intrinsic[0].T
        projected = projected[:, :2] / projected[:, 2:]
        distances = torch.cdist(pixels[valid], projected)
        self.assertTrue((distances.min(dim=1).values < 1e-4).all())
        self.assertGreaterEqual(int(valid.sum()), 3)
        self.assertTrue((offsets[valid].abs() <= 0.5).all())

    def test_input_order_and_coordinate_rows_do_not_change_support(self):
        coords = torch.cat((self.coords, torch.tensor([[0, 0, 24, 24]])))
        first = self.support(coords=coords)
        second = self.support(points=self.points.flip(0), coords=coords.flip(0))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b.flip(0))

    def test_batch_identity_and_mask(self):
        coords = torch.cat((self.coords, self.coords + torch.tensor([1, 0, 0, 0])))
        pixels, offsets, valid = surface_support(
            [self.points, self.points], coords, [2, 2, 2],
            self.intrinsic.repeat(2, 1, 1), self.identity.repeat(2, 1, 1),
            self.identity.repeat(2, 1, 1), self.bounds.repeat(2, 1),
            torch.cat((torch.ones(1, 1, 64, 64), torch.zeros(1, 1, 64, 64))))
        self.assertTrue(valid[0].any())
        self.assertFalse(valid[1].any())

    def test_no_support_does_not_change_lidar(self):
        module = SurfaceTokenAttention(8, 4)
        lidar = torch.randn(1, 8)
        pixels, offsets, valid = self.support(mask=torch.zeros(1, 1, 64, 64))
        output, support = module(lidar, torch.randn(1, 4, 64, 64), pixels,
                                 offsets, valid, torch.tensor([0]), self.bounds, None)
        torch.testing.assert_close(output, lidar, atol=0, rtol=0)
        self.assertFalse(support.any())

    def test_real_image_and_position_paths_have_gradients(self):
        module = SurfaceTokenAttention(8, 4)
        pixels, offsets, valid = self.support()
        image = torch.randn(1, 4, 64, 64, requires_grad=True)
        output, _ = module(torch.randn(1, 8), image, pixels, offsets, valid,
                           torch.tensor([0]), self.bounds, None)
        output.square().sum().backward()
        for name in ('query.weight', 'key.weight', 'value.weight',
                     'relative_key.weight', 'relative_value.weight'):
            grad = dict(module.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.norm()), 0, name)
        self.assertGreater(float(image.grad.norm()), 0)

    def test_zero_initialization_and_missing_points(self):
        module = SurfaceMaGiCFusion(lidar_channels=8, image_channels=4)
        stages = [SimpleNamespace(F=torch.randn(1, c),
                                  C=torch.tensor([[0, 0, 20 // s * s, 20 // s * s]]),
                                  tensor_stride=[s, s, s])
                  for c, s in zip((32, 128, 384), (2, 4, 16))]
        lidar = torch.randn(1, 8)
        stride = torch.tensor([2., 2., 2.])
        args = (lidar, polar_voxel_centers(self.coords, stride, 0.2, 1024),
                self.coords, stride, torch.randn(1, 256, 64, 64), self.intrinsic,
                self.identity, self.identity, self.bounds)
        with self.assertRaises(ValueError):
            module(*args, stages=stages)
        output, valid = module(*args, stages=stages, raw_points=[self.points], return_validity=True)
        torch.testing.assert_close(output, lidar, atol=0, rtol=0)
        self.assertTrue(valid.any())
        output.sum().backward()
        self.assertGreater(float(module.aggregate.output.weight.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
