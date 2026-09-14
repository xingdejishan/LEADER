import unittest

import torch

from fusion import FeatureFusion, sample_patches


class FusionTests(unittest.TestCase):
    def test_patch_centers_and_resize(self):
        features = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        uv = torch.tensor([[6.5, 6.5], [20.5, 6.5], [34.5, 20.5]])
        values = sample_patches(features, uv, (28, 42))
        torch.testing.assert_close(values[:, 0], torch.tensor([0., 1., 5.]), atol=1e-6, rtol=0)
        resized_uv = (uv + .5) * 2 - .5
        torch.testing.assert_close(sample_patches(features, resized_uv, (56, 84)), values)

    def test_missing_image_is_exact_identity_after_training(self):
        torch.manual_seed(2089)
        model = FeatureFusion()
        with torch.no_grad():
            model.residual[-1].weight.normal_()
        lidar = torch.randn(12, 512)
        image = torch.full((12, 128), float('nan'))
        valid = torch.zeros(12, dtype=torch.bool)
        self.assertTrue(torch.equal(model(lidar, image, valid), lidar))

    def test_initial_identity_and_gradient_through_frozen_decoder(self):
        model = FeatureFusion()
        decoder = torch.nn.Linear(512, 3).requires_grad_(False)
        lidar, image = torch.randn(12, 512), torch.randn(12, 128)
        valid = torch.arange(12) < 6
        self.assertTrue(torch.equal(model(lidar, image, valid), lidar))
        decoder(model(lidar, image, valid)).square().mean().backward()
        self.assertGreater(float(model.residual[-1].weight.grad.norm()), 0.)
        self.assertIsNone(decoder.weight.grad)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 83393)


if __name__ == '__main__':
    unittest.main()
