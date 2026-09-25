import unittest

import torch

from models.surface_registration import MultimodalSurfaceRotation


class SurfaceTests(unittest.TestCase):
    def test_geometry_rotation_visual_gradient_and_fixed_translation(self):
        torch.manual_seed(37)
        model = MultimodalSurfaceRotation()
        source = torch.randn(2, 32, 3)*10
        image = torch.randn(2, 32, 32, requires_grad=True)
        baseline = torch.eye(4)[None].repeat(2, 1, 1)
        baseline[:, :3, 3] = torch.randn(2, 3)
        batch = dict(source=source, source_image=image, source_normal=torch.randn(2, 32, 3),
                     reference=source[:, :, None].expand(2, 32, 8, 3)+baseline[:, None, None, :3, 3]+torch.randn(2, 32, 8, 3)*.1,
                     reference_normal=torch.nn.functional.normalize(torch.randn(2, 32, 8, 3), dim=-1),
                     reference_image=torch.randn(2, 32, 8, 32),
                     valid=torch.ones(2, 32, 8, dtype=torch.bool), baseline=baseline)
        batch['valid'][1] = False
        output = model(batch)
        self.assertTrue(torch.equal(output[:, :3, 3], baseline[:, :3, 3]))
        self.assertTrue(torch.equal(output[1], baseline[1]))
        torch.testing.assert_close(torch.linalg.det(output[:, :3, :3]), torch.ones(2))
        output[0, 0, 1].backward()
        self.assertGreater(float(image.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
