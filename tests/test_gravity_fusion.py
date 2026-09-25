import unittest

import torch

from models.gravity_fusion import MultimodalGravityFusion


class GravityFusionTest(unittest.TestCase):
    def test_rotation_translation_and_visual_path(self):
        torch.manual_seed(37)
        model = MultimodalGravityFusion()
        baseline = torch.eye(4, dtype=torch.float64)[None].repeat(2, 1, 1)
        baseline[:, :3, 3] = torch.randn(2, 3, dtype=torch.float64)
        batch = dict(baseline=baseline, source=torch.randn(2, 20, 3),
                     normal=torch.nn.functional.normalize(torch.randn(2, 20, 3), dim=-1),
                     image=torch.randn(2, 20, 32), visual_up=torch.tensor([[.03, .02, -1.], [-.01, .04, -1.]]),
                     uncertainty=torch.ones(2, 2)*.01, valid=torch.ones(2, 20, dtype=torch.bool))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(2):
            optimizer.zero_grad()
            output = model(batch)
            output[:, 0, 2].sum().backward()
            optimizer.step()
        self.assertTrue(torch.equal(output[:, :3, 3], baseline[:, :3, 3]))
        rotation = output[:, :3, :3]
        self.assertTrue(torch.allclose(rotation.transpose(1, 2)@rotation, torch.eye(3, dtype=torch.float64)[None], atol=2e-6))
        self.assertTrue(torch.all(torch.linalg.det(rotation) > .99999))
        self.assertGreater(float(model.image.weight.grad.abs().sum()), 0.)
        changed = dict(batch, visual_up=torch.tensor([[.07, .02, -1.], [-.04, .04, -1.]]))
        self.assertGreater(float((model(changed)-model(batch)).abs().max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
