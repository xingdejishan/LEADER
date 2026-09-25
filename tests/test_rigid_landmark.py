import unittest

import torch

from models.rigid_landmark import RigidLandmarkFusion


class RigidTests(unittest.TestCase):
    def test_zero_start_legal_rigid_pose_and_visual_gradient(self):
        torch.manual_seed(37)
        model = RigidLandmarkFusion()
        batch, points, candidates = 2, 12, 4
        source = torch.randn(batch, points, 3) * 10
        baseline = torch.eye(4)[None].repeat(batch, 1, 1)
        point_valid = torch.ones(batch, points, dtype=torch.bool)
        point_valid[1] = False
        image = torch.randn(batch*points, 256, requires_grad=True)
        inputs = (torch.randn(batch*points, 512), image, torch.randn(batch*points, 3),
                  torch.randn(batch*points, candidates, 512), torch.randn(batch*points, candidates, 256),
                  torch.randn(batch*points, candidates, 3), torch.randn(batch*points, candidates, 3),
                  torch.randn(batch*points, candidates), torch.ones(batch*points, candidates, dtype=torch.bool))
        result = model(source, baseline, point_valid, inputs)
        self.assertTrue(torch.equal(result, baseline))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        (result[:, :3, 3] - .1).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        result = model(source, baseline, point_valid, inputs)
        self.assertTrue(torch.equal(result[1], baseline[1]))
        rotation = result[:, :3, :3]
        torch.testing.assert_close(rotation.transpose(1, 2) @ rotation, torch.eye(3)[None].repeat(batch, 1, 1))
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(batch))
        result[:, :3, 3].sum().backward()
        self.assertGreater(float(image.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
