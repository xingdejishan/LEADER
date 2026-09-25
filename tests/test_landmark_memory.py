import unittest

import numpy as np
import torch

from models.landmark_memory import LandmarkMemory
from tools.build_landmark_memory import candidates


class LandmarkTests(unittest.TestCase):
    def test_zero_start_and_multimodal_reference_gradients(self):
        torch.manual_seed(37)
        model = LandmarkMemory()
        lidar, image = torch.randn(3, 512), torch.randn(3, 256)
        rl = torch.randn(3, 4, 512, requires_grad=True)
        rv = torch.randn(3, 4, 256, requires_grad=True)
        error = torch.randn(3, 4, 3, requires_grad=True)
        valid = torch.ones(3, 4, dtype=torch.bool)
        valid[2] = False
        args = (lidar, image, torch.randn(3, 3), rl, rv, torch.randn(3, 4, 3),
                error, torch.randn(3, 4), valid)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        self.assertTrue(torch.equal(model(*args), torch.zeros(3, 3)))
        (model(*args) - .1).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(*args).sum().backward()
        for tensor in (rl, rv, error):
            self.assertGreater(float(tensor.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(model(*args)[2], torch.zeros(3)))
        reordered = (*args[:3], rl.flip(1), rv.flip(1), args[5].flip(1), error.flip(1),
                     args[7].flip(1), valid.flip(1))
        torch.testing.assert_close(model(*args), model(*reordered))

    def test_training_candidate_exclusion_and_reference_frame_cap(self):
        reference = dict(world=np.zeros((20, 3)), frame=np.repeat(np.arange(4), 5),
                         date=np.ones(20), stamp=np.repeat([0, 5_000_000, 20_000_000, 30_000_000], 5))
        index, valid = candidates(reference, np.zeros((1, 3)), np.array([0]), np.array([1]), np.array([0]))
        frames = reference['frame'][index[valid]]
        self.assertEqual(set(frames), {2, 3})
        self.assertLessEqual(max(np.bincount(frames)), 4)


if __name__ == '__main__':
    unittest.main()
