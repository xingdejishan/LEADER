import unittest

import MinkowskiEngine as ME
import torch

from models.model_mink import SparseConvPadding
from models.spatial_magic_fusion import SpatialFusionDecoder, SpatialMaGiCFusion


class SpatialFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.device = 'cuda'

    def sparse(self, coords, channels=8, stride=1, values=None):
        features = torch.randn(len(coords), channels, device=self.device) if values is None else values
        return ME.SparseTensor(features=features,
                               coordinates=torch.tensor(coords, dtype=torch.int32, device=self.device),
                               tensor_stride=stride)

    def stages(self, channels=(8, 8, 8)):
        return [self.sparse([[batch, angle, 8, 8]
                             for batch in (0, 1) for angle in range(-8, 8, stride)], channel, stride)
                for stride, channel in zip((1, 2, 4), channels)]

    def test_periodic_neighbors_and_batch_isolation(self):
        layer = SparseConvPadding(16, ME.MinkowskiConvolution(1, 1, 3, dimension=3)).cuda()
        with torch.no_grad():
            layer.conv.kernel.fill_(1)
        coords = [[0, -8, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0], [1, -8, 0, 0]]
        values = torch.tensor([[1.], [2.], [4.], [8.]], device=self.device, requires_grad=True)
        source = self.sparse(coords, values=values)
        output = layer(source).features_at_coordinates(source.C.float())
        torch.testing.assert_close(output[:, 0], torch.tensor([3., 3., 4., 8.], device=self.device))
        output[0].sum().backward()
        torch.testing.assert_close(values.grad[:, 0], torch.tensor([1., 1., 0., 0.], device=self.device))

    def test_native_grid_reordering_and_cross_scale_gradients(self):
        stages = self.stages()
        decoder = SpatialFusionDecoder(8, hidden=4, horizontal=16).cuda().eval()
        with torch.no_grad():
            decoder.output.weight.normal_(std=0.1)
        features = [stage.F.detach().requires_grad_() for stage in stages]
        target = stages[0].C.flip(0)
        output = decoder(features, stages, target, torch.ones(3, device=self.device))
        reordered = [ME.SparseTensor(features=feature.flip(0), coordinates=stage.C.flip(0),
                                     tensor_stride=stage.tensor_stride)
                     for feature, stage in zip(features, stages)]
        alternate = decoder([s.F for s in reordered], reordered, target, torch.ones(3, device=self.device))
        torch.testing.assert_close(output, alternate, atol=2e-5, rtol=2e-5)
        output.square().sum().backward()
        for feature in features:
            self.assertGreater(float(feature.grad.abs().sum()), 0)

    def test_alignment_negative_coords_anisotropic_stride_and_batches(self):
        source = self.sparse([[0, -4, 8, 0], [0, 0, 8, 0], [1, -4, 8, 0]],
                             stride=(4, 2, 1), values=torch.tensor([[3.], [5.], [9.]], device=self.device))
        target = torch.tensor([[1, -3, 8, 0], [0, -1, 8, 0], [0, 1, 8, 0], [0, 8, 8, 0]], device=self.device)
        output = SpatialFusionDecoder.align(source, target, (1, 1, 1))
        torch.testing.assert_close(output[:, 0], torch.tensor([9., 3., 5., 0.], device=self.device))

    def test_zero_initialization_final_mask_and_visual_gradients(self):
        stages = self.stages((32, 128, 384))
        model = SpatialMaGiCFusion(8, 8, hidden=4, horizontal=16).cuda().train()
        coords = stages[0].C
        lidar = torch.randn(len(coords), 8, device=self.device)
        points = torch.tensor([[0., 0., 8.]] * len(coords), device=self.device)
        points[0, 2] = -8
        identity = torch.eye(4, device=self.device)[None].repeat(2, 1, 1)
        intrinsic = torch.tensor([[[20., 0., 512.], [0., 20., 512.], [0., 0., 1.]]], device=self.device).repeat(2, 1, 1)
        mask = torch.ones(2, 1, 64, 64, device=self.device)
        mask[1].zero_()
        inputs = (lidar, points, coords, torch.ones(3, device=self.device),
                  torch.randn(2, 256, 64, 64, device=self.device), intrinsic, identity, identity,
                  torch.full((2, 2), 1024., device=self.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        output, valid = model(*inputs, stages=stages, horizontal=16, image_valid_mask=mask, return_validity=True)
        self.assertTrue(torch.equal(output, lidar))
        self.assertFalse(bool(valid[0]))
        self.assertFalse(bool(valid[coords[:, 0] == 1].any()))
        output.square().mean().backward()
        self.assertGreater(float(model.aggregate.output.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output, valid = model(*inputs, stages=stages, horizontal=16, image_valid_mask=mask, return_validity=True)
        self.assertTrue(torch.equal(output[~valid], lidar[~valid]))
        output.square().mean().backward()
        for module in (model.image_encoder, model.attention, model.aggregate.native, model.aggregate.decode):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)


if __name__ == '__main__':
    unittest.main()
