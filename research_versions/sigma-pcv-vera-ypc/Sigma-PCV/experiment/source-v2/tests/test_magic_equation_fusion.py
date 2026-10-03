import unittest

import MinkowskiEngine as ME
import torch

from models.magic_paper_fusion import (EquationMMA, MaGiCPaperFusion, PolarPeriodicSparseConv,
                                       align_sparse_features,
                                       masked_sam_pyramid, region_token_candidates,
                                       region_token_mask,
                                       unscaled_dot_logits)
from models.model_mink import LEADER


class EquationFusionTests(unittest.TestCase):
    def test_attention_logits_are_unscaled_dot_products(self):
        query = torch.tensor([[2.0, 3.0]])
        keys = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
        logits = unscaled_dot_logits(query.expand_as(keys), keys)
        torch.testing.assert_close(logits, torch.tensor([14.0, 12.0]))

    def test_region_selects_all_valid_tokens_and_excludes_padding(self):
        coverage = torch.ones(1, 1, 64, 64)
        coverage[:, :, :, 32:] = 0
        region = torch.tensor([[[0.0, 0.0], [1024.0, 1024.0]]])
        mask = region_token_mask(region, torch.tensor([0]), coverage, 64)
        self.assertEqual(int(mask.sum()), 64 * 32)
        self.assertTrue(mask[:, :, :32].all())
        self.assertFalse(mask[:, :, 32:].any())

    def test_region_uses_resized_canvas_cell_centers_and_empty_bounds(self):
        coverage = torch.ones(1, 1, 32, 32)
        region = torch.tensor([[[15.0, 15.0], [17.0, 17.0]]])
        mask = region_token_mask(region, torch.tensor([0]), coverage, 32)
        self.assertEqual(int(mask.sum()), 1)
        region = torch.tensor([[[0.0, 0.0], [15.0, 15.0]]])
        mask = region_token_mask(region, torch.tensor([0]), coverage, 32)
        self.assertEqual(int(mask.sum()), 0)

    def test_small_valid_region_uses_masked_bilinear_neighbors(self):
        coverage = torch.ones(1, 1, 64, 64)
        region = torch.tensor([[[511.0, 335.0], [513.0, 337.0]]])
        mask, weights = region_token_candidates(region, torch.tensor([0]), coverage, 64)
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(float(weights.sum()), 1.0)
        coverage[:, :, :, 32:] = 0
        mask, weights = region_token_candidates(region, torch.tensor([0]), coverage, 64)
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(float(weights.sum()), 1.0)
        self.assertEqual(float(weights[:, :, 32:].sum()), 0.0)

    def test_sam_pyramid_masks_partial_tokens_before_pooling(self):
        embedding = torch.ones(1, 256, 64, 64)
        embedding[:, :, :, 32:] = 100
        bounds = torch.tensor([[512.0, 1024.0]])
        pyramid = masked_sam_pyramid(embedding, bounds)
        self.assertTrue(torch.equal(pyramid[32][0][:, :, :, 16:], torch.zeros_like(
            pyramid[32][0][:, :, :, 16:])))
        torch.testing.assert_close(pyramid[32][0][:, :, :, :16], torch.ones_like(
            pyramid[32][0][:, :, :, :16]))
        self.assertEqual(float(pyramid[32][1][:, :, :, 16:].sum()), 0.0)

    def test_sparse_alignment_is_keyed_and_differentiable(self):
        source_coordinates = torch.tensor([[0, 8, 0, 0], [0, 0, 0, 0], [0, 16, 0, 0]])
        source_features = torch.tensor([[30.0], [10.0], [50.0]], requires_grad=True)
        target_coordinates = torch.tensor([[0, 16, 0, 0], [0, 0, 0, 0], [0, 8, 0, 0]])
        aligned, present = align_sparse_features(source_features, source_coordinates,
                                                 target_coordinates, (1, 1, 1))
        torch.testing.assert_close(aligned[:, 0], torch.tensor([50.0, 10.0, 30.0]))
        self.assertTrue(present.all())
        aligned.sum().backward()
        self.assertTrue(torch.equal(source_features.grad, torch.ones_like(source_features)))

    def test_shared_gate_is_one_map_for_both_branches(self):
        first_gate = torch.tensor([[0.5], [-1.0]])
        second_gate = torch.tensor([[1.0], [0.25]])
        gate = EquationMMA.shared_gate(first_gate, second_gate)
        first = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        second = torch.tensor([[10.0], [12.0]])
        pair = EquationMMA.apply_shared_gate(first, second, gate)
        torch.testing.assert_close(pair[:, :2], first * gate)
        torch.testing.assert_close(pair[:, 2:], second * gate)

    def test_mma_spatial_gate_uses_neighbor_voxels(self):
        mma = EquationMMA().eval()
        block = mma.eq6_f3
        self.assertEqual(block[0].conv.kernel.shape[0], 27)
        with torch.no_grad():
            block[0].conv.kernel.fill_(1.0 / 384)
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 16, 0, 0]], dtype=torch.int32)
        baseline = ME.SparseTensor(torch.zeros(2, 384), coordinates=coordinates,
                                   tensor_stride=16)
        changed_features = torch.zeros(2, 384)
        changed_features[1] = 1
        changed = ME.SparseTensor(changed_features, coordinates=coordinates,
                                  tensor_stride=16)
        baseline_output = block(baseline)
        changed_output = block(changed)
        baseline_by_key = {tuple(key): value for key, value in
                           zip(baseline_output.C.tolist(), baseline_output.F[:, 0].tolist())}
        changed_by_key = {tuple(key): value for key, value in
                          zip(changed_output.C.tolist(), changed_output.F[:, 0].tolist())}
        self.assertGreater(changed_by_key[(0, 0, 0, 0)], baseline_by_key[(0, 0, 0, 0)])

    def test_polar_periodic_gate_connects_angular_seam(self):
        convolution = PolarPeriodicSparseConv(1, 1, width=32, kernel_size=3)
        with torch.no_grad():
            convolution.conv.kernel.fill_(1.0)
        coordinates = torch.tensor([[0, 15, 0, 0], [0, -16, 0, 0]], dtype=torch.int32)
        features = torch.tensor([[0.0], [1.0]])
        sparse = ME.SparseTensor(features, coordinates=coordinates, tensor_stride=1)
        output = convolution(sparse)
        values = {tuple(key): value for key, value in zip(output.C.tolist(), output.F[:, 0].tolist())}
        self.assertGreater(values[(0, 15, 0, 0)], 0.0)

    def test_finetuning_freezes_pretrained_batchnorm_statistics_only(self):
        model = LEADER(magic=True)
        model.train()
        encoder_bn = next(module for module in model.encoder.modules()
                          if isinstance(module, ME.MinkowskiBatchNorm))
        decoder_batchnorms = [module for module in model.decoder.modules()
                              if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        decoder_norm = next(module for module in model.decoder.modules()
                            if isinstance(module, torch.nn.LayerNorm))
        fusion_bn = model.magic_fusion.mma.eq4_f1[1]
        self.assertFalse(encoder_bn.training)
        self.assertEqual(decoder_batchnorms, [])
        self.assertTrue(fusion_bn.training)
        self.assertTrue(encoder_bn.bn.weight.requires_grad)
        self.assertTrue(decoder_norm.weight.requires_grad)

    def test_zero_initialized_residual_preserves_l0_and_opens_gradients_after_step(self):
        torch.manual_seed(31)
        stage1_coordinates = torch.tensor([[0, x, 0, 0] for x in range(0, 32, 2)], dtype=torch.int32)
        stage2_coordinates = torch.tensor([[0, x, 0, 0] for x in range(0, 32, 8)], dtype=torch.int32)
        stage3_coordinates = torch.tensor([[0, 0, 0, 0], [0, 16, 0, 0]], dtype=torch.int32)
        stage1 = ME.SparseTensor(torch.randn(len(stage1_coordinates), 32),
                                 coordinates=stage1_coordinates, tensor_stride=2)
        stage2 = ME.SparseTensor(torch.randn(len(stage2_coordinates), 128),
                                 coordinates=stage2_coordinates, tensor_stride=8)
        stage3 = ME.SparseTensor(torch.randn(len(stage3_coordinates), 384),
                                 coordinates=stage3_coordinates, tensor_stride=16)
        stages = (stage1, stage2, stage3)
        stage_features = tuple(stage.F for stage in stages)
        stage_valid = tuple(torch.ones(stage.F.shape[0], dtype=torch.bool) for stage in stages)
        lidar = torch.randn(len(stage3_coordinates), 512).flip(0)
        mma = EquationMMA()
        optimizer = torch.optim.SGD(mma.parameters(), lr=1e-3)
        output = mma(stages, stage_features, stage_valid, stage3.C, stage3.tensor_stride, lidar)
        self.assertTrue(torch.equal(output, lidar))
        output.square().sum().backward()
        self.assertGreater(float(mma.output_projection.weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = mma(stages, stage_features, stage_valid, stage3.C, stage3.tensor_stride, lidar)
        output.square().sum().backward()
        self.assertGreater(float(mma.eq4_f1[0].conv.kernel.grad.norm()), 0)

    def test_full_fusion_uses_three_native_strides_and_preserves_l0(self):
        torch.manual_seed(37)
        stage1_coordinates = torch.tensor([[0, x, 40, 8] for x in range(0, 32, 2)], dtype=torch.int32)
        stage2_coordinates = torch.tensor([[0, x, 40, 8] for x in range(0, 32, 8)], dtype=torch.int32)
        stage3_coordinates = torch.tensor([[0, 0, 32, 0], [0, 16, 32, 0]], dtype=torch.int32)
        stage1 = ME.SparseTensor(torch.randn(len(stage1_coordinates), 32),
                                 coordinates=stage1_coordinates, tensor_stride=2)
        stage2 = ME.SparseTensor(torch.randn(len(stage2_coordinates), 128),
                                 coordinates=stage2_coordinates, tensor_stride=8)
        stage3 = ME.SparseTensor(torch.randn(len(stage3_coordinates), 384),
                                 coordinates=stage3_coordinates, tensor_stride=16)
        stages = (stage1, stage2, stage3)
        lidar = torch.randn(len(stage3_coordinates), 512)
        image = torch.randn(1, 256, 64, 64)
        intrinsics = torch.tensor([[[1000.0, 0.0, 512.0],
                                    [0.0, 1000.0, 512.0],
                                    [0.0, 0.0, 1.0]]])
        camera_from_lidar = torch.tensor([[[0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]]])
        identity = torch.eye(4)[None]
        fusion = MaGiCPaperFusion()
        optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-4)
        arguments = (lidar, torch.zeros_like(lidar[:, :3]), stage3.C.flip(0),
                     torch.tensor([16.0, 16.0, 16.0]), image, intrinsics,
                     camera_from_lidar, identity, torch.tensor([[1024.0, 1024.0]]))
        output = fusion(*arguments, stages=stages, voxel_size=0.2, horizontal=1024)
        self.assertTrue(torch.equal(output, lidar))
        output.square().sum().backward()
        self.assertGreater(float(fusion.mma.output_projection.weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = fusion(*arguments, stages=stages, voxel_size=0.2, horizontal=1024)
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        self.assertGreater(float(fusion.attention[0].query.weight.grad.norm()), 0)
        self.assertGreater(float(fusion.mma.eq4_f1[0].conv.kernel.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
