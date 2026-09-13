import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'vendor'), str(ROOT.parent)]

from rscore_l.fusion import camera_score, fuse
from rscore_l.geometry import surface_targets
from rscore_l.losses import geometry_loss


class Contracts(unittest.TestCase):
    def test_metric_surface_with_zero_world_components(self):
        K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        x, y = np.meshgrid(np.linspace(-.5, .5, 31), np.linspace(-.5, .5, 31))
        world = np.c_[x.ravel(), y.ravel(), np.full(x.size, 10)]
        result = surface_targets(world, np.array([[50., 50.]]), K, np.eye(4), (100, 100))
        self.assertTrue(result['geometry_valid'][0])
        np.testing.assert_allclose(result['xyz_target_world'][0], [0, 0, 10], atol=1e-5)

    def test_no_geometry_batch_has_finite_zero_gradient(self):
        prediction = torch.randn(8, 3, requires_grad=True)
        value = geometry_loss(prediction, {'geometry_valid': torch.zeros(8, dtype=torch.bool)})
        value.backward()
        self.assertEqual(float(value), 0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_geometry_known_origin_components_have_gradient(self):
        prediction = torch.tensor([[0., 0., 11.]], requires_grad=True)
        batch = dict(geometry_valid=torch.ones(1, dtype=torch.bool), gt_poses_inv=torch.eye(4)[None, :3],
            gt_coords=torch.tensor([[0., 0., 10.]]), intrinsics_inv=torch.eye(3)[None], target_px=torch.zeros(1, 2),
            sigma_parallel_m=torch.ones(1), sigma_perpendicular_m=torch.ones(1), geometry_quality=torch.ones(1))
        loss = geometry_loss(prediction, batch)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertGreater(float(prediction.grad[0, 2]), 0)

    def test_real_two_stage_model_geometry_backward(self):
        from rscore_l.train import make_config
        config = make_config(Path('/unused'), Path('/unused'), 'geometry')
        torch.manual_seed(2089)
        model = config.pipeline.model.setup(metadata={'cluster_centers': torch.zeros(50, 3)})
        outputs = model({'features': torch.randn(16, 384)})
        target = torch.randn(16, 3)
        target[:, 2] = 10
        batch = dict(geometry_valid=torch.ones(16, dtype=torch.bool), gt_poses_inv=torch.eye(4)[None, :3].repeat(16, 1, 1),
            gt_coords=target, intrinsics=torch.eye(3)[None].repeat(16, 1, 1), intrinsics_inv=torch.eye(3)[None].repeat(16, 1, 1),
            target_px=target[:, :2] / 10, sigma_parallel_m=torch.ones(16), sigma_perpendicular_m=torch.ones(16),
            geometry_quality=torch.ones(16))
        metrics = model.get_metrics_dict(outputs, batch)
        self.assertTrue(torch.isfinite(metrics['loss']))
        metrics['loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_visible_surface_rejects_missing_support(self):
        result = surface_targets(None, np.zeros((5, 2)), np.eye(3), np.eye(4), (100, 100))
        self.assertFalse(result['geometry_valid'].any())

    def test_query_dataset_never_reads_reference_poses(self):
        from unittest.mock import patch
        from rscore_l.dataset import NCLTDatasetConfig
        data = Path('/home/zhang/rscore-l-local/data')
        if not data.exists():
            self.skipTest('Local experiment data not prepared')
        original = np.load
        def checked(path, *args, **kwargs):
            if Path(path).name == 'poses.npy':
                raise AssertionError('Inference attempted to read reference poses')
            return original(path, *args, **kwargs)
        with patch('numpy.load', side_effect=checked):
            dataset = NCLTDatasetConfig(data=data, split='val').setup()
            item = dataset[0]
            self.assertTrue(item['mask'].any())
            self.assertFalse(item['mask'].all())

    def test_out_of_view_does_not_reduce_score(self):
        K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        uv = np.array([[50., 50.], [40., 40.]])
        good = np.array([[0., 0, 10], [-1., -1, 10]])
        bad = good.copy()
        bad[:, 2] = -10
        p = np.full(2, .8)
        self.assertGreater(camera_score(np.eye(4), uv, bad, p, K, np.eye(4), (100, 100)),
            camera_score(np.eye(4), uv, good, p, K, np.eye(4), (100, 100)))

    def test_all_zero_visual_reliability_returns_lidar(self):
        pool = dict(v1_two_stage=np.eye(4), T_corr=np.eye(4), c_local_all=np.eye(3), c_pred_all=np.eye(3),
            center_t=np.zeros(3), u_pred_all=np.zeros(3), candidate_T_WB=np.eye(4)[None])
        correspondence = dict(uv=np.ones((20, 2))*50, xyz=np.ones((2, 20, 3)), reliability=np.zeros((2, 20)),
            image_size_hw=np.array([100, 100]), K=np.eye(3))
        pose, diagnostics = fuse(pool, correspondence, np.eye(4))
        np.testing.assert_array_equal(pose, pool['v1_two_stage'])
        self.assertFalse(diagnostics['accepted'])

    def test_zero_visual_control_really_refines_lidar(self):
        rng = np.random.default_rng(2089)
        points = rng.normal(size=(30, 3))
        initial = np.eye(4)
        initial[0, 3] = .08
        pool = dict(v1_two_stage=initial, T_corr=np.eye(4), c_local_all=points, c_pred_all=points,
            center_t=np.zeros(3), u_pred_all=np.zeros(30), candidate_T_WB=initial[None])
        pose, diagnostics = fuse(pool, {}, np.eye(4), enable_visual=False)
        self.assertLess(np.linalg.norm(pose[:3, 3]), .04)
        self.assertEqual(diagnostics['mode'], 'lidar_only_refinement')

    def test_training_and_inference_mixture_scores_match(self):
        from rscore_l.reliability import differentiable_camera_score
        K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        x, y = np.meshgrid([10., 35., 60., 85.], [10., 35., 60., 85.])
        uv = np.c_[x.ravel(), y.ravel()]
        xyz = np.c_[(uv-50)/10, np.full(len(uv), 10)]
        xyz = np.stack([xyz, xyz + [.1, 0, 0]])
        p = np.full((2, len(uv)), .7)
        blocks = np.clip((uv / [100, 100] * 4).astype(int), 0, 3)
        fit = blocks.sum(1) % 2 == 0
        expected = min(camera_score(np.eye(4), uv, points, probability, K, np.eye(4), (100, 100), fit) for points, probability in zip(xyz, p))
        actual = differentiable_camera_score(torch.from_numpy(p), torch.from_numpy(xyz), torch.from_numpy(uv),
            torch.from_numpy(K), torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64), (100, 100))
        self.assertAlmostEqual(float(actual), expected, places=8)


if __name__ == '__main__':
    unittest.main()
