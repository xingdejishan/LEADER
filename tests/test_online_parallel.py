import ast
import io
import json
import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import e2e_online as E
from online_frontend import CachedSAM


class OnlineParallelTests(unittest.TestCase):
    def test_defaults(self):
        args = E.parser().parse_args([])
        self.assertEqual(args.keys_from, 'records')
        self.assertEqual(args.dates, '')
        self.assertEqual(args.threads, 1)
        self.assertEqual(args.sam_device, 'cuda')
        self.assertEqual(args.sam_source, 'cached')
        self.assertEqual(args.gate2, -1)
        self.assertEqual(args.k, 8.4)
        self.assertEqual(args.frontend_workers, 4)
        self.assertEqual(args.fast_pnp, 1)
        self.assertEqual(args.verify_cache, 1)

    def test_cached_verifier_preserves_exact_scores(self):
        from collections import OrderedDict
        from visual_branch import VisualBranch
        from visual_accelerated import AcceleratedBranch

        rng = np.random.RandomState(42)
        clouds = [rng.normal(size=(600, 3)) + [i, 0, 0] for i in range(3)]
        cfg = dict(map_radius_m=25, primary_nn_threshold_m=0.3,
                   nn_thresholds_m=[0.2, 0.3, 0.5], residual_window_m=1,
                   azimuth_bins=12, min_points_per_bin=3, bin_support_overlap=0.3,
                   min_supported_bins=2, residual_median_max_m=0.4,
                   residual_p90_max_m=0.9, max_top_bin_share=0.5)
        branches = [VisualBranch.__new__(VisualBranch), AcceleratedBranch.__new__(AcceleratedBranch)]
        for branch in branches:
            branch.verify_cfg = cfg
            branch.positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
            branch._map_cloud = lambda i: clouds[i]
            branch.C = SimpleNamespace(transform=lambda q, T: q @ T[:3, :3].T + T[:3, 3])
        branches[1]._verification_maps = OrderedDict()
        query = rng.normal(size=(100, 3))
        for exclusion in ([], [0], [1, 2], [0, 1, 2]):
            left, right = [branch.verify(np.eye(4), query, exclusion) for branch in branches]
            self.assertEqual(left, right)
        self.assertEqual(len(branches[1]._verification_maps), 1)

    @unittest.skipUnless(os.name == 'posix' and (ROOT / 'kernels/libgrid.so').is_file(), 'Linux kernel required')
    def test_bounded_kernel_matches_allowed_nearest_distance(self):
        from bounded_kernel import BoundedKernel
        from scipy.spatial import cKDTree

        rng = np.random.RandomState(12)
        cloud = rng.normal(size=(1000, 3))
        labels = np.arange(len(cloud), dtype=np.int32) % 3
        query = rng.normal(size=(120, 3))
        kernel = BoundedKernel(cloud, labels)
        try:
            for allowed in ([1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0]):
                mask = np.asarray(allowed, dtype=bool)
                points = cloud[mask[labels]]
                expected = cKDTree(points).query(query)[0] if len(points) else np.full(len(query), np.inf)
                actual = kernel.query_allowed(query, mask)
                np.testing.assert_allclose(np.minimum(expected, 1), np.minimum(actual, 1), atol=1e-12)
        finally:
            kernel.close()

    def test_cached_sam_integrity_and_no_recompute(self):
        import hashlib

        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            root = Path(directory)
            date = root / '2012-02-12'
            date.mkdir()
            image = root / 'image.jpg'
            image.write_bytes(b'image')
            array = np.zeros((64, 64, 256), np.float16)
            np.save(str(date / 'feature.npy'), array)
            record = dict(frame_key='2012-02-12/123', timestamp=123,
                          image_sha256=E.sha_file(image), checkpoint_sha256='trusted',
                          sam_model='vit_l', embedding_shape_hwc=[64, 64, 256],
                          embedding_dtype='float16', cache_file='feature.npy',
                          embedding_sha256=hashlib.sha256(array.tobytes()).hexdigest())
            manifest = dict(checkpoint_sha256='trusted', sam_model='vit_l', encoder_size=1024,
                            stored_layout='HWC', stored_dtype='float16',
                            stored_feature_size_hw=[64, 64], feature_channels=256,
                            feature_levels=[64, 32, 16], frames={record['frame_key']: record})
            (date / 'cache_manifest.json').write_text(json.dumps(manifest))
            lnm = SimpleNamespace(SAM_CHECKPOINT_SHA256='trusted',
                                  SAMFeatureStore=SimpleNamespace(_validate_metadata=Mock()),
                                  _decode_valid_mask=lambda r: np.ones((64, 64), bool))
            tensor = Mock()
            tensor.permute.return_value.contiguous.return_value.float.return_value = tensor
            torch = SimpleNamespace(from_numpy=Mock(return_value=tensor))
            spec = dict(key=record['frame_key'], image_path=str(image))
            with patch.dict(sys.modules, {'torch': torch}):
                cache = CachedSAM(root, lnm, {})
                embedding, _, meta = cache.encode(None, spec)
                self.assertIs(embedding, tensor)
                self.assertEqual(meta['frame_key'], record['frame_key'])
                self.assertEqual(torch.from_numpy.call_count, 2)
                image.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    cache.encode(None, spec)
                with self.assertRaisesRegex(ValueError, 'missing'):
                    cache.encode(None, dict(key='2012-02-12/999', image_path=str(image)))
                image.write_bytes(b'image')
                array[0, 0, 0] = 1
                np.save(str(date / 'feature.npy'), array)
                with self.assertRaisesRegex(ValueError, 'embedding SHA'):
                    cache.encode(None, spec)

    def test_invalid_poses(self):
        for pose in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                E.valid_pose(pose)

    def test_scan_decode(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            path = Path(directory) / 'scan.bin'
            dtype = [('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')]
            raw = np.array([(20000, 20400, 19800, 17, 0)], dtype=dtype)
            raw.tofile(str(path))
            xyz, xyz32, intensity = E.read_scan(path)
            np.testing.assert_allclose(xyz, [[0, 2, -1]])
            np.testing.assert_array_equal(xyz32, xyz.astype(np.float32))
            self.assertEqual(intensity.tolist(), [17])
            path.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                E.read_scan(path)

    def test_npy_and_empty_cloud(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            path = Path(directory) / 'scan.npy'
            np.save(str(path), np.array([[1., 2., 3., 4.]]))
            self.assertEqual(E.read_scan(path)[2].tolist(), [4.])
            np.save(str(path), np.ones((3, 3)))
            with self.assertRaises(ValueError):
                E.read_scan(path)
        cfg = SimpleNamespace(min_scan_range=1, max_scan_range=60)
        with self.assertRaises(ValueError):
            E.prepare_scan(np.zeros((1, 3)), cfg, None)

    def test_same_evidence_and_calibrated_switch(self):
        scan = np.random.RandomState(4).normal(size=(600, 3))
        geo, vis = np.eye(4), np.eye(4)
        geo[0, 3] = 3
        maps = SimpleNamespace(nearby=Mock(return_value=(0, 1, 2)), local=Mock(return_value=scan))
        final, diag = E.shared_arbitration(dict(pose=geo), dict(pose=vis), scan, maps,
                                           SimpleNamespace(reference_count=3), 8.4)
        np.testing.assert_array_equal(final, vis)
        self.assertEqual(diag['chosen'], 'visual')
        self.assertTrue(diag['switched'])
        self.assertAlmostEqual(diag['thresh'], 8.4 * diag['sigma_c'])
        self.assertEqual(diag['n_map'], 1200)

    def test_ties_preserve_geometry_and_missing_visual(self):
        scan = np.random.RandomState(4).normal(size=(600, 3))
        pose = np.eye(4)
        maps = SimpleNamespace(nearby=Mock(return_value=(0,)), local=Mock(return_value=scan))
        for visual in (None, dict(pose=pose)):
            final, diag = E.shared_arbitration(dict(pose=pose), visual, scan, maps,
                                               SimpleNamespace(reference_count=3), 8.4)
            np.testing.assert_array_equal(final, pose)
            self.assertEqual(diag['chosen'], 'route3')

    def test_arbitration_reuses_geometry_map(self):
        from scipy.spatial import cKDTree

        scan = np.random.RandomState(4).normal(size=(600, 3))
        tree = cKDTree(scan)
        maps = SimpleNamespace(nearby=Mock(return_value=(0,)), local=Mock(return_value=scan))
        _, diag = E.shared_arbitration(dict(pose=np.eye(4), tree=tree), dict(pose=np.eye(4)),
                                      scan, maps, SimpleNamespace(reference_count=3), 8.4)
        maps.local.assert_called_once()
        self.assertEqual(diag['n_map'], 1200)

    def test_unsupported_visual_has_explicit_reason(self):
        maps = SimpleNamespace(nearby=lambda centre: ())
        _, diag = E.shared_arbitration(dict(pose=np.eye(4)), dict(pose=np.eye(4)),
                                      np.zeros((600, 3)), maps,
                                      SimpleNamespace(reference_count=3), 8.4)
        self.assertEqual(diag['note'], 'no_reference_near_visual')

    def test_parallel_visual_called_even_if_geometry_skips(self):
        localizer = E.ParallelLocalizer.__new__(E.ParallelLocalizer)
        pose = np.eye(4)
        localizer.args = SimpleNamespace(visual_budget=8, gate2=-1, k=8.4, sam_source='cached')
        localizer.cfg = SimpleNamespace()
        localizer.F, localizer.P = None, None
        localizer.maps = None
        localizer.frontend = SimpleNamespace(predict=Mock(return_value=(pose, {}, {})),
                                            torch=SimpleNamespace(no_grad=lambda: __import__('contextlib').nullcontext()),
                                            runtime=SimpleNamespace(denied=[]))
        localizer.visual = SimpleNamespace(hypothesize=Mock(return_value=None))
        geometry = dict(pose=pose, skipped='no_reference', ran_fine=False, map_points=0, gain=None)
        spec = dict(key='2012-02-12/123', scan_path='scan.bin', image_path='image.jpg', T_BC=pose)
        points = np.zeros((600, 3))
        with patch.object(E, 'read_scan', return_value=(points, points, np.zeros(600))), \
                patch.object(E, 'prepare_scan', return_value=points), \
                patch.object(E, 'geometric_branch', return_value=geometry):
            result = localizer.predict(spec)
        localizer.visual.hypothesize.assert_called_once()
        self.assertEqual(result['T'], pose.tolist())
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['upstream'], 'live')

    def test_frontend_batch_calls_existing_parallel_path(self):
        localizer = E.ParallelLocalizer.__new__(E.ParallelLocalizer)
        pose = np.eye(4)
        localizer.frontend = SimpleNamespace(predict_many=Mock(return_value=[(pose, {}, {})] * 4))
        localizer.predict = Mock(side_effect=[dict(key=str(i)) for i in range(4)])
        prepared = [(dict(key=str(i)), None, None, 'loaded', 0) for i in range(4)]
        rows = localizer.predict_many(prepared)
        localizer.frontend.predict_many.assert_called_once()
        self.assertEqual(localizer.predict.call_count, 4)
        self.assertEqual([r['batch_size'] for r in rows], [4] * 4)
        self.assertTrue(all(r['batch_finished_at'] >= r['batch_started_at'] for r in rows))

    def test_resume_repairs_torn_tail_and_ignores_errors(self):
        row = dict(key='a', status='success', T=np.eye(4).tolist(), input_signature='abc')
        stream = io.StringIO(json.dumps(row) + '\n' + json.dumps(dict(key='b', status='error', T=None)) + '\n{"key":')
        done = E.read_done(stream)
        self.assertEqual(done, {'a': 'abc'})
        self.assertTrue(stream.getvalue().endswith('\n'))

    def test_no_frozen_query_dependencies(self):
        for name in ('e2e_online.py', 'online_frontend.py'):
            source = (ROOT / name).read_text()
            ast.parse(source)
            self.assertNotIn('initial.npz', source)
            self.assertNotIn('VERIFY', source)
            self.assertNotIn('InputCache', source)

    def test_output_path_rejects_escape(self):
        for path in ('../other.jsonl', '/tmp/other.jsonl', 'results/../../other.jsonl'):
            with self.assertRaises(ValueError):
                E.output_path(path)

    def test_full_manifest_selection_and_canonical_indices(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            data = Path(directory)
            for date in E.DEV_DATES:
                parent = data / 'NCLT' / date
                parent.mkdir(parents=True)
                frames = [dict(timestamp_us=i, camera_id=5,
                               scan=dict(timestamp_us=i, path='velodyne_sync/%s.bin' % i, sha256='s'),
                               image=dict(path='cam5/%s.jpg' % i, sha256='i')) for i in (1, 2)]
                (parent / 'manifest.json').write_text(json.dumps(dict(
                    date=date, split='test', complete=True, frames=frames,
                    roi=dict(T_BC=np.eye(4).tolist()))))
            args = E.parser().parse_args(['--dates', E.DEV_DATES[-1], '--limit', '1'])
            args.shard_index, args.shard_count = 0, 1
            with patch.object(E, 'DATA', data):
                selected = E.select_inputs(args)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['canonical_index'], 6)
            self.assertEqual(selected[0]['key'], E.DEV_DATES[-1] + '/1')
            args.dates, args.limit = '', 0
            args.shard_index, args.shard_count = 1, 2
            with patch.object(E, 'DATA', data):
                selected = E.select_inputs(args)
            self.assertEqual([r['canonical_index'] for r in selected], [1, 3, 5, 7])

    def test_unknown_key_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            data = Path(directory)
            for date in E.DEV_DATES:
                parent = data / 'NCLT' / date
                parent.mkdir(parents=True)
                (parent / 'manifest.json').write_text(json.dumps(dict(
                    date=date, split='test', complete=True, frames=[], roi=dict(T_BC=np.eye(4).tolist()))))
            args = E.parser().parse_args(['--key', '2012-02-12/123'])
            args.shard_index, args.shard_count = 0, 1
            with patch.object(E, 'DATA', data), self.assertRaisesRegex(ValueError, 'unknown'):
                E.select_inputs(args)


if __name__ == '__main__':
    unittest.main()
