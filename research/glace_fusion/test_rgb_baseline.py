import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from .glace_adapter import GLACEAdapter
from .nclt_camera import preprocess_image
from .rgb_features import CachedRGBFeatures
from .inference_contract import InferenceSession, resolve_contract


class TestRGBBaseline(unittest.TestCase):
    def test_local_resize_uses_vendor_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            Image.new('RGB', (808, 616), (30, 90, 170)).save(path)
            image, K = preprocess_image(path, np.eye(3), 480)
            self.assertEqual(image.shape, (480, 630))
            self.assertAlmostEqual(K[0, 0], 480 / 616)

    def test_rgb_head_rejects_legacy_grayscale_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps({'global_feature_protocol': 'official_rgb_r2former_480x640', 'local_image_resolution': 480}))
            with self.assertRaisesRegex(ValueError, 'RGB heads require'):
                GLACEAdapter(root, root / 'head.pt', global_feature_fn=lambda image: np.zeros(256))

    def test_feature_cache_rejects_reordering_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'rgb').mkdir()
            for name in ['a.png', 'b.png']:
                Image.new('RGB', (4, 4)).save(root / 'rgb' / name)
            np.save(root / 'features.npy', np.ones((2, 256), np.float32))
            manifest = dict(protocol='official_rgb_r2former_480x640', images=['a.png', 'b.png'],
                            sha256=hashlib.sha256((root / 'features.npy').read_bytes()).hexdigest())
            (root / 'features_manifest.json').write_text(json.dumps(manifest))
            np.testing.assert_array_equal(CachedRGBFeatures(root)['a'], np.ones(256))
            manifest['images'].reverse()
            (root / 'features_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'order'):
                CachedRGBFeatures(root)
            manifest['images'].reverse()
            (root / 'features_manifest.json').write_text(json.dumps(manifest))
            np.save(root / 'features.npy', np.zeros((2, 256), np.float32))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                CachedRGBFeatures(root)

    def test_session_cache_and_online_share_explicit_contract(self):
        class Adapter:
            def __init__(self, *args, **kwargs):
                self.coordinate_precision = 'fp32_head'
                self.head_sha256 = 'fake'
                self.pnp_threshold = kwargs['pnp_threshold']
                self.hypotheses = kwargs['hypotheses']
                if kwargs['global_feature_fn'] is not None:
                    raise AssertionError('RGB must not use grayscale callbacks')

            def infer(self, gray, K, *, global_feature):
                return gray, K, global_feature

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'rgb').mkdir()
            image = root / 'rgb/a.png'
            Image.new('RGB', (808, 616), (200, 50, 0)).save(image)
            (root / 'head.pt').write_bytes(b'head')
            checkpoint = root / 'backbone'
            checkpoint.write_bytes(b'backbone')
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (root / 'config.json').write_text(json.dumps(dict(global_feature_protocol='official_rgb_r2former_480x640',
                local_image_resolution=480, global_backbone_sha256=sha)))
            features = np.ones((1, 256), np.float32) / 16
            np.save(root / 'features.npy', features)
            (root / 'features_manifest.json').write_text(json.dumps(dict(protocol='official_rgb_r2former_480x640',
                images=['a.png'], sha256=hashlib.sha256((root / 'features.npy').read_bytes()).hexdigest(), checkpoint_sha256=sha)))
            def extractor(*args):
                def extract(paths):
                    self.assertEqual(paths, [image])
                    return features
                return extract
            with patch('research.glace_fusion.inference_contract.GLACEAdapter', Adapter), patch(
                    'research.glace_fusion.inference_contract.rgb_feature_extractor', extractor):
                cached = InferenceSession(root, root / 'head.pt', checkpoint, split=root)
                online = InferenceSession(root, root / 'head.pt', checkpoint)
                for a, b in zip(cached.infer(image, np.eye(3)), online.infer(image, np.eye(3))):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(cached.infer(image, np.eye(3))[0].shape, (480, 630))
            with self.assertRaisesRegex(ValueError, 'expects image resolution'):
                resolve_contract(root / 'head.pt', 616)


if __name__ == '__main__':
    unittest.main()
