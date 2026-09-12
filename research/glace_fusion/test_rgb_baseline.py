import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .glace_adapter import GLACEAdapter
from .nclt_camera import preprocess_image
from .rgb_features import CachedRGBFeatures


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
            (root / 'config.json').write_text(json.dumps({'global_feature_protocol': 'official_rgb_r2former_480x640'}))
            with self.assertRaisesRegex(ValueError, 'requires RGB'):
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


if __name__ == '__main__':
    unittest.main()
