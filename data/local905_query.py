import hashlib
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from PIL import Image

from utils.pose_util import cartesian_to_polar_expansion


RAW_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
                      ('intensity', 'u1'), ('ring', 'u1')])


def load_sparse_scan(path, key, max_points, voxel_size=0.2, horizontal_res=1024):
    raw = np.fromfile(path, dtype=RAW_DTYPE)
    scan = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32) * 0.005 - 100
    label = raw['intensity'].astype(np.float32)
    ranges = np.linalg.norm(scan, axis=1)
    keep = (ranges > 1.0) & (ranges < 100.0)
    scan = scan[keep]
    label = label[keep]
    if max_points > 0 and len(scan) > max_points:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
        chosen = np.random.default_rng(seed).choice(len(scan), max_points, replace=False)
        chosen.sort()
        scan = scan[chosen]
        label = label[chosen]
    polar = cartesian_to_polar_expansion(scan, voxel_size * horizontal_res)
    features = np.column_stack((polar[:, 2], polar[:, 1], label)).astype(np.float32)
    coordinates, features = ME.utils.sparse_quantize(
        coordinates=polar, features=features, quantization_size=voxel_size)
    return coordinates, features


class Local905Query:
    def __init__(self, data_root, split_path, sam_manifest=None, max_points=4096):
        self.root = Path(data_root).resolve()
        self.keys = json.loads(Path(split_path).read_text(encoding='utf-8'))['splits']['test']
        self.max_points = max_points
        self.manifest_path = Path(sam_manifest).resolve() if sam_manifest else None
        self.manifest = (json.loads(self.manifest_path.read_text(encoding='utf-8'))
                         if self.manifest_path else None)
        if self.manifest is not None and set(self.keys) - set(self.manifest['frames']):
            raise ValueError('SAM manifest does not cover all test scans')

    def load(self, key, feature_key=None):
        if key not in self.keys:
            raise ValueError(f'Unknown test scan: {key}')
        coordinates, features = load_sparse_scan(self.root / key, key, self.max_points)
        result = {'coords': coordinates, 'feats': features}
        if self.manifest is None:
            return result
        record = self.manifest['frames'][key]
        feature_record = self.manifest['frames'][feature_key or key]
        feature_path = self.manifest_path.parent / feature_record['sam_features']
        embedding = np.load(feature_path, allow_pickle=False)
        if embedding.shape != (256, 64, 64) or not np.isfinite(embedding).all():
            raise ValueError(f'Invalid SAM feature: {feature_path}')
        image_path = self.manifest_path.parent / record['image']
        with Image.open(image_path) as image:
            width, height = image.size
        resized_width, resized_height = record['resized_size']
        if [int(width * 1024 / max(width, height) + 0.5),
                int(height * 1024 / max(width, height) + 0.5)] != record['resized_size']:
            raise ValueError(f'Invalid SAM resized size: {key}')
        intrinsic = np.asarray(record['K'], dtype=np.float32).copy()
        intrinsic[0] *= resized_width / width
        intrinsic[1] *= resized_height / height
        result.update({
            'sam_features': torch.from_numpy(embedding.astype(np.float32))[None],
            'intrinsics': torch.from_numpy(intrinsic)[None],
            'camera_from_lidar': torch.tensor(record['T_camera_lidar'], dtype=torch.float32)[None],
            'image_bounds': torch.tensor([[resized_width, resized_height]], dtype=torch.float32),
        })
        return result
