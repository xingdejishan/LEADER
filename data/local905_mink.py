import hashlib
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np

from utils.pose_util import cartesian_to_polar_expansion


RAW_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
                      ('intensity', 'u1'), ('ring', 'u1')])


class Local905_mink:
    def __init__(self, data_path, split_path, subset, voxel_size=0.2,
                 horizontal_res=1024, max_points=4096, min_range=1.0, max_range=100.0):
        self.root = Path(data_path).resolve()
        self.voxel_size = voxel_size
        self.horizontal_res = horizontal_res
        self.max_points = max_points
        self.min_range = min_range
        self.max_range = max_range
        split = json.loads(Path(split_path).read_text(encoding='utf-8'))
        if split['protocol'] not in ('local905_date_holdout_v1',
                                     'local905_date_holdout_masked_v2'):
            raise ValueError('Unknown local905 split protocol')
        self.valid_mask_sha256 = split.get('valid_mask_sha256')
        self.keys = split['splits'][subset]
        self.pcs = [str(self.root / key) for key in self.keys]
        if not all(Path(path).is_file() for path in self.pcs):
            raise FileNotFoundError('Missing local905 raw scan')
        scene = self.root / 'train_scene'
        metadata = json.loads((scene / 'scene_meta.json').read_text(encoding='utf-8'))
        self.body_from_camera = np.asarray(metadata['T_BC_camera_to_body'], dtype=np.float64)
        self.camera_from_body = np.linalg.inv(self.body_from_camera)
        self.transforms = []
        for key in self.keys:
            stem = Path(key).stem
            camera_pose = np.loadtxt(scene / 'train' / 'poses' / (stem + '.txt'))
            self.transforms.append((camera_pose @ self.camera_from_body).astype(np.float32))
        translations = np.stack([pose[:3, 3] for pose in self.transforms])
        self.center_t = np.concatenate((translations[:, :2].mean(axis=0),
                                        [translations[:, 2].min()]))

    def __len__(self):
        return len(self.pcs)

    def get_center_t(self):
        return self.center_t.copy()

    def __getitem__(self, index):
        raw = np.fromfile(self.pcs[index], dtype=RAW_DTYPE)
        scan = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32) * 0.005 - 100
        label = raw['intensity'].astype(np.float32)
        ranges = np.linalg.norm(scan, axis=1)
        keep = (ranges > self.min_range) & (ranges < self.max_range)
        scan = scan[keep]
        label = label[keep]
        if self.max_points > 0 and len(scan) > self.max_points:
            seed = int.from_bytes(hashlib.sha256(self.keys[index].encode()).digest()[:8], 'little')
            chosen = np.random.default_rng(seed).choice(len(scan), self.max_points, replace=False)
            chosen.sort()
            scan = scan[chosen]
            label = label[chosen]
        polar = cartesian_to_polar_expansion(scan, self.voxel_size * self.horizontal_res)
        features = np.column_stack((polar[:, 2], polar[:, 1], label)).astype(np.float32)
        coordinates, features = ME.utils.sparse_quantize(
            coordinates=polar, features=features, quantization_size=self.voxel_size)
        return coordinates, features, scan, self.transforms[index], np.eye(4, dtype=np.float32)
