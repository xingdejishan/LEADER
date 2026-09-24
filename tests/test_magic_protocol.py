import unittest

import numpy as np

from tools.eval_local905_online import shuffled_feature_keys
from tools.eval_magic_correspondence_swaps import alignment


class MagicProtocolTests(unittest.TestCase):
    def test_shuffle_preserves_camera_and_excludes_nearby_frames(self):
        keys = []
        records = {}
        for camera in (0, 1):
            extrinsic = np.eye(4)
            extrinsic[0, 3] = camera
            for index in range(20):
                stamp = 1_000_000_000 + camera * 100_000_000 + index * 1_000_000
                key = f'scans/date/velodyne_sync/{stamp}.bin'
                keys.append(key)
                records[key] = {'T_camera_lidar': extrinsic.tolist()}
        mapping = shuffled_feature_keys(keys, {'frames': records}, 37)
        self.assertEqual(set(mapping), set(keys))
        for key, other in mapping.items():
            self.assertEqual(records[key]['T_camera_lidar'], records[other]['T_camera_lidar'])
            self.assertGreaterEqual(abs(int(key.split('/')[-1][:-4]) -
                                        int(other.split('/')[-1][:-4])), 3_000_000)

    def test_swap_alignment_uses_voxel_identity(self):
        source = np.asarray([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=np.int32)
        points = np.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
        order = np.asarray([2, 0, 1])
        original = {'voxel_coordinates': source, 'input_local_xyz': points}
        changed = {'voxel_coordinates': source[order], 'input_local_xyz': points[order]}
        self.assertEqual(alignment(original, changed).tolist(), [1, 2, 0])
        changed['input_local_xyz'][0, 0] += 0.1
        with self.assertRaises(ValueError):
            alignment(original, changed)


if __name__ == '__main__':
    unittest.main()
