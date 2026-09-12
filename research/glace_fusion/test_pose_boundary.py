import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .joint_solver import validate_pose
from .pose_boundary import solver_pose


class PoseBoundaryTests(unittest.TestCase):
    def test_roundoff_is_corrected_without_changing_translation(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([.1, .2, .3]).as_matrix() * (1 + 2e-6)
        pose[:3, 3] = [100, -200, 3]
        original = pose.copy()
        with self.assertRaises(ValueError):
            validate_pose(pose)
        result = solver_pose(pose)
        validate_pose(result)
        np.testing.assert_array_equal(result[:3, 3], pose[:3, 3])
        np.testing.assert_array_equal(original, pose)

    def test_invalid_transforms_remain_rejected(self):
        for scale in [-1., 1.1]:
            pose = np.eye(4)
            pose[0, 0] = scale
            with self.assertRaises(ValueError):
                solver_pose(pose)


if __name__ == '__main__':
    unittest.main()
