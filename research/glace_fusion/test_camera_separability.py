import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_separability import camera_evidence, wrong_pose


class CameraEvidenceTests(unittest.TestCase):
    def test_clipped_score_and_invalid_depth_denominator(self):
        xyz = np.array([[0,0,1],[.05,0,1],[.2,0,1],[0,0,-1.]])
        K = np.diag([100.,100.,1.])
        result = camera_evidence(xyz,np.zeros((4,2)),K,np.eye(4))
        self.assertAlmostEqual(result['S_C'],.5625)
        self.assertAlmostEqual(result['q_C'],.5)

    def test_camera_to_world_direction(self):
        pose = np.eye(4)
        pose[:3,3] = [3,4,5]
        result = camera_evidence(np.array([[3.,4.,7.]]),np.zeros((1,2)),np.eye(3),pose)
        self.assertEqual(result['S_C'],0.)
        self.assertEqual(result['q_C'],1.)

    def test_wrong_pose_has_exact_requested_perturbation(self):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_rotvec([.3,.2,.1]).as_matrix()
        pose[:3,3] = [100.,200.,300.]
        wrong = wrong_pose(pose)
        self.assertAlmostEqual(np.linalg.norm(wrong[:3,3]-pose[:3,3]),2.)
        self.assertAlmostEqual(Rotation.from_matrix(wrong[:3,:3]@pose[:3,:3].T).magnitude()*180/np.pi,5.)


if __name__ == '__main__':
    unittest.main()
