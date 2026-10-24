import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arbitration as ARB
from gpu_selection import CudaKernel, GpuEvidence


@unittest.skipUnless(os.environ.get('RUN_GPU_SELECTION_TESTS') == '1', 'explicit idle GPU required')
class GpuSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(0.15)
        cls.torch, cls.kernel = torch, CudaKernel()

    def compare(self, points, query, poses):
        hypotheses = [ARB.Hypothesis(p, str(i), {'incumbent': i == 0}) for i, p in enumerate(poses)]
        cpu = ARB.Evidence(points, query, n_points=len(query))
        gpu = GpuEvidence(points, query, hypotheses, self.kernel, self.torch)
        for pose in poses + ARB.perturbations(poses[0]):
            self.assertAlmostEqual(cpu.score(pose), gpu.score(pose), places=12)
        a, da = ARB.decide(hypotheses, cpu, 8.4)
        b, db = ARB.decide(hypotheses, gpu, 8.4)
        self.assertEqual(a.source, b.source)
        self.assertAlmostEqual(da['thresh'], db['thresh'], places=12)

    def test_random_negative_coordinates_and_far_disconnected_crops(self):
        rng = np.random.default_rng(25)
        points = np.concatenate([rng.normal(size=(3000, 3))-5, rng.normal(size=(3000, 3))+400])
        query = rng.normal(size=(500, 3))
        first, second = np.eye(4), np.eye(4)
        first[:3, 3], second[:3, 3] = -5, 400
        self.compare(points, query, [first, second])

    def test_cell_boundaries_and_truncation(self):
        points = np.array([[-1., 0, 0], [0, 0, 0], [1, 1, 1], [3, 0, 0]])
        query = np.array([[0, 0, 0], [-1e-12, 0, 0], [1+1e-12, 1, 1], [20, 20, 20]])
        self.compare(points, query, [np.eye(4)])

    def test_rotated_pose_and_negative_world(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(71)
        points = rng.uniform(-20, 20, size=(12000, 3))
        query = rng.uniform(-4, 4, size=(1000, 3))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0.4, -0.7, 1.2]).as_matrix()
        pose[:3, 3] = [-10, -5, -3]
        self.compare(points, query, [pose, np.eye(4)])

    def test_dense_cell_rejected(self):
        with self.assertRaisesRegex(ValueError, 'occupancy'):
            GpuEvidence(np.zeros((4097, 3)), np.zeros((1, 3)),
                        [ARB.Hypothesis(np.eye(4), 'inc')], self.kernel, self.torch)


if __name__ == '__main__':
    unittest.main()
