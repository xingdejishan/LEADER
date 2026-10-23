import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms import MoE
import e2e_online


class NoRotationTests(unittest.TestCase):
    def setUp(self):
        self.scan = np.random.RandomState(42).normal(size=(600, 3))
        self.initial = np.eye(4)
        self.initial[:3, :3] = [[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]]
        self.tree = cKDTree(self.scan)
        self.coarse = dict(tree=self.tree, base_c=self.scan,
                           surface=np.zeros((17, 3)))
        self.maps = SimpleNamespace(
            nearby=Mock(return_value=(0, 1, 2)),
            local=Mock(return_value=self.scan))

    def model(self, gate=-1):
        model = MoE([], np.zeros((1, 3)), dict(workers=1, gate2=gate))
        model.maps = self.maps
        model.cfs.coarse = Mock(return_value=self.coarse)
        model.cfs.solve = Mock(return_value=dict(s=1.0, d=0.2, expanded=1))
        model.ambiguity = Mock(return_value=(0.0, 0.5, 0.5))
        return model

    def test_search_preserves_v7_rotation(self):
        model = self.model()
        before = self.initial.copy()
        pose, diag = model.localize(self.scan, self.initial)
        np.testing.assert_array_equal(pose[:3, :3], before[:3, :3])
        np.testing.assert_array_equal(self.initial, before)
        self.assertGreater(np.linalg.norm(pose[:3, 3] - before[:3, 3]), 0)
        self.assertTrue(diag['ran_fine'])
        self.assertFalse(diag['rotated'])

    def test_closed_gate_preserves_full_pose(self):
        model = self.model(gate=2)
        pose, diag = model.localize(self.scan, self.initial)
        np.testing.assert_array_equal(pose, self.initial)
        model.cfs.solve.assert_not_called()
        self.assertFalse(diag['rotated'])

    def test_visual_arbitration_still_accepts_full_pose(self):
        model = self.model()
        visual = np.eye(4)
        visual[:3, 3] = [3, 4, 5]
        model.cost_at = Mock(side_effect=[0.8, 0.2])
        pose, diag = model.localize(self.scan, self.initial, visual_pose=visual)
        np.testing.assert_array_equal(pose, visual)
        self.assertEqual(diag['chosen'], 'visual')
        self.assertFalse(diag['rotated'])

    def test_removed_module_cannot_be_enabled(self):
        with self.assertRaisesRegex(ValueError, 'removed'):
            MoE([], np.zeros((1, 3)), dict(rot=True))

    def test_online_search_and_closed_gate(self):
        cfg = SimpleNamespace(reference_count=3, coarse_points=1000, topk=6)
        target = self.initial.copy()
        target[:3, 3] = [1, 2, 0]
        F = SimpleNamespace(coarse_stage=Mock(return_value=self.coarse))
        P = SimpleNamespace(
            along_track_direction=lambda pose: pose[:3, 0],
            localize=Mock(return_value=(target, dict(s=1, d=2, expanded=1, evals=5))))
        for gate in (-1, 2):
            P.localize.reset_mock()
            out = e2e_online.geometric_branch(
                self.initial, self.scan, {}, self.maps, cfg, gate, F, P)
            np.testing.assert_array_equal(out['pose'][:3, :3], self.initial[:3, :3])
            self.assertFalse(out['rotated'])
            self.assertEqual(out['ran_fine'], gate == -1)
            self.assertEqual(P.localize.call_count, int(gate == -1))

    def test_online_missing_references_preserves_pose(self):
        self.maps.nearby.return_value = ()
        out = e2e_online.geometric_branch(
            self.initial, self.scan, {}, self.maps,
            SimpleNamespace(reference_count=3), -1, None, None)
        np.testing.assert_array_equal(out['pose'], self.initial)
        self.assertEqual(out['skipped'], 'no_reference')

    def test_entrypoints_have_no_rotation_module(self):
        for name in ('algorithms/moe.py', 'three_route_unified.py',
                     'run_pipeline.py', 'e2e_online.py'):
            source = (ROOT / name).read_text()
            tree = ast.parse(source)
            functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            self.assertNotIn('rot_only', functions)
            self.assertNotIn('refine_rotation', functions)
            self.assertNotIn("add_argument('--rot'", source)


if __name__ == '__main__':
    unittest.main()
