import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from light_selection import LightSelector, ProposalBranch, SparseSelector, BatchedEvidence, shortlist
import arbitration as ARB


def candidate(pose=None, sources=None, inliers=30, qualified=True):
    return dict(pose=(np.eye(4) if pose is None else pose).tolist(),
                reference_ids=[0] if sources is None else sources,
                unique_query_inliers=inliers, coverage_cells=8,
                median_reprojection_px=1.0, consensus_reference_count=2,
                qualified=qualified)


class LightSelectionTests(unittest.TestCase):
    def test_sparse_retains_unqualified_rescue_candidates(self):
        selector = SparseSelector.__new__(SparseSelector)
        poses = []
        for i in range(10):
            pose = np.eye(4)
            pose[0, 3] = i
            poses.append(candidate(pose, qualified=False))
        result = selector.candidates({'hypotheses': poses})
        self.assertEqual(len(result), 8)
        self.assertEqual(result[-1]['pose'][0][3], 7)

    def test_batch_scores_match_scalar_including_calibration(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(1500, 3))
        query = rng.normal(size=(100, 3))
        hypotheses = [ARB.Hypothesis(np.eye(4), 'route3', {'incumbent': True})]
        shifted = np.eye(4)
        shifted[0, 3] = 0.2
        hypotheses.append(ARB.Hypothesis(shifted, 'visual'))
        scalar = ARB.Evidence(points, query, n_points=100)
        batch = BatchedEvidence(points, query, hypotheses, 100)
        for pose in [h.pose for h in hypotheses] + ARB.perturbations(np.eye(4)):
            self.assertAlmostEqual(scalar.score(pose), batch.score(pose), places=13)
        self.assertEqual(ARB.decide(hypotheses, scalar, 8.4)[0].source,
                         ARB.decide(hypotheses, batch, 8.4)[0].source)

    def selector(self):
        refs = [dict(key='r%d' % i, scan_path='unused', world_from_body=np.eye(4).tolist())
                for i in range(3)]
        visual = [dict(index=i, key='r%d' % i) for i in range(3)]
        return LightSelector(refs, visual)

    def test_shortlist_filters_deduplicates_and_limits(self):
        shifted = np.eye(4)
        shifted[0, 3] = 1
        result = shortlist([candidate(inliers=10), candidate(inliers=50, qualified=False),
                            candidate(inliers=40), candidate(shifted, inliers=20)])
        self.assertEqual([r['unique_query_inliers'] for r in result], [40, 20])

    def test_proposal_never_calls_laser_verifier(self):
        branch = ProposalBranch.__new__(ProposalBranch)
        branch.retrieve = Mock(return_value=([0], {}))
        branch._match_frame = Mock(return_value=(np.zeros((1, 2)), [(0, 1, 0, 0)], {}, True))
        branch._hypotheses = Mock(return_value=([candidate()], [], 1, 1))
        branch.verify = Mock(side_effect=AssertionError('heavy verifier called'))
        proposal = branch.propose('d/1', 'image')
        branch.verify.assert_not_called()
        self.assertEqual(len(proposal['top2']), 1)
        self.assertEqual(proposal['detail']['seconds']['verify'], 0)

    def test_proposal_refuses_a_pose_prior(self):
        branch = ProposalBranch.__new__(ProposalBranch)
        branch.retrieve = Mock(return_value=([0], {}))
        with self.assertRaises(TypeError):
            branch.propose('d/1', 'image', np.eye(4))

    def test_shared_map_excludes_candidate_sources_for_all_poses(self):
        selector = self.selector()
        scan = np.random.RandomState(4).normal(size=(600, 3))
        selector.cloud = Mock(return_value=scan)
        geometry = np.eye(4)
        geometry[0, 3] = 3
        pose, diag, visual = selector.select(dict(pose=geometry),
            dict(top2=[candidate()], n_candidates=1), scan)
        np.testing.assert_array_equal(pose, np.eye(4))
        self.assertEqual(diag['chosen'], 'visual')
        self.assertEqual(diag['excluded_references'], [0])
        self.assertEqual(diag['evidence_references'], [1, 2])
        self.assertEqual(diag['candidates_scored'], 1)
        self.assertIsNone(visual['verified'])
        self.assertAlmostEqual(diag['thresh'], 8.4 * diag['sigma_c'])

    def test_no_independent_reference_preserves_geometry(self):
        selector = self.selector()
        selector.cloud = Mock(side_effect=AssertionError('should not read map'))
        pose, diag, _ = selector.select(dict(pose=np.eye(4)),
            dict(top2=[candidate(sources=[0, 1, 2])], n_candidates=1), np.zeros((600, 3)))
        np.testing.assert_array_equal(pose, np.eye(4))
        self.assertEqual(diag['note'], 'no_independent_geometry_reference')

    def test_unknown_source_is_not_silently_admitted(self):
        selector = self.selector()
        _, diag, _ = selector.select(dict(pose=np.eye(4)),
            dict(top2=[candidate(sources=[999])], n_candidates=1), np.zeros((600, 3)))
        self.assertIn('unknown_candidate_source', diag['dropped'])
        self.assertFalse(diag['decided'])

    def test_reference_sampling_is_bounded_and_cached(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            path = Path(directory) / 'ref.bin'
            selector = self.selector()
            raw = np.zeros(20, dtype=selector.dtype)
            raw['x'] = np.arange(20000, 20020)
            raw['y'], raw['z'] = 20000, 20000
            raw.tofile(str(path))
            selector.refs[0]['scan_path'] = str(path)
            selector.map_points_per_ref = 6
            cloud = selector.cloud(0)
            self.assertEqual(cloud.shape, (6, 3))
            self.assertIs(selector.cloud(0), cloud)
            self.assertAlmostEqual(cloud[-1, 0], 0.095)


if __name__ == '__main__':
    unittest.main()
