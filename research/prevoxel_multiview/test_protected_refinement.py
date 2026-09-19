import unittest

import numpy as np

from local_visual_refinement_roma import apply_local_delta, holdout_mask, reference_pair_ids


class LocalPoseUpdateTest(unittest.TestCase):
    def test_translation_correction_is_world_origin_invariant(self):
        initial = np.eye(4)
        initial[:3, 3] = [200., 100., -30.]
        shifted = initial.copy()
        world_shift = np.array([700., -250., 40.])
        shifted[:3, 3] += world_shift
        delta = np.array([.08, -.03, .01, .0, .0, np.deg2rad(.8)])
        updated = apply_local_delta(initial, delta)
        shifted_updated = apply_local_delta(shifted, delta)
        np.testing.assert_allclose(updated[:3, 3] - initial[:3, 3], delta[:3], atol=1e-12)
        np.testing.assert_allclose(shifted_updated[:3, 3] - shifted[:3, 3], delta[:3], atol=1e-12)
        np.testing.assert_allclose(shifted_updated[:3, 3] - updated[:3, 3], world_shift, atol=1e-12)
        np.testing.assert_allclose(shifted_updated[:3, :3], updated[:3, :3], atol=1e-12)

    def test_holdout_never_splits_an_anchor(self):
        anchors = np.array([3, 9, 3, 12, 9, 15, 18, 12], dtype=np.int64)
        held_out = holdout_mask(anchors, 5)
        for anchor in np.unique(anchors):
            self.assertEqual(len(np.unique(held_out[anchors == anchor])), 1)

    def test_holdout_never_splits_a_reference_pair(self):
        query_cameras = np.array([0, 0, 0, 1, 1, 1])
        reference_frames = np.array(["a", "a", "b", "a", "a", "a"])
        reference_cameras = np.array([2, 2, 3, 2, 2, 2])
        pairs = reference_pair_ids(query_cameras, reference_frames, reference_cameras)
        held_out = holdout_mask(pairs, 5, min_count=2)
        for pair in np.unique(pairs):
            self.assertEqual(len(np.unique(held_out[pairs == pair])), 1)


if __name__ == "__main__":
    unittest.main()
