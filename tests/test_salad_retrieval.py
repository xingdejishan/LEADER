"""The reference-selection contract: SALAD, one channel, K fixed, no pose anywhere.

These tests never build the model or read the 160 MB feature matrix; they check the
parts that a caller can get wrong -- the signature, the fixed K, and the refusal of a
pose, a position or a rotation.
"""
import inspect
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import salad_retrieval as SR                                          # noqa: E402
import visual_branch as VB                                            # noqa: E402


class Stub(object):
    """Enough of a branch for `retrieve`: the guard runs before the model is touched."""


class SaladRetrievalTests(unittest.TestCase):
    def test_k_is_fixed_at_five(self):
        self.assertEqual(SR.TOP_K_REFERENCES, 5)
        self.assertEqual(SR.DIMENSION, 8448)

    def test_retrieve_takes_only_an_image_path(self):
        parameters = inspect.signature(VB.VisualBranch.retrieve).parameters
        names = [n for n in parameters if n != 'self']
        self.assertEqual(names, ['image_path', 'extra'])
        for forbidden in ('pose', 'pose_prior', 'prior', 'position', 'rotation',
                          'channels', 'k'):
            self.assertNotIn(forbidden, ' '.join(names))

    def test_hypothesize_has_no_pose_argument(self):
        parameters = inspect.signature(VB.VisualBranch.hypothesize).parameters
        self.assertNotIn('pose_prior', parameters)
        self.assertNotIn('pose', parameters)

    def test_poses_positions_and_rotations_are_refused(self):
        for value in (np.eye(4), np.eye(4, dtype=np.int32), np.zeros(3), np.zeros(3),
                      np.zeros((1, 3)), np.eye(3)):
            with self.assertRaises(TypeError):
                VB.VisualBranch.retrieve(Stub(), 'image.jpg', value)

    def test_other_extra_arguments_are_refused(self):
        with self.assertRaises(TypeError):
            VB.VisualBranch.retrieve(Stub(), 'image.jpg', 20)

    def test_reject_pose_like_ignores_none_only(self):
        SR.reject_pose_like(None, 'x')
        with self.assertRaises(TypeError):
            SR.reject_pose_like(np.eye(4), 'x')


if __name__ == '__main__':
    unittest.main()
