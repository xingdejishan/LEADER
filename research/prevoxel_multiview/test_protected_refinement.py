import unittest
import tempfile

import numpy as np

from local_visual_refinement_roma import apply_local_delta, holdout_mask, reference_pair_ids
from joint_lidar_camera_refinement import adaptive_innovation_gate, frozen_lidar_information


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

    def test_lidar_information_and_adaptive_distance_are_world_translation_invariant(self):
        source = np.array([[0., 0., 5.], [1., 0., 5.], [0., 1., 5.], [1., 1., 6.], [2., 0., 6.], [0., 2., 6.]])
        pose = np.eye(4)
        pose[:3, 3] = [200., -100., 3.]
        target = source + pose[:3, 3]
        evidence = {"source": source, "target": target, "weights": np.ones(len(source))}
        shift = np.array([1000., -700., 50.])
        shifted_pose = pose.copy()
        shifted_pose[:3, 3] += shift
        shifted_evidence = {"source": source, "target": target + shift, "weights": np.ones(len(source))}
        first = frozen_lidar_information(pose, evidence)
        second = frozen_lidar_information(shifted_pose, shifted_evidence)
        np.testing.assert_allclose(first["hessian"], second["hessian"], atol=1e-10)
        np.testing.assert_allclose(first["covariance"], second["covariance"], atol=1e-10)
        calibration = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        calibration.close()
        np.savetxt(calibration.name, np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]))
        try:
            views = [{"camera": 0, "camera_to_body": np.eye(4).tolist(), "calibration": calibration.name}]
            world = source + pose[:3, 3]
            pixels = np.array([[50., 50.], [70., 50.], [50., 70.], [66.6667, 66.6667], [83.3333, 50.], [50., 83.3333]])
            kwargs = {"matched_pixels": pixels, "cameras": np.zeros(len(world), dtype=np.int64),
                      "precisions": np.broadcast_to(np.eye(2), (len(world), 2, 2)), "views": views,
                      "lidar_covariance": first["covariance"]}
            _, first_d2, _ = adaptive_innovation_gate(world, pose=pose, **kwargs)
            _, second_d2, _ = adaptive_innovation_gate(world + shift, pose=shifted_pose, **kwargs)
            np.testing.assert_allclose(first_d2, second_d2, rtol=1e-7, atol=1e-7)
        finally:
            import os
            os.unlink(calibration.name)

    def test_frozen_lidar_rotation_jacobian_matches_local_pose_update(self):
        source = np.array([[.5, -1., 3.], [2., .3, 4.], [-1., 1.2, 5.]])
        pose = np.eye(4)
        pose[:3, 3] = [300., -200., 10.]
        target = source + pose[:3, 3]
        information = frozen_lidar_information(pose, {"source": source, "target": target, "weights": np.ones(len(source))})
        analytic = information["jacobian"]
        numeric = np.empty_like(analytic)
        base = source @ pose[:3, :3].T + pose[:3, 3] - target
        for axis in range(6):
            delta = np.zeros(6)
            delta[axis] = 1e-6
            updated = apply_local_delta(pose, delta)
            residual = source @ updated[:3, :3].T + updated[:3, 3] - target
            numeric[:, :, axis] = (residual - base) / delta[axis]
        np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
