import numpy as np

from research.prevoxel_multiview.surface_patch_refinement import (apply_pose_delta, apply_rotation_delta,
                                                                   bind_reference_patch, build_visibility_depth_buffer,
                                                                   check_patch_visibility, finite_difference_jacobian,
                                                                   normalize_patch, sample_bilinear)


def test_normalize_patch_removes_affine_intensity_change():
    values = np.arange(16, dtype=np.float64)
    first, _ = normalize_patch(values)
    second, _ = normalize_patch(values * 3.0 + 0.4)
    np.testing.assert_allclose(first, second)


def test_rotation_update_preserves_translation():
    initial = np.eye(4)
    initial[:3, 3] = [1., 2., 3.]
    updated = apply_rotation_delta(initial, [0., 0., .2])
    np.testing.assert_allclose(updated[:3, 3], initial[:3, 3])
    assert not np.allclose(updated[:3, :3], initial[:3, :3])


def test_joint_update_changes_translation_and_rotation():
    initial = np.eye(4)
    updated = apply_pose_delta(initial, [.1, 0., 0., 0., 0., .2])
    np.testing.assert_allclose(updated[:3, 3], [.1, 0., 0.])
    assert not np.allclose(updated[:3, :3], initial[:3, :3])


def test_bilinear_sampling_uses_fractional_coordinates():
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    values, valid = sample_bilinear(image, np.array([[.5, .5]]))
    np.testing.assert_allclose(values, [2.])
    assert valid.tolist() == [True]


def test_fixed_difference_jacobian_uses_absolute_steps():
    jacobian = finite_difference_jacobian(lambda x: np.array([x[0] ** 2]), [1e-3])
    np.testing.assert_allclose(jacobian(np.array([2.])), [[4.]], rtol=1e-10)


def test_visibility_check_rejects_occluded_surface_point():
    calibration = np.array([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]])
    depth_buffer = build_visibility_depth_buffer(
        np.array([[0., 0., 5.], [0., 0., 2.]]), np.eye(4), np.eye(4), calibration, (20, 20), 4)
    accepted, stats = check_patch_visibility(np.full((4, 2), 5.), np.full(4, 5.), depth_buffer,
                                             4, 1, .1, .5, .9)
    assert not accepted
    assert stats["coverage"] == 1.


def test_surface_patch_rays_bind_to_plane():
    view = {"camera_to_body": np.eye(4), "shape": (20, 20)}
    calibration = np.eye(3)
    plane = (np.array([0., 0., 1.]), 5., 10, np.array([2., 1., .01]))
    points, pixels = bind_reference_patch(np.array([0., 0., 5.]), np.eye(4), view,
                                          np.array([10., 10.]), plane, 4, calibration)
    np.testing.assert_allclose(points[:, 2], 5.)
    np.testing.assert_allclose(points[:, :2] / 5., pixels)
