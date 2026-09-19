import numpy as np

from local_visual_refinement_roma import apply_local_delta
from oracle_pose_refinement import project_world


def skew(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    output = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    output[:, 0, 1], output[:, 0, 2] = -vectors[:, 2], vectors[:, 1]
    output[:, 1, 0], output[:, 1, 2] = vectors[:, 2], -vectors[:, 0]
    output[:, 2, 0], output[:, 2, 1] = -vectors[:, 1], vectors[:, 0]
    return output


def frozen_lidar_information(pose, evidence, inflation=1., residual_floor_m=.005, regularization_ratio=1e-6):
    source = np.asarray(evidence["source"], dtype=np.float64)
    target = np.asarray(evidence["target"], dtype=np.float64)
    weights = np.asarray(evidence["weights"], dtype=np.float64)
    transformed = source @ pose[:3, :3].T + pose[:3, 3]
    residual = transformed - target
    jacobian = np.concatenate((np.broadcast_to(np.eye(3), (len(source), 3, 3)), -skew(transformed)), axis=2)
    hessian = np.einsum("n,nki,nkj->ij", weights, jacobian, jacobian)
    regularization = regularization_ratio * max(float(np.trace(hessian) / 6), 1e-12)
    norms = np.linalg.norm(residual, axis=1)
    residual_scale = max(1.4826 * float(np.median(norms)), residual_floor_m)
    covariance = inflation ** 2 * residual_scale ** 2 * np.linalg.inv(hessian + regularization * np.eye(6))
    return {"source": source, "target": target, "weights": weights, "residual": residual,
            "jacobian": jacobian, "hessian": hessian, "regularization": regularization,
            "residual_scale_m": residual_scale, "covariance": covariance}


def pixel_pose_jacobian(points, pose, camera_to_body, calibration, translation_step_m=1e-4, rotation_step_rad=1e-5):
    points = np.asarray(points, dtype=np.float64)
    base, _ = project_world(points, pose, camera_to_body, calibration)
    jacobian = np.empty((len(points), 2, 6), dtype=np.float64)
    for axis in range(6):
        delta = np.zeros(6, dtype=np.float64)
        delta[axis] = translation_step_m if axis < 3 else rotation_step_rad
        shifted, _ = project_world(points, apply_local_delta(pose, delta), camera_to_body, calibration)
        jacobian[:, :, axis] = (shifted - base) / delta[axis]
    return base, jacobian


def adaptive_innovation_gate(points, matched_pixels, cameras, precisions, pose, views, lidar_covariance,
                             covariance_inflation=1., innovation_floor_px=1., threshold=9.):
    points = np.asarray(points, dtype=np.float64)
    matched_pixels = np.asarray(matched_pixels, dtype=np.float64)
    cameras = np.asarray(cameras, dtype=np.int64)
    precisions = np.asarray(precisions, dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    d2 = np.full(len(points), np.inf, dtype=np.float64)
    covariance = np.empty((len(points), 2, 2), dtype=np.float64)
    view_by_camera = {int(view["camera"]): view for view in views}
    for camera in np.unique(cameras):
        keep = cameras == camera
        view = view_by_camera[int(camera)]
        base, jacobian = pixel_pose_jacobian(points[keep], pose, np.asarray(view["camera_to_body"], dtype=np.float64),
                                             np.loadtxt(view["calibration"]).astype(np.float64))
        roma_covariance = np.linalg.inv(precisions[keep])
        innovation = covariance_inflation ** 2 * np.einsum("nai,ij,nbj->nab", jacobian, lidar_covariance, jacobian)
        innovation += roma_covariance + innovation_floor_px ** 2 * np.eye(2)[None]
        innovation = .5 * (innovation + np.swapaxes(innovation, 1, 2))
        delta = matched_pixels[keep] - base
        inverse = np.linalg.inv(innovation)
        local_d2 = np.einsum("ni,nij,nj->n", delta, inverse, delta)
        indices = np.where(keep)[0]
        covariance[indices], d2[indices] = innovation, local_d2
        valid[indices] = np.isfinite(local_d2) & (local_d2 < threshold)
    return valid, d2, covariance
