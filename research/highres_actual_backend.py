import numpy as np
import torch
from torch.autograd import Function

import nre_scoremap_pose_runner as nre
from highres_patch_pose_refiner import rotation_exp_torch


POSE_BOUNDS = np.asarray([.1, .1, .1, np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.)],
                         dtype=np.float64)


def pose_from_delta(delta, baseline_pose):
    rotation = rotation_exp_torch(delta[3:]) @ baseline_pose[:3, :3]
    translation = baseline_pose[:3, 3] + delta[:3]
    top = torch.cat((rotation, translation[:, None]), dim=1)
    bottom = torch.tensor([[0., 0., 0., 1.]], dtype=delta.dtype, device=delta.device)
    return torch.cat((top, bottom), dim=0)


def project_pixels(points, delta, baseline_pose, camera_to_body, calibration):
    rotation = rotation_exp_torch(delta[3:]) @ baseline_pose[:3, :3]
    translation = baseline_pose[:3, 3] + delta[:3]
    camera_rotation = rotation.unsqueeze(0) @ camera_to_body[:, :3, :3]
    camera_translation = translation.unsqueeze(0) + (
        rotation.unsqueeze(0) @ camera_to_body[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    camera_points = torch.bmm((points - camera_translation).unsqueeze(1),
                              camera_rotation).squeeze(1)
    homogeneous = torch.bmm(calibration, camera_points.unsqueeze(-1)).squeeze(-1)
    depth = camera_points[:, 2]
    denominator = homogeneous[:, 2].clamp_min(1e-12)
    pixels = homogeneous[:, :2] / denominator[:, None]
    return pixels, depth


def backend_objective(delta, pixels, points, precision, camera_to_body, calibration,
                      baseline_pose, lidar_information):
    projected, depth = project_pixels(points, delta, baseline_pose,
                                      camera_to_body, calibration)
    residual = projected - pixels
    visual = .5 * torch.einsum("ni,nij,nj->", residual, precision, residual)
    lidar = .5 * (delta @ lidar_information @ delta)
    behind_camera = 2. * (depth <= 0).to(dtype=delta.dtype).sum()
    return visual + lidar + behind_camera


def _implicit_pose_backward(ctx, grad_pose):
    pixels, points, precision, camera_to_body, calibration, baseline_pose, lidar_information, delta_star, active = \
        ctx.saved_tensors
    if grad_pose is None:
        return (None,) * 9
    with torch.enable_grad():
        pixels_var = pixels.detach().to(torch.float64).requires_grad_(True)
        delta_var = delta_star.detach().to(torch.float64).requires_grad_(True)
        objective = backend_objective(delta_var, pixels_var, points, precision,
                                      camera_to_body, calibration, baseline_pose,
                                      lidar_information)
        gradient = torch.autograd.grad(objective, delta_var, create_graph=True)[0]
        hessian = torch.stack([
            torch.autograd.grad(gradient[index], delta_var, retain_graph=True)[0]
            for index in range(6)
        ])
        hessian = .5 * (hessian + hessian.transpose(0, 1))
        free = ~active
        pose = pose_from_delta(delta_var, baseline_pose)
        output_delta_gradient = torch.autograd.grad(
            pose, delta_var, grad_outputs=grad_pose.to(torch.float64), retain_graph=True)[0]
        if not bool(free.any()):
            pixel_gradient = torch.zeros_like(pixels_var)
            condition = 1.
        else:
            reduced_hessian = hessian[free][:, free]
            condition_tensor = torch.linalg.cond(reduced_hessian)
            condition = float(condition_tensor.detach().cpu())
            if not torch.isfinite(condition_tensor) or condition > 1e12:
                raise RuntimeError("actual-backend implicit Hessian is singular or ill-conditioned")
            adjoint = torch.linalg.solve(reduced_hessian.transpose(0, 1),
                                         output_delta_gradient[free])
            cross = torch.autograd.grad(gradient[free], pixels_var,
                                        grad_outputs=adjoint, retain_graph=False)[0]
            pixel_gradient = -cross
        _ImplicitBackendPose.last_backward = {
            "free_dimensions": int(free.sum().item()),
            "active_dimensions": int(active.sum().item()),
            "reduced_hessian_condition": condition,
            "pixel_gradient_norm": float(torch.linalg.vector_norm(pixel_gradient).detach().cpu()),
        }
    return (pixel_gradient.to(dtype=pixels.dtype), None, None, None, None, None,
            None, None, None)


class _ImplicitBackendPose(Function):
    last_backward = {}

    @staticmethod
    def forward(ctx, pixels, points, precision, camera_to_body, calibration,
                baseline_pose, lidar_information, delta_star, active):
        ctx.save_for_backward(pixels, points, precision, camera_to_body, calibration,
                              baseline_pose, lidar_information, delta_star, active)
        return pose_from_delta(delta_star, baseline_pose)

    @staticmethod
    def backward(ctx, grad_pose):
        return _implicit_pose_backward(ctx, grad_pose)


def solve_actual_backend(frame, row, pixels, geometry, device, objective_tolerance=1e-6):
    corrected = pixels.detach().to(torch.float64).cpu().numpy()
    baseline = np.asarray(frame["baseline_pose"], dtype=np.float64)
    points = np.asarray(frame["points"], dtype=np.float64)
    cameras = np.asarray(frame["cameras"], dtype=np.int64)
    precisions = np.asarray(frame["peak_precisions"], dtype=np.float64)
    lidar_information = np.asarray(frame["lidar_information"], dtype=np.float64)
    optimized_pose, result, initial_value, final_value = nre.optimize_pose(
        baseline, points, cameras, row["views"], frame["cost_maps"], frame["map_valid"],
        frame["centers"], corrected, precisions, lidar_information, "peak")
    delta_np = np.asarray(result.x, dtype=np.float64)
    dtype = torch.float64
    delta = torch.as_tensor(delta_np, dtype=dtype, device=device)
    pixels64 = pixels.detach().to(dtype=dtype)
    torch_value = backend_objective(
        delta, pixels64, geometry["points"], geometry["precision"],
        geometry["camera_to_body"], geometry["calibration"],
        geometry["baseline_pose"], geometry["lidar_information"])
    objective_difference = abs(float(torch_value.detach().cpu()) - float(final_value))
    allowed_difference = objective_tolerance * max(1., abs(float(final_value)))
    if objective_difference > allowed_difference:
        raise RuntimeError("Torch objective does not match deployed backend objective: " +
                           str(objective_difference))

    delta_probe = delta.detach().clone().requires_grad_(True)
    objective_probe = backend_objective(
        delta_probe, pixels64, geometry["points"], geometry["precision"],
        geometry["camera_to_body"], geometry["calibration"],
        geometry["baseline_pose"], geometry["lidar_information"])
    objective_gradient = torch.autograd.grad(objective_probe, delta_probe)[0]
    bounds = torch.as_tensor(POSE_BOUNDS, dtype=dtype, device=device)
    near_lower = delta <= -bounds + 1e-7
    near_upper = delta >= bounds - 1e-7
    active = ((near_lower & (objective_gradient >= -1e-6)) |
              (near_upper & (objective_gradient <= 1e-6)))
    projected_gradient = objective_gradient.clone()
    projected_gradient = torch.where(near_lower, torch.minimum(projected_gradient,
                                                                torch.zeros_like(projected_gradient)),
                                     projected_gradient)
    projected_gradient = torch.where(near_upper, torch.maximum(projected_gradient,
                                                                torch.zeros_like(projected_gradient)),
                                     projected_gradient)
    kkt_inf = float(projected_gradient.abs().max().detach().cpu())
    implicit_pose = _ImplicitBackendPose.apply(
        pixels, geometry["points"], geometry["precision"], geometry["camera_to_body"],
        geometry["calibration"], geometry["baseline_pose"], geometry["lidar_information"],
        delta, active)
    return implicit_pose, {
        "success": bool(result.success),
        "status": int(result.status),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
        "initial_objective": float(initial_value),
        "final_objective": float(final_value),
        "objective_parity_abs": float(objective_difference),
        "projected_kkt_inf": kkt_inf,
        "active_dimensions": int(active.sum().item()),
        "free_dimensions": int((~active).sum().item()),
        "pose": optimized_pose,
        "delta": delta_np,
    }
