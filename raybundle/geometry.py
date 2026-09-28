import torch


def skew(x):
    zero = torch.zeros_like(x[..., 0])
    a, b, c = x.unbind(-1)
    return torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), -1).reshape(*x.shape[:-1], 3, 3)


def retract(pose, delta):
    top = torch.cat((skew(delta[3:]), delta[:3, None]), -1)
    algebra = torch.cat((top, delta.new_zeros(1, 4)), 0)
    return pose @ torch.matrix_exp(algebra)


def tangent_basis(direction):
    axis = torch.nn.functional.one_hot(direction.abs().argmin(-1), 3).to(direction)
    first = torch.nn.functional.normalize(torch.cross(direction, axis, dim=-1), dim=-1)
    second = torch.cross(direction, first, dim=-1)
    return torch.stack((first, second), -2)


def bearing_linearization(pose, xyz, origins, directions):
    rotation, translation = pose[:3, :3], pose[:3, 3]
    body = (xyz - translation) @ rotation
    camera = body - origins
    distance = camera.norm(dim=-1).clamp_min(1e-8)
    unit = camera / distance[..., None]
    basis = tangent_basis(directions)
    eye = torch.eye(3, dtype=xyz.dtype, device=xyz.device)
    normalizer = (eye - unit[..., :, None] * unit[..., None, :]) / distance[..., None, None]
    derivative = basis @ normalizer
    pose_jacobian = derivative @ torch.cat((-eye.expand(*body.shape[:-1], 3, 3), skew(body)), -1)
    point_jacobian = derivative @ rotation.T
    residual = (basis @ (unit - directions)[..., None]).squeeze(-1)
    visible = (camera * directions).sum(-1) > 0
    return residual, pose_jacobian, point_jacobian, visible


def lidar_normal(pose, source, target, weights, independent_sigma, shared_covariance):
    rotation = pose[:3, :3]
    residual = source @ rotation.T + pose[:3, 3] - target
    jacobian = torch.cat((rotation.expand(len(source), 3, 3), -rotation @ skew(source)), -1)
    precision = weights / independent_sigma ** 2
    hessian = torch.einsum('nai,n,naj->ij', jacobian, precision, jacobian)
    gradient = torch.einsum('nai,n,na->i', jacobian, precision, residual)
    nuisance = torch.linalg.inv(shared_covariance) + hessian
    hessian_effective = hessian - hessian @ torch.linalg.solve(nuisance, hessian)
    gradient_effective = gradient - hessian @ torch.linalg.solve(nuisance, gradient)
    return (hessian_effective + hessian_effective.T) * .5, gradient_effective


def normal_system(lidar_h, lidar_g, points, map_mean, map_precision, indices,
                  residual, pose_jacobian, point_jacobian, weights, variance):
    ids = indices.clamp_min(0).reshape(-1)
    w = (weights * (indices >= 0)).reshape(-1) / variance
    j = pose_jacobian.reshape(-1, 2, 6)
    d = point_jacobian.reshape(-1, 2, 3)
    r = residual.reshape(-1, 2)
    h = lidar_h + torch.einsum('nai,n,naj->ij', j, w, j)
    g = lidar_g + torch.einsum('nai,n,na->i', j, w, r)
    c = map_precision.index_add(0, ids, torch.einsum('nai,n,naj->nij', d, w, d))
    b = points.new_zeros(len(points), 6, 3).index_add(0, ids, torch.einsum('nai,n,naj->nij', j, w, d))
    initial_gradient = (map_precision @ (points - map_mean)[..., None]).squeeze(-1)
    gx = initial_gradient.index_add(0, ids, torch.einsum('nai,n,na->ni', d, w, r))
    c_inv = torch.linalg.inv(c)
    bc = b @ c_inv
    schur = h - (bc @ b.transpose(-1, -2)).sum(0)
    reduced_g = g - (bc @ gx[..., None]).squeeze(-1).sum(0)
    schur = (schur + schur.T) * .5
    eigenvalues = torch.linalg.eigvalsh(schur.detach())
    if not torch.isfinite(eigenvalues).all() or eigenvalues[0] <= 1e-10:
        raise ValueError('LiDAR plus ray geometry does not constrain all six pose directions')
    covariance = torch.linalg.inv(schur)
    delta_pose = -torch.linalg.solve(schur, reduced_g)
    delta_points = -(c_inv @ (gx + (b.transpose(-1, -2) @ delta_pose).squeeze(-1))[..., None]).squeeze(-1)
    return delta_pose, delta_points, covariance, c_inv, bc


def predictive_score(system, indices, residual, pose_jacobian, point_jacobian, variance):
    delta, dx, covariance, c_inv, bc = system
    ids = indices.clamp_min(0)
    j, d = pose_jacobian, point_jacobian
    effective_j = j - d @ bc[ids].transpose(-1, -2)
    predicted = residual + (j @ delta).squeeze(-1) + (d @ dx[ids][..., None]).squeeze(-1)
    eye = torch.eye(2, dtype=j.dtype, device=j.device)
    s = variance * eye + d @ c_inv[ids] @ d.transpose(-1, -2)
    s = s + effective_j @ covariance @ effective_j.transpose(-1, -2)
    mahalanobis = (predicted[..., None, :] @ torch.linalg.solve(s, predicted[..., None])).squeeze(-1).squeeze(-1)
    sign, logdet = torch.linalg.slogdet(s)
    if torch.any(sign <= 0):
        raise FloatingPointError('non-positive predictive covariance')
    return -.5 * (mahalanobis + logdet + 2 * torch.log(j.new_tensor(2 * torch.pi)))
