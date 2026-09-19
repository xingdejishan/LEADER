import torch


def transform_points(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_transform(source, target, weights):
    weights = weights.clamp_min(0)
    total = weights.sum()
    if source.shape[0] < 3 or total <= 1e-8:
        raise RuntimeError("Insufficient weighted correspondences")
    source_mean = (source * weights[:, None]).sum(0) / total
    target_mean = (target * weights[:, None]).sum(0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ (target_centered * weights[:, None])
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, dtype=source.dtype, device=source.device)
    correction[-1, -1] = torch.det(vh.T @ u.T)
    rotation = vh.T @ correction @ u.T
    translation = target_mean - rotation @ source_mean
    transform = torch.eye(4, dtype=source.dtype, device=source.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def refine(initial, source, target, thresholds, return_evidence=False):
    transform = initial.clone()
    selected_count = 0
    evidence = None
    for threshold in thresholds:
        residual = torch.linalg.norm(transform_points(transform, source) - target, dim=1)
        mask = residual < threshold
        selected_count = int(mask.sum())
        if selected_count < 6:
            break
        scaled = residual[mask] / threshold
        weights = (1.0 - scaled.square()).clamp_min(0).square()
        evidence = {
            "source": source[mask],
            "target": target[mask],
            "weights": weights,
            "residual": residual[mask],
            "threshold": threshold,
        }
        transform = rigid_transform(source[mask], target[mask], weights)
    return (transform, selected_count, evidence) if return_evidence else (transform, selected_count)


def full_pool_refine(initial, source, target, return_evidence=False):
    return refine(initial, source, target, (1.2, 0.6), return_evidence)
