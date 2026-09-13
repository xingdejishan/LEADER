import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def project(T_WB, xyz, K, T_BC):
    T_CW = np.linalg.inv(T_WB @ T_BC)
    camera = xyz @ T_CW[:3, :3].T + T_CW[:3, 3]
    pixel = camera @ K.T
    uv = pixel[:, :2] / np.maximum(pixel[:, 2:], 1e-8)
    return uv, camera[:, 2]


def camera_score(T, uv, xyz, probability, K, T_BC, image_size_hw, subset=None, sigma=4.0):
    h, w = image_size_hw
    predicted, depth = project(T, xyz, K, T_BC)
    error = ((predicted - uv) ** 2).sum(1)
    inside = (depth > 0) & np.isfinite(predicted).all(1) & (predicted >= 0).all(1)
    inside &= (predicted[:, 0] < w) & (predicted[:, 1] < h)
    gaussian = np.exp(-np.minimum(error / (2 * sigma ** 2), 80)) / (2 * np.pi * sigma ** 2)
    gaussian[~inside] = 0
    p = np.clip(probability, 0, 1 - 1e-6)
    cost = -np.log(np.maximum(p * gaussian + (1 - p) / (h * w), 1e-15))
    blocks_xy = np.clip((uv / [w, h] * 4).astype(int), 0, 3)
    blocks = blocks_xy[:, 0] + 4 * blocks_xy[:, 1]
    keep = np.ones(len(uv), bool) if subset is None else subset
    block_costs = [cost[(blocks == b) & keep].mean() for b in range(16) if ((blocks == b) & keep).any()]
    if not block_costs:
        return float(np.log(h * w))
    return float(np.mean(block_costs))


def lidar_score(T, body, world, weight):
    residual = np.linalg.norm(body @ T[:3, :3].T + T[:3, 3] - world, axis=1)
    return float(np.sum(weight * np.minimum((residual / .3) ** 2, 1)))


def update_pose(T, delta):
    result = T.copy()
    result[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ T[:3, :3]
    result[:3, 3] += delta[:3]
    return result


def spatial_budget(uv, image_size_hw, budget=256):
    h, w = image_size_hw
    cells = np.clip((uv / [w, h] * 4).astype(int), 0, 3)
    ids = cells[:, 0] + 4 * cells[:, 1]
    bins = [np.flatnonzero(ids == i).tolist() for i in range(16)]
    selected = []
    for position in range(max(map(len, bins), default=0)):
        for values in bins:
            if position < len(values):
                selected.append(values[position])
                if len(selected) == budget:
                    return np.array(selected)
    return np.array(selected, dtype=int)


def fuse(pool, correspondences, T_BC, enable_visual=True):
    initial = pool['v1_two_stage'].astype(float).copy()
    Q = pool['T_corr']
    body = (pool['c_local_all'].astype(float) - Q[:3, 3]) @ Q[:3, :3]
    world = pool['c_pred_all'].astype(float) + pool['center_t']
    weights = np.exp(np.log(10) / np.pi * np.arctan(np.clip(pool['u_pred_all'].reshape(-1), -10*np.pi, 10*np.pi)))
    weights /= weights.sum()
    candidates = np.concatenate([initial[None], pool['candidate_T_WB']])
    candidates = candidates[np.isfinite(candidates).all(axis=(1, 2))]
    if not enable_visual:
        objective = lambda delta: lidar_score(update_pose(initial, delta), body, world, weights)
        optimum = minimize(objective, np.zeros(6), method='Powell', options={'maxiter': 20, 'maxfev': 150})
        refined = update_pose(initial, optimum.x)
        return (refined if objective(optimum.x) < objective(np.zeros(6)) else initial), dict(accepted=True, mode='lidar_only_refinement')
    uv, xyz, p = correspondences['uv'], correspondences['xyz'], correspondences['reliability']
    size, K = correspondences['image_size_hw'], correspondences['K']
    if xyz.ndim == 2:
        xyz = xyz[None]
    if p.ndim == 1:
        p = p[None]
    take = spatial_budget(uv, size)
    uv, xyz, p = uv[take], xyz[:, take], p[:, take]
    if not len(take) or np.max(p.mean(1)) < .05:
        return initial, dict(accepted=False, reason='visual_reliability_insufficient')
    h, w = size
    cells = np.clip((uv / [w, h] * 4).astype(int), 0, 3)
    fit = (cells.sum(1) % 2) == 0
    holdout = ~fit
    if min(fit.sum(), holdout.sum()) < 12:
        return initial, dict(accepted=False, reason='spatial_holdout_insufficient')
    normalizer = np.log(h*w)
    def objective(T, hypothesis, subset):
        return .5 * lidar_score(T, body, world, weights) + .5 * camera_score(T, uv, xyz[hypothesis], p[hypothesis], K, T_BC, size, subset) / normalizer
    ranked = sorted((objective(T, hyp, fit), c, hyp) for c, T in enumerate(candidates) for hyp in range(len(xyz)))
    _, candidate_id, hypothesis = ranked[0]
    winner = candidates[candidate_id]
    initial_fit = objective(winner, hypothesis, fit)
    optimization = minimize(lambda delta: objective(update_pose(winner, delta), hypothesis, fit), np.zeros(6), method='Powell',
        options={'maxiter': 10, 'maxfev': 150})
    refined = update_pose(winner, optimization.x)
    if objective(refined, hypothesis, fit) < initial_fit:
        winner = refined
    base_holdout = objective(initial, hypothesis, holdout)
    new_holdout = objective(winner, hypothesis, holdout)
    projected, depth = project(winner, xyz[hypothesis], K, T_BC)
    support = (np.linalg.norm(projected - uv, axis=1) < 10) & (depth > 0) & (p[hypothesis] >= .5) & holdout
    occupied = len(set(map(tuple, cells[support].tolist())))
    ambiguous = False
    for score, c, h_id in ranked[1:]:
        if score - ranked[0][0] >= .01:
            break
        dt = np.linalg.norm(candidates[c, :3, 3] - candidates[candidate_id, :3, 3])
        angle = Rotation.from_matrix(candidates[c, :3, :3] @ candidates[candidate_id, :3, :3].T).magnitude()
        if dt > .5 or angle > np.deg2rad(3):
            ambiguous = True
            break
    accepted = bool(new_holdout < base_holdout - 1e-4 and support.sum() >= 12 and occupied >= 3 and not ambiguous)
    return winner if accepted else initial, dict(accepted=accepted, hypothesis=int(hypothesis), candidate_id=int(candidate_id),
        holdout_before=base_holdout, holdout_after=new_holdout, holdout_support=int(support.sum()), holdout_blocks=occupied, ambiguous=ambiguous)
