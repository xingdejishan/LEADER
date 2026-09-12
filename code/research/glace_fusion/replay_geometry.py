import numpy as np

from .joint_solver import JointProblem, JointSolverConfig, lidar_reliability_weights
from .pose_boundary import solver_pose


def candidates_from_pool(pool):
    values = [pool['v1_two_stage'], pool['leader'], *pool['candidate_T_WB']]
    names = ['v1_two_stage', 'leader'] + [f'sc2_{i}' for i in range(len(pool['candidate_T_WB']))]
    return np.array([solver_pose(p) for p in values]), names


def grid_cells(uv, shape, grid=4):
    height, width = shape
    cells = np.floor(np.asarray(uv) / [width, height] * grid).astype(int)
    cells = np.clip(cells, 0, grid - 1)
    return cells[:, 1] * grid + cells[:, 0]


def sample_pixels(uv, shape, budget, seed):
    rng = np.random.default_rng(seed)
    cells = grid_cells(uv, shape)
    groups = [rng.permutation(np.flatnonzero(cells == k)).tolist() for k in range(16)]
    selected = []
    while len(selected) < min(budget, len(uv)):
        for group in groups:
            if group and len(selected) < budget:
                selected.append(group.pop())
    return np.sort(np.asarray(selected, dtype=int))


def matched_subset(uv, reference_mask, shape, seed):
    rng = np.random.default_rng(seed)
    cells = grid_cells(uv, shape)
    selected = []
    for cell in range(16):
        count = int(np.sum(reference_mask & (cells == cell)))
        selected.extend(rng.choice(np.flatnonzero(cells == cell), count, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def point_errors(xyz, uv, K, T_WC, shape, target_camera=None):
    xyz, uv = np.asarray(xyz), np.asarray(uv)
    camera = (xyz - T_WC[:3, 3]) @ T_WC[:3, :3]
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    positive = np.isfinite(camera).all(1) & (camera[:, 2] > 1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        projection = camera @ K.T
        projected = projection[:, :2] / projection[:, 2:]
    signed = projected - uv
    finite_projection = positive & np.isfinite(projected).all(1)
    height, width = shape
    in_bounds = finite_projection & (projected[:, 0] >= 0) & (projected[:, 0] < width) & (projected[:, 1] >= 0) & (projected[:, 1] < height)
    squared = np.sum(signed ** 2, axis=1)
    squared[~finite_projection] = np.inf
    along = np.sum(camera * rays, axis=1)
    perpendicular = np.linalg.norm(camera - along[:, None] * rays, axis=1)
    angle = np.degrees(np.arctan2(perpendicular, along))
    angle[~positive] = np.inf
    result = dict(camera=camera, signed_px=signed, squared_px=squared, positive=positive,
                  in_bounds=in_bounds, angle_deg=angle, perpendicular_m=perpendicular)
    if target_camera is not None:
        target_range = np.linalg.norm(target_camera, axis=1)
        result.update(along_error_m=along - target_range,
                      range_error_m=np.linalg.norm(camera, axis=1) - target_range)
    return result


def camera_scores(poses, xyz, uv, K, extrinsic, shape, block_weights=False, strict=False):
    if not len(uv):
        return np.ones(len(poses))
    weights = np.full(len(uv), 1 / len(uv))
    if block_weights:
        cells = grid_cells(uv, shape)
        counts = np.bincount(cells, minlength=16)
        weights = 1 / counts[cells] / np.count_nonzero(counts)
    scores = []
    for pose in poses:
        errors = point_errors(xyz, uv, K, pose @ extrinsic, shape)
        penalties = np.minimum(errors['squared_px'] / 100, 1)
        if strict:
            penalties[~errors['in_bounds']] = 1
        scores.append(float(penalties @ weights))
    return np.asarray(scores)


def lidar_arrays(pool):
    Q = pool['T_corr']
    body = (pool['c_local_all'].astype(float) - Q[:3, 3]) @ Q[:3, :3]
    world = pool['c_pred_all'].astype(float) + pool['center_t']
    return body, world, pool['u_pred_all']


def lidar_scores(poses, pool):
    body, world, reliability = lidar_arrays(pool)
    weights = lidar_reliability_weights(reliability)
    return np.array([np.minimum(np.sum((body @ T[:3, :3].T + T[:3, 3] - world) ** 2, axis=1) / .09, 1) @ weights for T in poses])


def refine_selected(pose, pool, xyz, uv, K, extrinsic, shape, block_weights=False):
    cfg = JointSolverConfig(camera_scale_px=10.)
    problem = JointProblem(*lidar_arrays(pool), uv, xyz, K, extrinsic, cfg)
    if block_weights and len(uv):
        cells = grid_cells(uv, shape)
        counts = np.bincount(cells, minlength=16)
        problem.w_C = 1 / counts[cells] / np.count_nonzero(counts)
    try:
        refined, diagnostics = problem.refine(pose)
        return solver_pose(refined), diagnostics
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return pose.copy(), dict(success=False, reason=type(error).__name__)


def spatial_holdout(uv, shape):
    cells = grid_cells(uv, shape)
    selection = ((cells // 4 + cells % 4) % 2) == 0
    return selection, ~selection


def accept_on_holdout(baseline, proposal, pool, xyz, uv, K, extrinsic, shape):
    if len(uv) < 6 or len(np.unique(grid_cells(uv, shape))) < 3:
        return False, 'insufficient_heldout_coverage'
    poses = np.array([baseline, proposal])
    camera = camera_scores(poses, xyz, uv, K, extrinsic, shape, block_weights=True, strict=True)
    lidar = lidar_scores(poses, pool)
    errors = point_errors(xyz, uv, K, proposal @ extrinsic, shape)
    support = errors['in_bounds'] & (errors['squared_px'] <= 100)
    if support.sum() < 6 or support.mean() < .2 or len(np.unique(grid_cells(uv[support], shape))) < 3:
        return False, 'insufficient_heldout_support'
    accepted = bool(camera[1] < camera[0] - 1e-8 and lidar[1] <= lidar[0] + 1e-8)
    return accepted, 'heldout_improved_and_lidar_nonworse' if accepted else 'heldout_or_lidar_not_improved'
