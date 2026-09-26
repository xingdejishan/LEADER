import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


RAW_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
MAP_PROTOCOL = 'train_lidar_sift_world_points_v1'
ONLINE_PROTOCOL = 'local905_gt_isolated_online_v1'
CHECKPOINT_SHA256 = 'e369fd653badec1dccc8753e4521cfd63b263e1cb59e0b72c55762d2ef42502f'
SPLIT_SHA256 = '948bcb0d81ea9be9d5627c7b00bab36efa8fcacec2a3027b256c9574308835ba'
FROZEN_PREDICTION_SHA256 = {
    'val': 'fd6691630832c65b474650f481924abed3f5e52b7b807228fd7ac17585a418fc',
    'test': '5915d5f55f1605b96d163a6da7fdf4818df43214b10241daae619cd4d696e289',
}
RATIO_THRESHOLD = 0.75
INITIAL_REPROJECTION_LIMIT_PX = 12.0
MAP_KEYPOINT_RADIUS_PX = 5.0
MAX_REFERENCE_DISTANCE_M = 20.0
MAX_REFERENCES = 5
GRID_COLUMNS = 8
GRID_ROWS = 6
MAX_PER_GRID_CELL = 3
MAX_CORRESPONDENCES = 128
MIN_CORRESPONDENCES = 6
TRANSLATION_BOUND_M = 0.1
ROTATION_BOUND_RAD = math.radians(1.0)
HUBER_SCALE_PX = 4.0
MAX_ITERATIONS = 20
DEGENERACY_CONDITION_LIMIT = 1e8


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def transform_points(transform, points):
    return (transform[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + transform[:3, 3]


def project_camera(points_camera, intrinsics):
    points_camera = np.asarray(points_camera, dtype=np.float64)
    projected = (np.asarray(intrinsics, dtype=np.float64) @ points_camera.T).T
    valid = np.isfinite(projected).all(axis=1) & (points_camera[:, 2] > 1e-6)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def read_body_scan(path):
    raw = np.fromfile(path, dtype=RAW_DTYPE)
    points = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float64) * 0.005 - 100.0
    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges > 1.0) & (ranges < 100.0)
    return points[keep], np.flatnonzero(keep).astype(np.int32)


def read_camera_pose(path):
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f'Invalid training camera pose: {path}')
    return pose


def resolve_image(manifest_path, record):
    path = Path(record['image'])
    return path.resolve() if path.is_absolute() else (Path(manifest_path).resolve().parent / path).resolve()


def extract_sift(image_path, max_features=4096):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read Cam5 image: {image_path}')
    detector = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    xy = np.asarray([item.pt for item in keypoints], dtype=np.float64).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors, dtype=np.float32)
    return image, xy, descriptors


def associate_keypoints(keypoints_xy, body_points, raw_point_indices, camera_from_body, intrinsics, image_size, radius_px=MAP_KEYPOINT_RADIUS_PX):
    width, height = map(int, image_size)
    points_camera = transform_points(camera_from_body, body_points)
    uv, valid = project_camera(points_camera, intrinsics)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not valid.any() or not len(keypoints_xy):
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    point_rows = np.flatnonzero(valid)
    pixels = np.rint(uv[point_rows]).astype(np.int32)
    in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    point_rows = point_rows[in_bounds]
    pixels = pixels[in_bounds]
    if not len(point_rows):
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    flat_pixels = pixels[:, 1].astype(np.int64) * width + pixels[:, 0]
    depths = points_camera[point_rows, 2]
    order = np.lexsort((depths, flat_pixels))
    ordered_pixels = flat_pixels[order]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    visible_rows = point_rows[order[first]]
    visible_pixels = ordered_pixels[first]
    zbuffer_rows = np.full(width * height, -1, dtype=np.int32)
    zbuffer_rows[visible_pixels] = visible_rows.astype(np.int32)
    offsets = np.asarray([(dx, dy) for dy in range(-math.ceil(radius_px), math.ceil(radius_px) + 1)
                          for dx in range(-math.ceil(radius_px), math.ceil(radius_px) + 1)
                          if dx * dx + dy * dy <= radius_px * radius_px], dtype=np.int32)
    rounded = np.rint(keypoints_xy).astype(np.int32)
    candidate_x = rounded[:, None, 0] + offsets[None, :, 0]
    candidate_y = rounded[:, None, 1] + offsets[None, :, 1]
    in_image = (candidate_x >= 0) & (candidate_x < width) & (candidate_y >= 0) & (candidate_y < height)
    safe_x = np.clip(candidate_x, 0, width - 1)
    safe_y = np.clip(candidate_y, 0, height - 1)
    candidate_rows = zbuffer_rows[safe_y * width + safe_x]
    has_point = in_image & (candidate_rows >= 0)
    candidate_distance = ((candidate_x - keypoints_xy[:, None, 0]) ** 2
                          + (candidate_y - keypoints_xy[:, None, 1]) ** 2)
    candidate_distance[~has_point] = np.inf
    candidate_depth = np.full(candidate_rows.shape, np.inf, dtype=np.float64)
    selected = has_point
    candidate_depth[selected] = points_camera[candidate_rows[selected], 2]
    score = candidate_distance + candidate_depth * 1e-10
    nearest = np.argmin(score, axis=1)
    best_distance = candidate_distance[np.arange(len(keypoints_xy)), nearest]
    supported = np.isfinite(best_distance) & (best_distance <= radius_px * radius_px)
    feature_rows = np.flatnonzero(supported)
    point_rows = candidate_rows[feature_rows, nearest[feature_rows]].astype(np.int64)
    return feature_rows, point_rows, uv[point_rows]


def verify_calibration(raw_manifest, scene_meta):
    expected_camera_from_body = np.linalg.inv(np.asarray(scene_meta['T_BC_camera_to_body'], dtype=np.float64))
    raw_k = np.asarray(scene_meta['K_raw'], dtype=np.float64)
    if not np.isfinite(expected_camera_from_body).all() or not np.isfinite(raw_k).all():
        raise ValueError('Invalid scene calibration')
    calibration_records = list(raw_manifest['frames'].values())
    if not calibration_records:
        raise ValueError('Empty raw manifest')
    for record in calibration_records:
        intrinsics = np.asarray(record['K'], dtype=np.float64)
        camera_from_body = np.asarray(record['T_camera_lidar'], dtype=np.float64)
        if intrinsics.shape != (3, 3) or camera_from_body.shape != (4, 4):
            raise ValueError('Invalid raw-manifest camera calibration shape')
        if not np.allclose(camera_from_body, expected_camera_from_body, atol=2e-5, rtol=0):
            raise ValueError('Raw-manifest T_camera_lidar is inconsistent with synced body-frame camera calibration')
        scaled_k = raw_k.copy()
        scaled_k[:2] *= 0.5
        if not np.allclose(scaled_k, intrinsics, atol=1e-5, rtol=0):
            raise ValueError('Raw-manifest intrinsics are not the fixed half-resolution Cam5 intrinsics')


def build_train_map(data_root, split_path, raw_manifest_path, output_dir):
    data_root = Path(data_root).resolve()
    split_path = Path(split_path).resolve()
    raw_manifest_path = Path(raw_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    if split_sha != SPLIT_SHA256:
        raise ValueError(f'Unexpected frozen split SHA256: {split_sha}')
    raw_manifest = load_json(raw_manifest_path)
    raw_sha = sha256_file(raw_manifest_path)
    scene_meta_path = data_root / 'train_scene' / 'scene_meta.json'
    scene_meta = load_json(scene_meta_path)
    verify_calibration(raw_manifest, scene_meta)
    train_keys = list(split['splits']['train'])
    query_keys = set(split['splits']['val']) | set(split['splits']['test'])
    if len(train_keys) != 552 or len(set(train_keys)) != len(train_keys) or set(train_keys) & query_keys:
        raise ValueError('Training split is not the frozen 552-frame disjoint subset')
    missing = [key for key in train_keys if key not in raw_manifest['frames']]
    if missing:
        raise ValueError(f'Raw manifest misses training frames: {missing[:3]}')
    cv2.setNumThreads(1)
    detector = cv2.SIFT_create(nfeatures=4096)
    all_descriptors = []
    all_uv = []
    all_world = []
    all_body = []
    all_raw_index = []
    all_reference_id = []
    reference_centers = []
    reference_world_body = []
    reference_camera_from_body = []
    references = []
    source_hashes = []
    reference_keys = []
    build_started = time.perf_counter()
    for reference_id, key in enumerate(train_keys):
        record = raw_manifest['frames'][key]
        scan_path = data_root / key
        image_path = resolve_image(raw_manifest_path, record)
        pose_path = data_root / 'train_scene' / 'train' / 'poses' / (Path(key).stem + '.txt')
        for path in (scan_path, image_path, pose_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        camera_from_body = np.asarray(record['T_camera_lidar'], dtype=np.float64)
        intrinsics = np.asarray(record['K'], dtype=np.float64)
        world_camera = read_camera_pose(pose_path)
        world_body = world_camera @ camera_from_body
        body_points, raw_point_indices = read_body_scan(scan_path)
        image, keypoints_xy, descriptors = extract_sift(image_path, max_features=4096)
        feature_rows, point_rows, _ = associate_keypoints(
            keypoints_xy, body_points, raw_point_indices, camera_from_body, intrinsics,
            (image.shape[1], image.shape[0]), MAP_KEYPOINT_RADIUS_PX)
        world_points = transform_points(world_body, body_points[point_rows])
        body_points_supported = body_points[point_rows]
        descriptors_supported = descriptors[feature_rows]
        uv_supported = keypoints_xy[feature_rows]
        reference_centers.append(world_body[:3, 3])
        reference_world_body.append(world_body)
        reference_camera_from_body.append(camera_from_body)
        reference_keys.append(key)
        all_descriptors.append(descriptors_supported.astype(np.float32, copy=False))
        all_uv.append(uv_supported.astype(np.float32, copy=False))
        all_world.append(world_points.astype(np.float64, copy=False))
        all_body.append(body_points_supported.astype(np.float64, copy=False))
        all_raw_index.append(raw_point_indices[point_rows].astype(np.int32, copy=False))
        all_reference_id.append(np.full(len(feature_rows), reference_id, dtype=np.int32))
        source_hashes.append({
            'scan': key,
            'scan_sha256': sha256_file(scan_path),
            'image': image_path.relative_to(data_root).as_posix(),
            'image_sha256': sha256_file(image_path),
            'camera_pose': str(pose_path.relative_to(data_root)),
            'camera_pose_sha256': sha256_file(pose_path),
            'sift_features': int(len(keypoints_xy)),
            'lidar_supported_features': int(len(feature_rows)),
        })
        references.append({
            'scan': key,
            'camera_center_world': world_body[:3, 3].tolist(),
            'sift_features': int(len(keypoints_xy)),
            'lidar_supported_features': int(len(feature_rows)),
        })
        if (reference_id + 1) % 50 == 0 or reference_id + 1 == len(train_keys):
            print(f'map {reference_id + 1}/{len(train_keys)} refs; supported={sum(len(x) for x in all_descriptors)}', flush=True)
    arrays = {
        'reference_keys': np.asarray(reference_keys, dtype=np.str_),
        'reference_centers_world': np.asarray(reference_centers, dtype=np.float64),
        'reference_world_body': np.asarray(reference_world_body, dtype=np.float64),
        'reference_camera_from_body': np.asarray(reference_camera_from_body, dtype=np.float64),
        'descriptors': np.concatenate(all_descriptors, axis=0) if all_descriptors else np.empty((0, 128), np.float32),
        'keypoints_xy': np.concatenate(all_uv, axis=0) if all_uv else np.empty((0, 2), np.float32),
        'world_points': np.concatenate(all_world, axis=0) if all_world else np.empty((0, 3), np.float64),
        'body_points': np.concatenate(all_body, axis=0) if all_body else np.empty((0, 3), np.float64),
        'raw_point_indices': np.concatenate(all_raw_index, axis=0) if all_raw_index else np.empty((0,), np.int32),
        'reference_ids': np.concatenate(all_reference_id, axis=0) if all_reference_id else np.empty((0,), np.int32),
    }
    map_path = output_dir / 'train_map.npz'
    np.savez(map_path, **arrays)
    source_aggregate = sha256_bytes(json_bytes(source_hashes))
    metadata = {
        'protocol': MAP_PROTOCOL,
        'split_sha256': split_sha,
        'raw_manifest_sha256': raw_sha,
        'scene_meta_sha256': sha256_file(scene_meta_path),
        'training_frames': len(train_keys),
        'training_keys': train_keys,
        'query_keys_excluded': len(query_keys),
        'map_feature_count': int(len(arrays['descriptors'])),
        'map_sha256': sha256_file(map_path),
        'source_hashes_sha256': source_aggregate,
        'source_hashes': source_hashes,
        'references': references,
        'fixed_geometry': {
            'lidar_frame': 'velodyne_sync body frame; no additional sensor-to-body transform',
            'training_pose_file': 'T_world_camera from train/poses; T_world_body=T_world_camera@T_camera_body',
            'camera_from_body': 'raw_manifest T_camera_lidar, validated against inverse scene_meta T_BC_camera_to_body',
            'image_intrinsics': 'raw_manifest K for unresized half-resolution Cam5 pixels; no pixel scaling',
            'scan_range_m': [1.0, 100.0],
            'keypoint_lidar_association_radius_px': MAP_KEYPOINT_RADIUS_PX,
            'z_buffer': 'closest positive camera depth at each rounded image pixel',
            'sift': {'nfeatures': 4096, 'opencv': cv2.__version__},
        },
        'build_elapsed_seconds': time.perf_counter() - build_started,
    }
    write_json(output_dir / 'train_map.json', metadata)
    print(json.dumps({
        'training_frames': len(train_keys), 'supported_features': metadata['map_feature_count'],
        'map_sha256': metadata['map_sha256'], 'source_hashes_sha256': source_aggregate,
        'build_elapsed_seconds': metadata['build_elapsed_seconds'],
    }, ensure_ascii=False), flush=True)


def se3_exp(delta):
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    v = delta[:3]
    w = delta[3:]
    theta = np.linalg.norm(w)
    rotation = Rotation.from_rotvec(w).as_matrix()
    if theta < 1e-8:
        wx = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        V = np.eye(3) + 0.5 * wx + (1.0 / 6.0) * wx @ wx
    else:
        wx = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        V = np.eye(3) + ((1.0 - math.cos(theta)) / theta ** 2) * wx + ((theta - math.sin(theta)) / theta ** 3) * (wx @ wx)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = V @ v
    return transform


def project_world_points(pose_world_body, camera_from_body, intrinsics, world_points):
    body_from_world = np.linalg.inv(np.asarray(pose_world_body, dtype=np.float64))
    body_points = transform_points(body_from_world, world_points)
    camera_points = transform_points(camera_from_body, body_points)
    return project_camera(camera_points, intrinsics)


def residuals_for_delta(normalized_delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv):
    scales = np.array([TRANSLATION_BOUND_M] * 3 + [ROTATION_BOUND_RAD] * 3, dtype=np.float64)
    pose = np.asarray(pose_world_body, dtype=np.float64) @ se3_exp(np.asarray(normalized_delta, dtype=np.float64) * scales)
    uv, valid = project_world_points(pose, camera_from_body, intrinsics, world_points)
    residual = np.full((len(world_points), 2), 1e6, dtype=np.float64)
    residual[valid] = uv[valid] - query_uv[valid]
    return residual


def normalized_pixel_jacobian(pose_world_body, camera_from_body, intrinsics, world_points, query_uv):
    step = 1e-5
    columns = []
    zero = np.zeros(6, dtype=np.float64)
    for axis in range(6):
        plus = zero.copy()
        minus = zero.copy()
        plus[axis] = step
        minus[axis] = -step
        r_plus = residuals_for_delta(plus, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
        r_minus = residuals_for_delta(minus, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
        columns.append(((r_plus - r_minus) / (2 * step)).reshape(-1))
    return np.column_stack(columns)


def analytic_normalized_pixel_jacobian(pose_world_body, camera_from_body, intrinsics, world_points):
    scales = np.array([TRANSLATION_BOUND_M] * 3 + [ROTATION_BOUND_RAD] * 3, dtype=np.float64)
    body_points = transform_points(np.linalg.inv(np.asarray(pose_world_body, dtype=np.float64)), world_points)
    rotation = np.asarray(camera_from_body, dtype=np.float64)[:3, :3]
    camera_points = transform_points(camera_from_body, body_points)
    projected = (np.asarray(intrinsics, dtype=np.float64) @ camera_points.T).T
    denominator = projected[:, 2]
    projection_jacobian = (intrinsics[None, :2, :] * denominator[:, None, None]
                           - projected[:, :2, None] * intrinsics[None, 2:3, :]) / denominator[:, None, None] ** 2
    motion_jacobian = np.empty((len(world_points), 3, 6), dtype=np.float64)
    motion_jacobian[:, :, :3] = -np.eye(3, dtype=np.float64)[None]
    p = body_points
    motion_jacobian[:, :, 3:] = np.stack([
        np.array([[0.0, -point[2], point[1]], [point[2], 0.0, -point[0]],
                  [-point[1], point[0], 0.0]], dtype=np.float64) for point in p
    ])
    camera_motion = np.einsum('ij,njk->nik', rotation, motion_jacobian)
    jacobian = np.einsum('nij,njk->nik', projection_jacobian, camera_motion)
    jacobian *= scales[None, None, :]
    return jacobian.reshape(-1, 6)


def huber_rho(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 1.0, 0.5 * value ** 2, value - 0.5)


def refinement_objective(normalized_delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv):
    delta = np.asarray(normalized_delta, dtype=np.float64)
    residual = residuals_for_delta(delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    normalized_norm = np.linalg.norm(residual, axis=1) / HUBER_SCALE_PX
    data_term = float(np.mean(huber_rho(normalized_norm))) if len(normalized_norm) else 0.0
    prior = float(np.sum(delta[:3] ** 2) + np.sum(delta[3:] ** 2))
    return data_term + prior


def refine_pose(pose_world_body, camera_from_body, intrinsics, world_points, query_uv):
    pose_world_body = np.asarray(pose_world_body, dtype=np.float64)
    camera_from_body = np.asarray(camera_from_body, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    query_uv = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)
    if len(world_points) != len(query_uv):
        raise ValueError('3D and 2D correspondence counts differ')
    if len(world_points) < MIN_CORRESPONDENCES:
        return pose_world_body.copy(), {'status': 'fallback', 'reason': 'fewer_than_6_correspondences', 'correspondences': len(world_points)}
    jacobian = normalized_pixel_jacobian(pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if len(singular_values) >= 6 and singular_values[-1] > 0 else float('inf')
    if not np.isfinite(condition) or condition > DEGENERACY_CONDITION_LIMIT:
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'geometric_degeneracy', 'correspondences': len(world_points),
            'condition_number': condition, 'jacobian_singular_values': singular_values.tolist(),
        }
    initial = np.zeros(6, dtype=np.float64)
    objective_before = refinement_objective(initial, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    result = minimize(
        refinement_objective, initial,
        args=(pose_world_body, camera_from_body, intrinsics, world_points, query_uv),
        method='L-BFGS-B', bounds=[(-1.0, 1.0)] * 6,
        options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20},
    )
    candidate_delta = np.asarray(result.x, dtype=np.float64)
    objective_after = float(refinement_objective(candidate_delta, pose_world_body, camera_from_body,
                                                 intrinsics, world_points, query_uv))
    if not np.isfinite(candidate_delta).all() or not np.isfinite(objective_after):
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'nonfinite_optimizer_output', 'correspondences': len(world_points),
            'condition_number': condition,
        }
    if not bool(result.success):
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'optimizer_failed', 'correspondences': len(world_points),
            'condition_number': condition, 'objective_before': objective_before,
            'objective_after': objective_after, 'iterations': int(result.nit),
            'optimizer_success': False, 'optimizer_status': int(result.status),
            'optimizer_message': str(result.message),
        }
    if objective_after >= objective_before - 1e-12:
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'objective_not_reduced', 'correspondences': len(world_points),
            'condition_number': condition, 'objective_before': objective_before,
            'objective_after': objective_after, 'iterations': int(result.nit),
            'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
        }
    scales = np.array([TRANSLATION_BOUND_M] * 3 + [ROTATION_BOUND_RAD] * 3, dtype=np.float64)
    refined = pose_world_body @ se3_exp(candidate_delta * scales)
    initial_residual = residuals_for_delta(initial, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    final_residual = residuals_for_delta(candidate_delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    return refined, {
        'status': 'refined', 'reason': None, 'correspondences': len(world_points),
        'condition_number': condition, 'jacobian_singular_values': singular_values.tolist(),
        'normalized_delta': candidate_delta.tolist(), 'objective_before': objective_before,
        'objective_after': objective_after, 'iterations': int(result.nit),
        'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
        'median_reprojection_before_px': float(np.median(np.linalg.norm(initial_residual, axis=1))),
        'median_reprojection_after_px': float(np.median(np.linalg.norm(final_residual, axis=1))),
    }


def mutual_ratio_matches(query_descriptors, reference_descriptors, ratio=RATIO_THRESHOLD):
    if len(query_descriptors) == 0 or len(reference_descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(np.asarray(query_descriptors, np.float32), np.asarray(reference_descriptors, np.float32), k=2)
    reverse = matcher.knnMatch(np.asarray(reference_descriptors, np.float32), np.asarray(query_descriptors, np.float32), k=2)
    forward_best = {}
    for pair in forward:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            forward_best[pair[0].queryIdx] = pair[0]
    reverse_best = {}
    for pair in reverse:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            reverse_best[pair[0].queryIdx] = pair[0]
    matches = []
    for query_index, match in forward_best.items():
        backward = reverse_best.get(match.trainIdx)
        if backward is not None and backward.trainIdx == query_index:
            matches.append((int(query_index), int(match.trainIdx), float(match.distance)))
    return matches


def choose_references(query_position, map_arrays):
    centers = map_arrays['reference_centers_world']
    distances = np.linalg.norm(centers - np.asarray(query_position, dtype=np.float64)[None, :], axis=1)
    order = np.argsort(distances, kind='stable')
    return [(int(index), float(distances[index])) for index in order[:MAX_REFERENCES]
            if distances[index] <= MAX_REFERENCE_DISTANCE_M]


def gather_correspondences(query_xy, query_descriptors, pose_world_body, camera_from_body,
                           intrinsics, map_arrays, image_size):
    width, height = map(int, image_size)
    references = choose_references(pose_world_body[:3, 3], map_arrays)
    if not references:
        return [], {'references': [], 'matched_before_filters': 0, 'after_projection_filter': 0}
    all_candidates = []
    map_reference_ids = map_arrays['reference_ids']
    for reference_id, distance in references:
        rows = np.flatnonzero(map_reference_ids == reference_id)
        if len(rows) < 2:
            continue
        reference_descriptors = map_arrays['descriptors'][rows]
        matches = mutual_ratio_matches(query_descriptors, reference_descriptors)
        for query_id, local_reference_id, descriptor_distance in matches:
            map_row = int(rows[local_reference_id])
            all_candidates.append((query_id, map_row, descriptor_distance, reference_id, distance))
    before_count = len(all_candidates)
    if not all_candidates:
        return [], {'references': [{'id': item[0], 'distance_m': item[1]} for item in references],
                    'matched_before_filters': before_count, 'after_projection_filter': 0}
    best_by_query = {}
    for candidate in all_candidates:
        query_id = candidate[0]
        if query_id not in best_by_query or (candidate[2], candidate[3], candidate[1]) < (best_by_query[query_id][2], best_by_query[query_id][3], best_by_query[query_id][1]):
            best_by_query[query_id] = candidate
    candidates = list(best_by_query.values())
    world_points = np.asarray([map_arrays['world_points'][item[1]] for item in candidates], dtype=np.float64)
    query_uv = np.asarray([query_xy[item[0]] for item in candidates], dtype=np.float64)
    projected_uv, positive = project_world_points(pose_world_body, camera_from_body, intrinsics, world_points)
    reprojection_error = np.linalg.norm(projected_uv - query_uv, axis=1)
    visible = positive & (projected_uv[:, 0] >= 0) & (projected_uv[:, 0] < width)
    visible &= (projected_uv[:, 1] >= 0) & (projected_uv[:, 1] < height)
    visible &= np.isfinite(reprojection_error) & (reprojection_error <= INITIAL_REPROJECTION_LIMIT_PX)
    candidate_rows = []
    for index in np.flatnonzero(visible):
        item = candidates[index]
        candidate_rows.append({
            'query_index': int(item[0]), 'map_row': int(item[1]),
            'descriptor_distance': float(item[2]), 'reference_id': int(item[3]),
            'reference_distance_m': float(item[4]), 'initial_reprojection_error_px': float(reprojection_error[index]),
            'query_uv': query_uv[index], 'world_xyz': world_points[index],
        })
    candidate_rows.sort(key=lambda row: (row['descriptor_distance'], row['initial_reprojection_error_px'],
                                         row['query_uv'][1], row['query_uv'][0], row['reference_id'], row['map_row']))
    counts = {}
    selected = []
    for row in candidate_rows:
        x, y = row['query_uv']
        cell_x = min(GRID_COLUMNS - 1, max(0, int(x * GRID_COLUMNS / width)))
        cell_y = min(GRID_ROWS - 1, max(0, int(y * GRID_ROWS / height)))
        cell = cell_y * GRID_COLUMNS + cell_x
        if counts.get(cell, 0) >= MAX_PER_GRID_CELL:
            continue
        selected.append(row)
        counts[cell] = counts.get(cell, 0) + 1
        if len(selected) >= MAX_CORRESPONDENCES:
            break
    return selected, {
        'references': [{'id': item[0], 'distance_m': item[1]} for item in references],
        'matched_before_filters': before_count,
        'after_projection_filter': len(candidate_rows),
        'after_grid_cap': len(selected),
    }


def load_map(map_path, map_metadata_path, train_keys, split_sha):
    metadata = load_json(map_metadata_path)
    if metadata['protocol'] != MAP_PROTOCOL or metadata['split_sha256'] != split_sha:
        raise ValueError('Training map protocol or split mismatch')
    if metadata['training_keys'] != list(train_keys):
        raise ValueError('Training map does not match the fixed training split')
    if sha256_file(map_path) != metadata['map_sha256']:
        raise ValueError('Training map SHA256 mismatch')
    archive = np.load(map_path, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files}
    n = len(arrays['descriptors'])
    if arrays['descriptors'].shape != (n, 128) or arrays['world_points'].shape != (n, 3):
        raise ValueError('Training map arrays have inconsistent dimensions')
    if arrays['body_points'].shape != (n, 3) or arrays['reference_ids'].shape != (n,):
        raise ValueError('Training map lacks body-frame source points or reference provenance')
    if arrays['raw_point_indices'].shape != (n,) or arrays['keypoints_xy'].shape != (n, 2):
        raise ValueError('Training map feature provenance has inconsistent dimensions')
    if arrays['reference_world_body'].shape != (len(train_keys), 4, 4):
        raise ValueError('Training map is missing reference training poses')
    if arrays['reference_camera_from_body'].shape != (len(train_keys), 4, 4):
        raise ValueError('Training map is missing per-reference camera calibration')
    if n and (arrays['reference_ids'].min() < 0 or arrays['reference_ids'].max() >= len(train_keys)):
        raise ValueError('Training map has an invalid reference identifier')
    if n:
        source_poses = arrays['reference_world_body'][arrays['reference_ids']]
        expected_world = np.einsum('nij,nj->ni', source_poses[:, :3, :3], arrays['body_points']) + source_poses[:, :3, 3]
        if not np.allclose(expected_world, arrays['world_points'], atol=1e-7, rtol=0):
            raise ValueError('Training map 3D points do not match their train LiDAR points and training poses')
    if not np.isfinite(arrays['world_points']).all() or not np.isfinite(arrays['descriptors']).all():
        raise ValueError('Training map contains nonfinite values')
    return arrays, metadata


def run_subset(data_root, split_path, raw_manifest_path, map_path, map_metadata_path,
               baseline_predictions_path, output_dir, subset):
    started = time.perf_counter()
    data_root = Path(data_root).resolve()
    split_path = Path(split_path).resolve()
    raw_manifest_path = Path(raw_manifest_path).resolve()
    baseline_predictions_path = Path(baseline_predictions_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    if split_sha != SPLIT_SHA256:
        raise ValueError(f'Unexpected frozen split SHA256: {split_sha}')
    if subset not in ('val', 'test'):
        raise ValueError('Subset must be val or test')
    expected_baseline_sha = FROZEN_PREDICTION_SHA256[subset]
    observed_baseline_sha = sha256_file(baseline_predictions_path)
    sidecar = baseline_predictions_path.with_suffix('.sha256')
    if observed_baseline_sha != expected_baseline_sha or sidecar.read_text(encoding='ascii').strip() != observed_baseline_sha:
        raise ValueError('Frozen relation prediction SHA256/sidecar mismatch')
    baseline = load_json(baseline_predictions_path)
    keys = list(split['splits'][subset])
    if baseline.get('protocol') != ONLINE_PROTOCOL or baseline.get('checkpoint_sha256') != CHECKPOINT_SHA256:
        raise ValueError('Frozen relation prediction protocol or checkpoint mismatch')
    if baseline.get('split_sha256') != split_sha or baseline.get('subset') != subset:
        raise ValueError('Frozen relation prediction split or subset mismatch')
    if [row['scan'] for row in baseline['predictions']] != keys or baseline.get('expected_frames') != len(keys):
        raise ValueError('Frozen relation prediction does not cover the fixed subset in order')
    raw_manifest = load_json(raw_manifest_path)
    raw_sha = sha256_file(raw_manifest_path)
    map_arrays, map_metadata = load_map(map_path, map_metadata_path, split['splits']['train'], split_sha)
    if map_metadata['raw_manifest_sha256'] != raw_sha:
        raise ValueError('Training map and online raw-manifest hashes differ')
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    detector = cv2.SIFT_create(nfeatures=4096)
    refined_rows = []
    detail_rows = []
    query_image_hashes = []
    for frame_index, row in enumerate(baseline['predictions']):
        frame_started = time.perf_counter()
        scan_key = row['scan']
        original_pose = row.get('T_world_body')
        if row.get('status') != 'ok' or original_pose is None:
            refined_rows.append({'scan': scan_key, 'status': row.get('status', 'missing'),
                                 'T_world_body': original_pose, 'seconds': time.perf_counter() - frame_started})
            detail_rows.append({'scan': scan_key, 'status': 'baseline_failure', 'reason': 'relation_prediction_not_ok',
                                'correspondences': 0, 'refined': False})
            continue
        if scan_key not in raw_manifest['frames']:
            raise ValueError(f'Raw manifest misses query image/calibration: {scan_key}')
        record = raw_manifest['frames'][scan_key]
        image_path = resolve_image(raw_manifest_path, record)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image, query_xy, query_descriptors = extract_sift(image_path, max_features=4096)
        query_image_sha = sha256_file(image_path)
        query_image_hashes.append({'scan': scan_key, 'image_sha256': query_image_sha})
        pose = np.asarray(original_pose, dtype=np.float64)
        camera_from_body = np.asarray(record['T_camera_lidar'], dtype=np.float64)
        intrinsics = np.asarray(record['K'], dtype=np.float64)
        selected, matching_summary = gather_correspondences(
            query_xy, query_descriptors, pose, camera_from_body, intrinsics, map_arrays,
            (image.shape[1], image.shape[0]))
        world_points = np.asarray([item['world_xyz'] for item in selected], dtype=np.float64).reshape(-1, 3)
        query_uv = np.asarray([item['query_uv'] for item in selected], dtype=np.float64).reshape(-1, 2)
        refined_pose, refinement = refine_pose(pose, camera_from_body, intrinsics, world_points, query_uv)
        if refinement['status'] == 'fallback':
            result_pose = original_pose
        else:
            result_pose = refined_pose.tolist()
        frame_seconds = time.perf_counter() - frame_started
        refined_rows.append({'scan': scan_key, 'status': 'ok', 'T_world_body': result_pose, 'seconds': frame_seconds})
        detail_rows.append({
            'scan': scan_key, 'status': refinement['status'], 'reason': refinement.get('reason'),
            'refined': refinement['status'] == 'refined', 'query_image_sha256': query_image_sha,
            'query_sift_features': int(len(query_xy)), 'query_sift_descriptors': int(len(query_descriptors)),
            **matching_summary, **refinement, 'seconds': frame_seconds,
        })
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(keys):
            print(f'{subset} {frame_index + 1}/{len(keys)}; matched={sum(x.get("correspondences", 0) for x in detail_rows)}', flush=True)
    elapsed = time.perf_counter() - started
    refined_predictions = {
        'protocol': ONLINE_PROTOCOL,
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'split_sha256': split_sha,
        'training_split_sha256': split_sha,
        'subset': subset,
        'expected_frames': len(keys),
        'parent_prediction_sha256': observed_baseline_sha,
        'training_map_sha256': map_metadata['map_sha256'],
        'elapsed_seconds': elapsed,
        'predictions': refined_rows,
    }
    prediction_path = output_dir / 'predictions.json'
    write_json(prediction_path, refined_predictions)
    prediction_sha = sha256_file(prediction_path)
    prediction_path.with_suffix('.sha256').write_text(prediction_sha + '\n', encoding='ascii')
    details = {
        'protocol': 'camera_reprojection_diagnostics_v1', 'subset': subset,
        'frozen_relation_predictions_sha256': observed_baseline_sha,
        'relation_checkpoint_sha256': CHECKPOINT_SHA256,
        'split_sha256': split_sha,
        'raw_manifest_sha256': sha256_file(raw_manifest_path),
        'training_map_sha256': map_metadata['map_sha256'],
        'training_map_metadata_sha256': sha256_file(map_metadata_path),
        'refined_predictions_sha256': prediction_sha,
        'elapsed_seconds': elapsed,
        'frames': len(keys),
        'query_images': query_image_hashes,
        'fixed_parameters': {
            'sift_nfeatures': 4096, 'mutual_lowe_ratio': RATIO_THRESHOLD,
            'max_reference_distance_m': MAX_REFERENCE_DISTANCE_M, 'max_references': MAX_REFERENCES,
            'initial_reprojection_limit_px': INITIAL_REPROJECTION_LIMIT_PX,
            'grid': [GRID_COLUMNS, GRID_ROWS], 'max_per_cell': MAX_PER_GRID_CELL,
            'max_correspondences': MAX_CORRESPONDENCES, 'min_correspondences': MIN_CORRESPONDENCES,
            'optimizer_iterations': MAX_ITERATIONS, 'translation_bound_m_per_axis': TRANSLATION_BOUND_M,
            'rotation_bound_deg_per_axis': 1.0, 'huber_scale_px': HUBER_SCALE_PX,
            'degeneracy_condition_limit': DEGENERACY_CONDITION_LIMIT,
        },
        'query_ground_truth_access': 'none in online runner; GT evaluator invoked after this prediction SHA was frozen',
        'rows': detail_rows,
    }
    write_json(output_dir / 'details.json', details)
    print(json.dumps({
        'subset': subset, 'frames': len(keys), 'baseline_predictions_sha256': observed_baseline_sha,
        'refined_predictions_sha256': prediction_sha,
        'refined_frames': sum(row['refined'] for row in detail_rows),
        'fallback_frames': sum(row['status'] == 'fallback' for row in detail_rows),
        'elapsed_seconds': elapsed,
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    build = subparsers.add_parser('build-map')
    build.add_argument('--data-root', type=Path, required=True)
    build.add_argument('--split', type=Path, required=True)
    build.add_argument('--raw-manifest', type=Path, required=True)
    build.add_argument('--out-dir', type=Path, required=True)
    run = subparsers.add_parser('run')
    run.add_argument('--data-root', type=Path, required=True)
    run.add_argument('--split', type=Path, required=True)
    run.add_argument('--raw-manifest', type=Path, required=True)
    run.add_argument('--map', type=Path, required=True)
    run.add_argument('--map-metadata', type=Path, required=True)
    run.add_argument('--baseline-predictions', type=Path, required=True)
    run.add_argument('--out-dir', type=Path, required=True)
    run.add_argument('--subset', choices=('val', 'test'), required=True)
    args = parser.parse_args()
    if args.command == 'build-map':
        build_train_map(args.data_root, args.split, args.raw_manifest, args.out_dir)
    else:
        run_subset(args.data_root, args.split, args.raw_manifest, args.map, args.map_metadata,
                   args.baseline_predictions, args.out_dir, args.subset)


if __name__ == '__main__':
    main()
