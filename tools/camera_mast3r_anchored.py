import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize

import camera_reprojection as camera


MAP_PROTOCOL = 'train_lidar_pixel_anchor_map_v1'
ONLINE_PROTOCOL = 'relation_mast3r_lidar_anchor_refinement_v1'
MAST3R_COMMIT = 'f5209afc300cec36239a7ac992263f36847bbba0'
DUST3R_COMMIT = '3cc8c88c413bb9e34c41db0e0eef99c2ee010b12'
CROCO_COMMIT = 'd7de0705845239092414480bd829228723bf20de'
WEIGHT_BYTES = 2754910614
MODEL_NAME = 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
WEIGHT_URL = 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
SPLIT_SHA256 = camera.SPLIT_SHA256
CHECKPOINT_SHA256 = camera.CHECKPOINT_SHA256
FROZEN_PREDICTION_SHA256 = camera.FROZEN_PREDICTION_SHA256
MAX_REFERENCE_DISTANCE_M = 20.0
MAX_REFERENCES = 5
COARSE_SUBSAMPLE = 8
COARSE_BORDER_PX = 3
CROP_OVERLAP = 0.5
MIN_COARSE_MATCHES_PER_CROP = 10
FINE_PIXEL_TOLERANCE = 5
CONFIDENCE_THRESHOLD = 1.001
FAST_NN_BLOCK_SIZE = 2 ** 13
FINE_MAX_BATCH_SIZE = 48
INITIAL_REPROJECTION_LIMIT_PX = 12.0
GRID_COLUMNS = 8
GRID_ROWS = 6
MAX_PER_GRID_CELL = 3
MAX_CORRESPONDENCES = 128
MIN_CORRESPONDENCES = 6
TRANSLATION_BOUND_M = camera.TRANSLATION_BOUND_M
ROTATION_BOUND_RAD = camera.ROTATION_BOUND_RAD
HUBER_SCALE_PX = camera.HUBER_SCALE_PX
MAX_ITERATIONS = camera.MAX_ITERATIONS
DEGENERACY_CONDITION_LIMIT = camera.DEGENERACY_CONDITION_LIMIT


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def project_lidar_anchors(body_points, raw_point_indices, world_body, camera_from_body,
                          intrinsics, image_size):
    width, height = map(int, image_size)
    body_points = np.asarray(body_points, dtype=np.float64).reshape(-1, 3)
    raw_point_indices = np.asarray(raw_point_indices, dtype=np.int32).reshape(-1)
    if len(body_points) != len(raw_point_indices):
        raise ValueError('LiDAR point and source-index counts differ')
    camera_points = camera.transform_points(camera_from_body, body_points)
    uv, positive = camera.project_camera(camera_points, intrinsics)
    continuous_in_image = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width)
    continuous_in_image &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    rows = np.flatnonzero(continuous_in_image)
    if len(rows) == 0:
        return {
            'pixel_xy': np.empty((0, 2), dtype=np.int32),
            'projection_uv': np.empty((0, 2), dtype=np.float64),
            'body_points': np.empty((0, 3), dtype=np.float64),
            'world_points': np.empty((0, 3), dtype=np.float64),
            'raw_point_indices': np.empty((0,), dtype=np.int32),
            'camera_depth': np.empty((0,), dtype=np.float32),
            'quantization_error_px': np.empty((0, 2), dtype=np.float64),
            'counts': {
                'input_lidar_points': int(len(body_points)),
                'positive_finite_projection': int(positive.sum()),
                'continuous_in_image': 0,
                'rounded_in_image': 0,
                'visible_pixels': 0,
                'occluded_or_duplicate_pixels': 0,
            },
        }
    rounded = np.floor(uv[rows] + 0.5).astype(np.int32)
    quantization_error = uv[rows] - rounded
    if len(quantization_error) and np.max(np.abs(quantization_error)) > 0.5 + 1e-10:
        raise ValueError('Pixel rounding exceeded half a pixel')
    in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
    in_bounds &= (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    rounded_rows = rows[in_bounds]
    rounded = rounded[in_bounds]
    quantization_error = quantization_error[in_bounds]
    flat = rounded[:, 1].astype(np.int64) * width + rounded[:, 0]
    depth = camera_points[rounded_rows, 2]
    order = np.lexsort((raw_point_indices[rounded_rows], depth, flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    selected_positions = order[first]
    selected_rows = rounded_rows[selected_positions]
    selected_pixels = rounded[selected_positions]
    selected_errors = quantization_error[selected_positions]
    world_points = camera.transform_points(world_body, body_points[selected_rows])
    result = {
        'pixel_xy': selected_pixels.astype(np.int32, copy=False),
        'projection_uv': uv[selected_rows].astype(np.float64, copy=False),
        'body_points': body_points[selected_rows].astype(np.float64, copy=False),
        'world_points': world_points.astype(np.float64, copy=False),
        'raw_point_indices': raw_point_indices[selected_rows].astype(np.int32, copy=False),
        'camera_depth': depth[selected_positions].astype(np.float32, copy=False),
        'quantization_error_px': selected_errors.astype(np.float64, copy=False),
        'counts': {
            'input_lidar_points': int(len(body_points)),
            'positive_finite_projection': int(positive.sum()),
            'continuous_in_image': int(len(rows)),
            'rounded_in_image': int(len(rounded_rows)),
            'visible_pixels': int(len(selected_rows)),
            'occluded_or_duplicate_pixels': int(len(rounded_rows) - len(selected_rows)),
        },
    }
    if len(result['pixel_xy']) > 1:
        flat_selected = result['pixel_xy'][:, 1].astype(np.int64) * width + result['pixel_xy'][:, 0]
        if np.any(flat_selected[1:] <= flat_selected[:-1]):
            sort_order = np.argsort(flat_selected, kind='stable')
            for name in ('pixel_xy', 'projection_uv', 'body_points', 'world_points',
                         'raw_point_indices', 'camera_depth', 'quantization_error_px'):
                result[name] = result[name][sort_order]
    return result


def build_train_map(data_root, split_path, raw_manifest_path, output_dir):
    data_root = Path(data_root).resolve()
    split_path = Path(split_path).resolve()
    raw_manifest_path = Path(raw_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    if split_sha != SPLIT_SHA256:
        raise ValueError(f'Unexpected frozen split SHA256: {split_sha}')
    raw_manifest = load_json(raw_manifest_path)
    raw_sha = sha256_file(raw_manifest_path)
    scene_meta_path = data_root / 'train_scene' / 'scene_meta.json'
    scene_meta = load_json(scene_meta_path)
    camera.verify_calibration(raw_manifest, scene_meta)
    train_keys = list(split['splits']['train'])
    query_keys = set(split['splits']['val']) | set(split['splits']['test'])
    if len(train_keys) != 552 or len(set(train_keys)) != len(train_keys) or set(train_keys) & query_keys:
        raise ValueError('Training split is not the frozen 552-frame disjoint subset')
    if len(split['splits']['val']) != 40 or len(split['splits']['test']) != 313:
        raise ValueError('The frozen validation or test split size changed')
    missing = [key for key in train_keys if key not in raw_manifest['frames']]
    if missing:
        raise ValueError(f'Raw manifest misses training frames: {missing[:3]}')
    pixel_offsets = [0]
    all_pixel_xy = []
    all_projection_uv = []
    all_body_points = []
    all_world_points = []
    all_raw_point_indices = []
    all_reference_ids = []
    all_camera_depth = []
    all_quantization_error = []
    reference_centers = []
    reference_world_body = []
    reference_camera_from_body = []
    reference_intrinsics = []
    reference_sizes = []
    source_hashes = []
    reference_counts = []
    started = time.perf_counter()
    for reference_id, key in enumerate(train_keys):
        record = raw_manifest['frames'][key]
        scan_path = data_root / key
        image_path = camera.resolve_image(raw_manifest_path, record)
        pose_path = data_root / 'train_scene' / 'train' / 'poses' / (Path(key).stem + '.txt')
        for path in (scan_path, image_path, pose_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with Image.open(image_path) as image_file:
            width, height = image_file.size
            if image_file.mode not in ('RGB', 'RGBA', 'L'):
                image_file.convert('RGB')
        camera_from_body = np.asarray(record['T_camera_lidar'], dtype=np.float64)
        intrinsics = np.asarray(record['K'], dtype=np.float64)
        world_camera = camera.read_camera_pose(pose_path)
        world_body = world_camera @ np.linalg.inv(camera_from_body)
        body_points, raw_point_indices = camera.read_body_scan(scan_path)
        anchors = project_lidar_anchors(
            body_points, raw_point_indices, world_body, camera_from_body, intrinsics, (width, height))
        for name, values in (
                ('pixel_xy', anchors['pixel_xy']),
                ('projection_uv', anchors['projection_uv']),
                ('body_points', anchors['body_points']),
                ('world_points', anchors['world_points']),
                ('raw_point_indices', anchors['raw_point_indices']),
                ('camera_depth', anchors['camera_depth']),
                ('quantization_error_px', anchors['quantization_error_px'])):
            if len(values):
                {'pixel_xy': all_pixel_xy,
                 'projection_uv': all_projection_uv,
                 'body_points': all_body_points,
                 'world_points': all_world_points,
                 'raw_point_indices': all_raw_point_indices,
                 'camera_depth': all_camera_depth,
                 'quantization_error_px': all_quantization_error}[name].append(values)
        all_reference_ids.append(np.full(len(anchors['pixel_xy']), reference_id, dtype=np.int16))
        pixel_offsets.append(pixel_offsets[-1] + len(anchors['pixel_xy']))
        reference_centers.append(world_body[:3, 3])
        reference_world_body.append(world_body)
        reference_camera_from_body.append(camera_from_body)
        reference_intrinsics.append(intrinsics)
        reference_sizes.append([width, height])
        source_hashes.append({
            'scan': key,
            'scan_sha256': sha256_file(scan_path),
            'image': image_path.relative_to(data_root).as_posix(),
            'image_sha256': sha256_file(image_path),
            'training_pose': pose_path.relative_to(data_root).as_posix(),
            'training_pose_sha256': sha256_file(pose_path),
        })
        reference_counts.append({'scan': key, **anchors['counts']})
        if (reference_id + 1) % 50 == 0 or reference_id + 1 == len(train_keys):
            print(f'train_map {reference_id + 1}/{len(train_keys)}; anchors={pixel_offsets[-1]}', flush=True)

    def concatenate(values, shape, dtype):
        return np.concatenate(values, axis=0).astype(dtype, copy=False) if values else np.empty(shape, dtype=dtype)

    arrays = {
        'reference_keys': np.asarray(train_keys, dtype=np.str_),
        'pixel_offsets': np.asarray(pixel_offsets, dtype=np.int64),
        'pixel_xy': concatenate(all_pixel_xy, (0, 2), np.int32),
        'projection_uv': concatenate(all_projection_uv, (0, 2), np.float64),
        'body_points': concatenate(all_body_points, (0, 3), np.float64),
        'world_points': concatenate(all_world_points, (0, 3), np.float64),
        'raw_point_indices': concatenate(all_raw_point_indices, (0,), np.int32),
        'reference_ids': concatenate(all_reference_ids, (0,), np.int16),
        'camera_depth': concatenate(all_camera_depth, (0,), np.float32),
        'quantization_error_px': concatenate(all_quantization_error, (0, 2), np.float64),
        'reference_centers_world': np.asarray(reference_centers, dtype=np.float64),
        'reference_world_body': np.asarray(reference_world_body, dtype=np.float64),
        'reference_camera_from_body': np.asarray(reference_camera_from_body, dtype=np.float64),
        'reference_intrinsics': np.asarray(reference_intrinsics, dtype=np.float64),
        'reference_sizes': np.asarray(reference_sizes, dtype=np.int32),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / 'lidar_pixel_anchor_map.npz'
    np.savez_compressed(map_path, **arrays)
    source_hash_sha = json_sha256(source_hashes)
    metadata = {
        'protocol': MAP_PROTOCOL,
        'split_sha256': split_sha,
        'raw_manifest_sha256': raw_sha,
        'scene_meta_sha256': sha256_file(scene_meta_path),
        'training_keys': train_keys,
        'query_keys_excluded': {'val': len(split['splits']['val']), 'test': len(split['splits']['test'])},
        'anchor_count': int(len(arrays['world_points'])),
        'map_sha256': sha256_file(map_path),
        'source_hashes_sha256': source_hash_sha,
        'source_hashes': source_hashes,
        'reference_counts': reference_counts,
        'fixed_geometry': {
            'lidar_frame': 'velodyne_sync body frame; no additional sensor-to-body transform',
            'camera_projection': 'raw_manifest T_camera_lidar and original Cam5 K',
            'training_world_transform': 'T_world_camera from train/poses times inverse(T_camera_lidar)',
            'pixel_quantization': 'nearest integer with floor(u+0.5); each-axis absolute error <= 0.5 px',
            'occlusion': 'closest positive camera depth per integer image pixel; raw point index breaks exact depth ties',
            'stored_identity': 'reference scan key plus original raw LiDAR point index',
        },
        'build_elapsed_seconds': time.perf_counter() - started,
    }
    write_json(output_dir / 'lidar_pixel_anchor_map.json', metadata)
    print(json.dumps({
        'protocol': MAP_PROTOCOL,
        'training_frames': len(train_keys),
        'anchor_count': metadata['anchor_count'],
        'map_sha256': metadata['map_sha256'],
        'source_hashes_sha256': source_hash_sha,
        'build_elapsed_seconds': metadata['build_elapsed_seconds'],
    }, ensure_ascii=False), flush=True)


def load_map(map_path, metadata_path, data_root, train_keys, split_sha, raw_manifest_sha):
    metadata = load_json(metadata_path)
    if metadata['protocol'] != MAP_PROTOCOL or metadata['split_sha256'] != split_sha:
        raise ValueError('Training map protocol or split mismatch')
    if metadata['raw_manifest_sha256'] != raw_manifest_sha:
        raise ValueError('Training map raw-manifest SHA256 mismatch')
    if metadata['training_keys'] != list(train_keys):
        raise ValueError('Training map does not match the frozen training split')
    if sha256_file(map_path) != metadata['map_sha256']:
        raise ValueError('Training map SHA256 mismatch')
    if json_sha256(metadata['source_hashes']) != metadata['source_hashes_sha256']:
        raise ValueError('Training source-hash aggregate mismatch')
    archive = np.load(map_path, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files}
    n = len(arrays['world_points'])
    ref_n = len(train_keys)
    shapes = {
        'pixel_xy': (n, 2), 'projection_uv': (n, 2), 'body_points': (n, 3),
        'raw_point_indices': (n,), 'reference_ids': (n,), 'camera_depth': (n,),
        'quantization_error_px': (n, 2), 'pixel_offsets': (ref_n + 1,),
        'reference_centers_world': (ref_n, 3), 'reference_world_body': (ref_n, 4, 4),
        'reference_camera_from_body': (ref_n, 4, 4), 'reference_intrinsics': (ref_n, 3, 3),
        'reference_sizes': (ref_n, 2),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f'Training map field {name} has shape {arrays[name].shape}, expected {shape}')
    if arrays['reference_keys'].tolist() != list(train_keys):
        raise ValueError('Training map key order changed')
    offsets = arrays['pixel_offsets']
    if offsets[0] != 0 or offsets[-1] != n or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('Training map offsets are invalid')
    if n and (arrays['reference_ids'].min() < 0 or arrays['reference_ids'].max() >= ref_n):
        raise ValueError('Training map contains an invalid reference ID')
    expected_ids = np.repeat(np.arange(ref_n, dtype=np.int16), np.diff(offsets))
    if not np.array_equal(expected_ids, arrays['reference_ids']):
        raise ValueError('Training map anchors are not grouped by reference frame')
    if n:
        transforms = arrays['reference_world_body'][arrays['reference_ids']]
        expected_world = np.einsum('nij,nj->ni', transforms[:, :3, :3], arrays['body_points'])
        expected_world += transforms[:, :3, 3]
    else:
        expected_world = arrays['world_points']
    if not np.allclose(expected_world, arrays['world_points'], atol=1e-7, rtol=0):
        raise ValueError('Training map world points do not match source body points and training poses')
    if n and np.max(np.abs(arrays['quantization_error_px'])) > 0.5 + 1e-10:
        raise ValueError('Training map pixel quantization exceeds 0.5 px')
    for ref_id in range(ref_n):
        start, end = offsets[ref_id:ref_id + 2]
        width, height = arrays['reference_sizes'][ref_id]
        pixels = arrays['pixel_xy'][start:end]
        if len(pixels):
            flat = pixels[:, 1].astype(np.int64) * int(width) + pixels[:, 0]
            if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= width):
                raise ValueError(f'Out-of-bounds anchor x coordinate in reference {ref_id}')
            if np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= height):
                raise ValueError(f'Out-of-bounds anchor y coordinate in reference {ref_id}')
            if len(np.unique(flat)) != len(flat):
                raise ValueError(f'Duplicate z-buffered pixels in reference {ref_id}')
    if not np.isfinite(arrays['world_points']).all() or not np.isfinite(arrays['projection_uv']).all():
        raise ValueError('Training map contains nonfinite coordinates')
    source_root = Path(data_root).resolve()
    source_failures = []
    for source in metadata['source_hashes']:
        source_paths = (
            (source['scan'], source['scan_sha256']),
            (source['image'], source['image_sha256']),
            (source['training_pose'], source['training_pose_sha256']),
        )
        for relative, expected in source_paths:
            path = source_root / Path(relative)
            if not path.is_file() or sha256_file(path) != expected:
                source_failures.append(str(path))
    if source_failures:
        raise ValueError(f'Training input hash verification failed for {len(source_failures)} file(s): {source_failures[:3]}')
    return arrays, metadata


def choose_references(query_position, map_arrays):
    centers = map_arrays['reference_centers_world']
    distances = np.linalg.norm(centers - np.asarray(query_position, dtype=np.float64)[None], axis=1)
    eligible = np.flatnonzero(np.isfinite(distances) & (distances <= MAX_REFERENCE_DISTANCE_M))
    order = np.lexsort((eligible, distances[eligible]))
    selected = eligible[order[:MAX_REFERENCES]]
    return [(int(ref_id), float(distances[ref_id])) for ref_id in selected]


def apply_pixel_transform(matrix, xy, pixel_centers=False):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return xy.copy()
    offset = 0.5 if pixel_centers else 0.0
    points = np.column_stack((xy + offset, np.ones(len(xy), dtype=np.float64)))
    mapped = (np.asarray(matrix, dtype=np.float64) @ points.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped - offset


def geometry_filter_then_deduplicate(candidates, pose_world_body, camera_from_body,
                                     intrinsics, image_size):
    width, height = map(int, image_size)
    geometry_valid = []
    rejected = {
        'nonfinite': 0,
        'query_pixel_outside': 0,
        'behind_camera': 0,
        'projected_pixel_outside': 0,
        'initial_reprojection_over_12px': 0,
        'reference_anchor_identity_mismatch': 0,
    }
    for candidate in candidates:
        query_uv = np.asarray(candidate['query_uv'], dtype=np.float64).reshape(2)
        world_xyz = np.asarray(candidate['world_xyz'], dtype=np.float64).reshape(3)
        if not np.isfinite(query_uv).all() or not np.isfinite(world_xyz).all():
            rejected['nonfinite'] += 1
            continue
        query_pixel = np.floor(query_uv + 0.5).astype(np.int32)
        if np.max(np.abs(query_uv - query_pixel)) > 0.5 + 1e-10:
            rejected['nonfinite'] += 1
            continue
        if not (0 <= query_pixel[0] < width and 0 <= query_pixel[1] < height):
            rejected['query_pixel_outside'] += 1
            continue
        projected, positive = camera.project_world_points(
            pose_world_body, camera_from_body, intrinsics, world_xyz[None])
        if not positive[0] or not np.isfinite(projected[0]).all():
            rejected['behind_camera'] += 1
            continue
        if not (0 <= projected[0, 0] < width and 0 <= projected[0, 1] < height):
            rejected['projected_pixel_outside'] += 1
            continue
        error = float(np.linalg.norm(projected[0] - query_uv))
        if not np.isfinite(error):
            rejected['nonfinite'] += 1
            continue
        if error > INITIAL_REPROJECTION_LIMIT_PX:
            rejected['initial_reprojection_over_12px'] += 1
            continue
        row = dict(candidate)
        row['query_pixel_xy'] = query_pixel.tolist()
        row['initial_reprojection_error_px'] = error
        geometry_valid.append(row)
    ranked = sorted(geometry_valid, key=lambda row: (
        -float(row['confidence']),
        float(row['initial_reprojection_error_px']),
        int(row['reference_id']),
        int(row['raw_point_index']),
        int(row['reference_pixel_xy'][1]),
        int(row['reference_pixel_xy'][0]),
        float(row['query_uv'][1]),
        float(row['query_uv'][0]),
    ))
    unique = []
    seen_query_pixels = set()
    seen_landmarks = set()
    duplicate_query_pixels = 0
    duplicate_landmarks = 0
    for row in ranked:
        query_pixel = tuple(row['query_pixel_xy'])
        landmark = (str(row['train_scan']), int(row['raw_point_index']))
        if query_pixel in seen_query_pixels:
            duplicate_query_pixels += 1
            continue
        if landmark in seen_landmarks:
            duplicate_landmarks += 1
            continue
        seen_query_pixels.add(query_pixel)
        seen_landmarks.add(landmark)
        unique.append(row)
    return unique, {
        'raw_candidates': int(len(candidates)),
        'after_geometry': int(len(geometry_valid)),
        'after_unique_query_pixel_and_landmark': int(len(unique)),
        'geometry_rejections': rejected,
        'duplicate_query_pixel': int(duplicate_query_pixels),
        'duplicate_landmark': int(duplicate_landmarks),
        'deduplicated': int(len(geometry_valid) - len(unique)),
    }


def apply_grid_cap(candidates, image_size):
    width, height = map(int, image_size)
    ranked = sorted(candidates, key=lambda row: (
        -float(row['confidence']),
        float(row['initial_reprojection_error_px']),
        int(row['query_pixel_xy'][1]),
        int(row['query_pixel_xy'][0]),
        int(row['reference_id']),
        int(row['raw_point_index']),
    ))
    counts = {}
    selected = []
    rejected_grid = 0
    for row in ranked:
        x, y = row['query_pixel_xy']
        cell_x = min(GRID_COLUMNS - 1, max(0, int(x * GRID_COLUMNS / width)))
        cell_y = min(GRID_ROWS - 1, max(0, int(y * GRID_ROWS / height)))
        cell_id = cell_y * GRID_COLUMNS + cell_x
        if counts.get(cell_id, 0) >= MAX_PER_GRID_CELL:
            rejected_grid += 1
            continue
        selected.append(row)
        counts[cell_id] = counts.get(cell_id, 0) + 1
        if len(selected) >= MAX_CORRESPONDENCES:
            break
    return selected, {
        'after_grid_and_total_cap': int(len(selected)),
        'grid_cells_occupied': int(len(counts)),
        'grid_cap_rejections': int(rejected_grid),
    }


def summed_refinement_objective(normalized_delta, pose_world_body, camera_from_body,
                                intrinsics, world_points, query_uv):
    delta = np.asarray(normalized_delta, dtype=np.float64)
    residual = camera.residuals_for_delta(
        delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    normalized_norm = np.linalg.norm(residual, axis=1) / HUBER_SCALE_PX
    visual_sum = float(np.sum(camera.huber_rho(normalized_norm))) if len(normalized_norm) else 0.0
    prior = float(np.sum(delta[:3] ** 2) + np.sum(delta[3:] ** 2))
    return visual_sum + prior


def refine_pose_sum(pose_world_body, camera_from_body, intrinsics, world_points, query_uv):
    pose_world_body = np.asarray(pose_world_body, dtype=np.float64)
    camera_from_body = np.asarray(camera_from_body, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    query_uv = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)
    if len(world_points) != len(query_uv):
        raise ValueError('3D and 2D correspondence counts differ')
    if len(world_points) < MIN_CORRESPONDENCES:
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'fewer_than_6_correspondences',
            'correspondences': int(len(world_points)),
        }
    jacobian = camera.normalized_pixel_jacobian(
        pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if len(singular_values) >= 6 and singular_values[-1] > 0 else float('inf')
    if not np.isfinite(condition) or condition > DEGENERACY_CONDITION_LIMIT:
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'geometric_degeneracy',
            'correspondences': int(len(world_points)), 'condition_number': condition,
            'jacobian_singular_values': singular_values.tolist(),
        }
    initial = np.zeros(6, dtype=np.float64)
    objective_before = summed_refinement_objective(
        initial, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    result = minimize(
        summed_refinement_objective,
        initial,
        args=(pose_world_body, camera_from_body, intrinsics, world_points, query_uv),
        method='L-BFGS-B',
        bounds=[(-1.0, 1.0)] * 6,
        options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20},
    )
    candidate_delta = np.asarray(result.x, dtype=np.float64)
    objective_after = float(summed_refinement_objective(
        candidate_delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv))
    if not np.isfinite(candidate_delta).all() or not np.isfinite(objective_after):
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'nonfinite_optimizer_output',
            'correspondences': int(len(world_points)), 'condition_number': condition,
        }
    if not bool(result.success):
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'optimizer_failed',
            'correspondences': int(len(world_points)), 'condition_number': condition,
            'objective_before': objective_before, 'objective_after': objective_after,
            'iterations': int(result.nit), 'optimizer_success': False,
            'optimizer_status': int(result.status), 'optimizer_message': str(result.message),
        }
    if objective_after >= objective_before - 1e-12:
        return pose_world_body.copy(), {
            'status': 'fallback', 'reason': 'objective_not_reduced',
            'correspondences': int(len(world_points)), 'condition_number': condition,
            'objective_before': objective_before, 'objective_after': objective_after,
            'iterations': int(result.nit), 'optimizer_success': bool(result.success),
            'optimizer_message': str(result.message),
        }
    scales = np.array([TRANSLATION_BOUND_M] * 3 + [ROTATION_BOUND_RAD] * 3, dtype=np.float64)
    refined = pose_world_body @ camera.se3_exp(candidate_delta * scales)
    final_residual = camera.residuals_for_delta(
        candidate_delta, pose_world_body, camera_from_body, intrinsics, world_points, query_uv)
    return refined, {
        'status': 'refined', 'reason': None,
        'correspondences': int(len(world_points)), 'condition_number': condition,
        'jacobian_singular_values': singular_values.tolist(),
        'normalized_delta': candidate_delta.tolist(), 'objective_before': objective_before,
        'objective_after': objective_after, 'iterations': int(result.nit),
        'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
        'median_reprojection_before_px': float(np.median(np.linalg.norm(
            camera.residuals_for_delta(initial, pose_world_body, camera_from_body,
                                       intrinsics, world_points, query_uv), axis=1))),
        'median_reprojection_after_px': float(np.median(np.linalg.norm(final_residual, axis=1))),
    }


class Mast3rCoarseToFineMatcher:
    def __init__(self, official_root, checkpoint_path, device='cuda'):
        self.official_root = Path(official_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if self.checkpoint_path.stat().st_size != WEIGHT_BYTES:
            raise ValueError(f'Official checkpoint byte size differs: {self.checkpoint_path.stat().st_size}')
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        git_head = subprocess.run(
            ['git', '-C', str(self.official_root), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True).stdout.strip()
        if git_head != MAST3R_COMMIT:
            raise ValueError(f'Unexpected MASt3R source commit: {git_head}')
        submodules = subprocess.run(
            ['git', '-C', str(self.official_root), 'submodule', 'status'],
            check=True, capture_output=True, text=True).stdout.splitlines()
        submodule_revs = {line.split()[1]: line.split()[0].lstrip('-+') for line in submodules}
        if submodule_revs.get('dust3r') != DUST3R_COMMIT:
            raise ValueError(f'Unexpected dust3r submodule revision: {submodule_revs}')
        sys.path.insert(0, str(self.official_root))
        from mast3r.model import AsymmetricMASt3R
        import mast3r.utils.path_to_dust3r
        from mast3r.fast_nn import fast_reciprocal_NNs
        from mast3r.utils.coarse_to_fine import select_pairs_of_crops
        from dust3r.inference import inference
        from dust3r.utils.image import ImgNorm
        dust3r_visloc_utils_path = self.official_root / 'dust3r' / 'dust3r_visloc' / 'datasets' / 'utils.py'
        helper_spec = importlib.util.spec_from_file_location('mast3r_anchored_visloc_utils', dust3r_visloc_utils_path)
        helper_module = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper_module)
        self.fast_reciprocal_NNs = fast_reciprocal_NNs
        self.select_pairs_of_crops = select_pairs_of_crops
        self.inference = inference
        self.ImgNorm = ImgNorm
        self.get_HW_resolution = helper_module.get_HW_resolution
        self.get_resize_function = helper_module.get_resize_function
        self.device = device
        self.model = AsymmetricMASt3R.from_pretrained(str(self.checkpoint_path)).to(device).eval()
        self.maxdim = max(self.model.patch_embed.img_size)
        self.patch_size = self.model.patch_embed.patch_size
        if self.maxdim != 512:
            raise ValueError(f'Unexpected official model input size: {self.model.patch_embed.img_size}')
        self._view_cache = OrderedDict()
        self._cache_limit = 12
        print(json.dumps({
            'mast3r_commit': git_head,
            'dust3r_commit': submodule_revs['dust3r'],
            'checkpoint_sha256': self.checkpoint_sha256,
            'device': self.device,
            'torch_version': __import__('torch').__version__,
            'cuda_available': __import__('torch').cuda.is_available(),
            'cuda_rope_extension': False,
        }, ensure_ascii=False), flush=True)

    def _view(self, image_path, reference_id, arrays):
        image_path = Path(image_path).resolve()
        cache_key = (str(image_path), int(reference_id))
        if cache_key in self._view_cache:
            value = self._view_cache.pop(cache_key)
            self._view_cache[cache_key] = value
            return value
        import torch
        with Image.open(image_path) as image_file:
            rgb = image_file.convert('RGB')
        width, height = rgb.size
        normalized = self.ImgNorm(rgb)
        rgb_full = normalized.permute(1, 2, 0).contiguous()
        resize_op, _, to_orig = self.get_resize_function(
            self.maxdim, self.patch_size, height, width)
        rgb_rescaled = resize_op(normalized).contiguous()
        valid = torch.zeros((height, width), dtype=torch.bool)
        if reference_id >= 0:
            start, end = arrays['pixel_offsets'][reference_id:reference_id + 2]
            pixel_xy = arrays['pixel_xy'][start:end]
            if len(pixel_xy):
                xy = torch.as_tensor(pixel_xy, dtype=torch.long)
                valid[xy[:, 1], xy[:, 0]] = True
        view = {
            'image_path': image_path,
            'width': width,
            'height': height,
            'rgb_full': rgb_full,
            'rgb_rescaled': rgb_rescaled,
            'to_orig': np.asarray(to_orig, dtype=np.float64),
            'resolution': self.get_HW_resolution(height, width, self.maxdim, self.patch_size),
            'valid': valid,
        }
        self._view_cache[cache_key] = view
        while len(self._view_cache) > self._cache_limit:
            self._view_cache.popitem(last=False)
        return view

    def _coarse_matches(self, query_view, map_view):
        import numpy as np
        q_shape = np.int32([query_view['rgb_rescaled'].shape[1:]])
        m_shape = np.int32([map_view['rgb_rescaled'].shape[1:]])
        pair = (
            {'img': query_view['rgb_rescaled'].unsqueeze(0), 'true_shape': q_shape,
             'idx': 0, 'instance': 'query'},
            {'img': map_view['rgb_rescaled'].unsqueeze(0), 'true_shape': m_shape,
             'idx': 1, 'instance': 'map'},
        )
        output = self.inference([pair], self.model, self.device, batch_size=1, verbose=False)
        desc_query = output['pred1']['desc'].squeeze(0).detach()
        desc_map = output['pred2']['desc'].squeeze(0).detach()
        if len(desc_map) == 0 or len(desc_query) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        coarse_map, coarse_query = self.fast_reciprocal_NNs(
            desc_map, desc_query, subsample_or_initxy1=COARSE_SUBSAMPLE,
            device=self.device, dist='dot', block_size=FAST_NN_BLOCK_SIZE)
        map_height, map_width = map_view['rgb_rescaled'].shape[-2:]
        query_height, query_width = query_view['rgb_rescaled'].shape[-2:]
        valid_map = (coarse_map[:, 0] >= COARSE_BORDER_PX) & (coarse_map[:, 0] < map_width - COARSE_BORDER_PX)
        valid_map &= (coarse_map[:, 1] >= COARSE_BORDER_PX) & (coarse_map[:, 1] < map_height - COARSE_BORDER_PX)
        valid_query = (coarse_query[:, 0] >= COARSE_BORDER_PX) & (coarse_query[:, 0] < query_width - COARSE_BORDER_PX)
        valid_query &= (coarse_query[:, 1] >= COARSE_BORDER_PX) & (coarse_query[:, 1] < query_height - COARSE_BORDER_PX)
        valid = valid_map & valid_query
        coarse_map = apply_pixel_transform(map_view['to_orig'], coarse_map[valid], pixel_centers=True)
        coarse_query = apply_pixel_transform(query_view['to_orig'], coarse_query[valid], pixel_centers=True)
        return coarse_map, coarse_query

    def _fine_matches(self, query_view, map_view, coarse_map, coarse_query):
        import torch
        from dust3r.inference import inference
        if len(coarse_map) == 0:
            return [], {'coarse_matches': 0, 'crop_pairs': 0, 'raw_matches': 0,
                        'confidence_kept': 0, 'identity_mismatch': 0}
        crop_pairs = list(self.select_pairs_of_crops(
            map_view['rgb_full'], query_view['rgb_full'], coarse_map, coarse_query,
            maxdim=self.maxdim, overlap=CROP_OVERLAP,
            forced_resolution=[map_view['resolution'], query_view['resolution']]))
        if not crop_pairs:
            return [], {'coarse_matches': int(len(coarse_map)), 'crop_pairs': 0,
                        'raw_matches': 0, 'confidence_kept': 0, 'identity_mismatch': 0}
        map_crops = []
        query_crops = []
        map_valid_crops = []
        map_transforms = []
        query_transforms = []
        for cell_map, cell_query, _ in crop_pairs:
            x0m, y0m, x1m, y1m = map(int, cell_map)
            x0q, y0q, x1q, y1q = map(int, cell_query)
            map_crops.append(map_view['rgb_full'][y0m:y1m, x0m:x1m])
            query_crops.append(query_view['rgb_full'][y0q:y1q, x0q:x1q])
            map_valid_crops.append(map_view['valid'][y0m:y1m, x0m:x1m])
            map_transform = np.eye(3, dtype=np.float64)
            query_transform = np.eye(3, dtype=np.float64)
            map_transform[:2, 2] = [x0m, y0m]
            query_transform[:2, 2] = [x0q, y0q]
            map_transforms.append(map_transform)
            query_transforms.append(query_transform)
        map_view_batch = {
            'img': torch.stack(map_crops).permute(0, 3, 1, 2).contiguous(),
            'instance': ['map' for _ in map_crops],
        }
        query_view_batch = {
            'img': torch.stack(query_crops).permute(0, 3, 1, 2).contiguous(),
            'instance': ['query' for _ in query_crops],
        }
        output = inference(
            [(query_view_batch, map_view_batch)], self.model, self.device,
            batch_size=FINE_MAX_BATCH_SIZE, verbose=False)
        pred_query = output['pred1']
        pred_map = output['pred2']
        raw = []
        identity_mismatch = 0
        for crop_id, valid_map in enumerate(map_valid_crops):
            y, x = torch.where(valid_map)
            if len(x) == 0:
                continue
            matches_map, matches_query = self.fast_reciprocal_NNs(
                pred_map['desc'][crop_id], pred_query['desc'][crop_id], (x, y),
                pixel_tol=FINE_PIXEL_TOLERANCE, device=self.device,
                dist='dot', block_size=FAST_NN_BLOCK_SIZE)
            if len(matches_map) == 0:
                continue
            confidence_map = pred_map['desc_conf'][crop_id].cpu().numpy()
            confidence_query = pred_query['desc_conf'][crop_id].cpu().numpy()
            conf = np.minimum(
                confidence_map[matches_map[:, 1], matches_map[:, 0]],
                confidence_query[matches_query[:, 1], matches_query[:, 0]])
            map_full = apply_pixel_transform(map_transforms[crop_id], matches_map)
            query_full = apply_pixel_transform(query_transforms[crop_id], matches_query)
            for idx in range(len(matches_map)):
                rounded_ref = np.floor(map_full[idx] + 0.5).astype(np.int32)
                if np.max(np.abs(map_full[idx] - rounded_ref)) > 0.5 + 1e-10:
                    identity_mismatch += 1
                    continue
                raw.append({
                    'reference_uv': map_full[idx].astype(np.float64),
                    'query_uv': query_full[idx].astype(np.float64),
                    'confidence': float(conf[idx]),
                })
        confident = [row for row in raw if np.isfinite(row['confidence']) and row['confidence'] >= CONFIDENCE_THRESHOLD]
        return confident, {
            'coarse_matches': int(len(coarse_map)),
            'crop_pairs': int(len(crop_pairs)),
            'raw_matches': int(len(raw)),
            'confidence_kept': int(len(confident)),
            'identity_mismatch': int(identity_mismatch),
        }

    def match_pair(self, query_image_path, reference_id, map_arrays):
        pair_started = time.perf_counter()
        offsets = map_arrays['pixel_offsets']
        start, end = offsets[reference_id:reference_id + 2]
        reference_key = str(map_arrays['reference_keys'][reference_id])
        reference_image = self._image_path_by_key[reference_key]
        query_view = self._view(query_image_path, -1, map_arrays)
        map_view = self._view(reference_image, reference_id, map_arrays)
        coarse_started = time.perf_counter()
        coarse_map, coarse_query = self._coarse_matches(query_view, map_view)
        coarse_elapsed = time.perf_counter() - coarse_started
        fine_started = time.perf_counter()
        raw_matches, summary = self._fine_matches(query_view, map_view, coarse_map, coarse_query)
        fine_elapsed = time.perf_counter() - fine_started
        width, height = map_view['width'], map_view['height']
        pixels = map_arrays['pixel_xy'][start:end]
        lookup = np.full((height, width), -1, dtype=np.int32)
        if len(pixels):
            lookup[pixels[:, 1], pixels[:, 0]] = np.arange(len(pixels), dtype=np.int32)
        candidates = []
        for row in raw_matches:
            ref_pixel = np.floor(row['reference_uv'] + 0.5).astype(np.int32)
            x, y = ref_pixel
            if not (0 <= x < width and 0 <= y < height):
                summary['identity_mismatch'] += 1
                continue
            local_index = int(lookup[y, x])
            if local_index < 0:
                summary['identity_mismatch'] += 1
                continue
            global_index = int(start + local_index)
            candidates.append({
                'reference_id': int(reference_id),
                'reference_key': reference_key,
                'reference_map_index': local_index,
                'reference_pixel_xy': ref_pixel.tolist(),
                'reference_uv': row['reference_uv'].tolist(),
                'query_uv': row['query_uv'].tolist(),
                'world_xyz': map_arrays['world_points'][global_index].tolist(),
                'body_xyz': map_arrays['body_points'][global_index].tolist(),
                'raw_point_index': int(map_arrays['raw_point_indices'][global_index]),
                'train_scan': reference_key,
                'confidence': float(row['confidence']),
            })
        summary['anchor_identity_kept'] = int(len(candidates))
        summary['anchor_identity_rejected'] = int(summary['confidence_kept'] - len(candidates))
        summary['coarse_elapsed_seconds'] = coarse_elapsed
        summary['fine_elapsed_seconds'] = fine_elapsed
        summary['pair_elapsed_seconds'] = time.perf_counter() - pair_started
        return candidates, summary

    def bind_data(self, data_root, raw_manifest, raw_manifest_path):
        root = Path(data_root).resolve()
        self._image_path_by_key = {
            key: resolve_manifest_image(raw_manifest_path, root, record)
            for key, record in raw_manifest['frames'].items()
        }


def resolve_manifest_image(raw_manifest_path, data_root, record):
    image_path = Path(record['image'])
    if image_path.is_absolute():
        return image_path.resolve()
    from_manifest = (Path(raw_manifest_path).resolve().parent / image_path).resolve()
    if from_manifest.is_file():
        return from_manifest
    return (Path(data_root).resolve() / image_path).resolve()


def run_subset(data_root, split_path, raw_manifest_path, map_path, map_metadata_path,
               baseline_predictions_path, output_dir, subset, matcher=None,
               official_root=None, checkpoint_path=None, device='cuda'):
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
    if observed_baseline_sha != expected_baseline_sha or not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip() != observed_baseline_sha:
        raise ValueError('Frozen relation prediction SHA256/sidecar mismatch')
    baseline = load_json(baseline_predictions_path)
    keys = list(split['splits'][subset])
    if baseline.get('protocol') != camera.ONLINE_PROTOCOL or baseline.get('checkpoint_sha256') != CHECKPOINT_SHA256:
        raise ValueError('Frozen relation prediction protocol or checkpoint mismatch')
    if baseline.get('split_sha256') != split_sha or baseline.get('subset') != subset:
        raise ValueError('Frozen relation prediction split or subset mismatch')
    if [row['scan'] for row in baseline['predictions']] != keys or baseline.get('expected_frames') != len(keys):
        raise ValueError('Frozen relation prediction does not cover the fixed subset in order')
    raw_manifest = load_json(raw_manifest_path)
    raw_sha = sha256_file(raw_manifest_path)
    map_arrays, map_metadata = load_map(
        map_path, map_metadata_path, data_root, split['splits']['train'], split_sha, raw_sha)
    if matcher is None:
        if official_root is None or checkpoint_path is None:
            raise ValueError('Official MASt3R source and local checkpoint are required')
        matcher = Mast3rCoarseToFineMatcher(official_root, checkpoint_path, device)
    matcher.bind_data(data_root, raw_manifest, raw_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
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
            detail_rows.append({'scan': scan_key, 'status': 'baseline_failure',
                                'reason': 'relation_prediction_not_ok', 'correspondences': 0, 'refined': False})
            continue
        if scan_key not in raw_manifest['frames']:
            raise ValueError(f'Raw manifest misses query image/calibration: {scan_key}')
        record = raw_manifest['frames'][scan_key]
        image_path = resolve_manifest_image(raw_manifest_path, data_root, record)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image_file:
            width, height = image_file.size
        query_sha = sha256_file(image_path)
        query_image_hashes.append({'scan': scan_key, 'image_sha256': query_sha})
        pose = np.asarray(original_pose, dtype=np.float64)
        camera_from_body = np.asarray(record['T_camera_lidar'], dtype=np.float64)
        intrinsics = np.asarray(record['K'], dtype=np.float64)
        references = choose_references(pose[:3, 3], map_arrays)
        raw_candidates = []
        reference_matches = []
        for reference_id, distance_m in references:
            candidates, match_summary = matcher.match_pair(image_path, reference_id, map_arrays)
            for candidate in candidates:
                candidate['reference_distance_m'] = distance_m
            raw_candidates.extend(candidates)
            reference_matches.append({'reference_id': reference_id, 'distance_m': distance_m, **match_summary})
        unique_candidates, selection = geometry_filter_then_deduplicate(
            raw_candidates, pose, camera_from_body, intrinsics, (width, height))
        selected, grid_summary = apply_grid_cap(unique_candidates, (width, height))
        world_points = np.asarray([item['world_xyz'] for item in selected], dtype=np.float64).reshape(-1, 3)
        query_uv = np.asarray([item['query_uv'] for item in selected], dtype=np.float64).reshape(-1, 2)
        refined_pose, refinement = refine_pose_sum(
            pose, camera_from_body, intrinsics, world_points, query_uv)
        result_pose = row['T_world_body'] if refinement['status'] == 'fallback' else refined_pose.tolist()
        frame_seconds = time.perf_counter() - frame_started
        refined_rows.append({'scan': scan_key, 'status': 'ok', 'T_world_body': result_pose,
                             'seconds': frame_seconds})
        correspondence_rows = [{
            'train_scan': item['train_scan'],
            'reference_id': int(item['reference_id']),
            'reference_map_index': int(item['reference_map_index']),
            'raw_point_index': int(item['raw_point_index']),
            'reference_pixel_xy': item['reference_pixel_xy'],
            'reference_uv': item['reference_uv'],
            'query_uv': item['query_uv'],
            'query_pixel_xy': item['query_pixel_xy'],
            'world_xyz': item['world_xyz'],
            'confidence': float(item['confidence']),
            'initial_reprojection_error_px': float(item['initial_reprojection_error_px']),
        } for item in selected]
        detail_rows.append({
            'scan': scan_key,
            'status': refinement['status'],
            'reason': refinement.get('reason'),
            'refined': refinement['status'] == 'refined',
            'query_image_sha256': query_sha,
            'image_size': [width, height],
            'references': reference_matches,
            **selection,
            **grid_summary,
            **refinement,
            'selected_correspondences': correspondence_rows,
            'seconds': frame_seconds,
        })
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(keys):
            print(f'{subset} {frame_index + 1}/{len(keys)}; raw={sum(x.get("raw_candidates", 0) for x in detail_rows)}; '
                  f'selected={sum(x.get("after_grid_and_total_cap", 0) for x in detail_rows)}', flush=True)
    elapsed = time.perf_counter() - started
    prediction = {
        'protocol': ONLINE_PROTOCOL,
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'split_sha256': split_sha,
        'training_split_sha256': split_sha,
        'subset': subset,
        'expected_frames': len(keys),
        'parent_prediction_sha256': observed_baseline_sha,
        'training_map_sha256': map_metadata['map_sha256'],
        'elapsed_seconds': elapsed,
        'query_ground_truth_access': 'none; runner inputs contain no query-pose path',
        'predictions': refined_rows,
    }
    prediction_path = output_dir / 'predictions.json'
    write_json(prediction_path, prediction)
    prediction_sha = sha256_file(prediction_path)
    prediction_path.with_suffix('.sha256').write_text(prediction_sha + '\n', encoding='ascii')
    details = {
        'protocol': 'camera_mast3r_anchored_diagnostics_v1',
        'subset': subset,
        'frozen_relation_predictions_sha256': observed_baseline_sha,
        'relation_checkpoint_sha256': CHECKPOINT_SHA256,
        'split_sha256': split_sha,
        'raw_manifest_sha256': raw_sha,
        'training_map_sha256': map_metadata['map_sha256'],
        'training_map_metadata_sha256': sha256_file(map_metadata_path),
        'refined_predictions_sha256': prediction_sha,
        'elapsed_seconds': elapsed,
        'frames': len(keys),
        'query_images': query_image_hashes,
        'fixed_parameters': fixed_parameters(),
        'query_ground_truth_access': 'none in online runner; independent evaluator may read after prediction SHA is frozen',
        'rows': detail_rows,
    }
    write_json(output_dir / 'details.json', details)
    print(json.dumps({
        'subset': subset,
        'frames': len(keys),
        'baseline_predictions_sha256': observed_baseline_sha,
        'refined_predictions_sha256': prediction_sha,
        'refined_frames': sum(item['refined'] for item in detail_rows),
        'fallback_frames': sum(item['status'] == 'fallback' for item in detail_rows),
        'elapsed_seconds': elapsed,
    }, ensure_ascii=False), flush=True)


def fixed_parameters():
    return {
        'model_name': MODEL_NAME,
        'coarse_resolution': 'nearest official MASt3R 512 resolution by aspect ratio',
        'coarse_mutual_nn_subsample': COARSE_SUBSAMPLE,
        'coarse_descriptor_distance': 'dot',
        'coarse_border_px': COARSE_BORDER_PX,
        'crop_maxdim': 512,
        'crop_overlap': CROP_OVERLAP,
        'minimum_coarse_matches_per_crop': MIN_COARSE_MATCHES_PER_CROP,
        'crop_homography': False,
        'fine_mutual_nn_pixel_tolerance': FINE_PIXEL_TOLERANCE,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'confidence_comparison': 'official MASt3R visloc >= threshold',
        'fast_nn_block_size': FAST_NN_BLOCK_SIZE,
        'fine_max_batch_size': FINE_MAX_BATCH_SIZE,
        'reference_distance_m': MAX_REFERENCE_DISTANCE_M,
        'max_references': MAX_REFERENCES,
        'initial_reprojection_limit_px': INITIAL_REPROJECTION_LIMIT_PX,
        'grid': [GRID_COLUMNS, GRID_ROWS],
        'max_per_grid_cell': MAX_PER_GRID_CELL,
        'max_correspondences': MAX_CORRESPONDENCES,
        'min_correspondences': MIN_CORRESPONDENCES,
        'duplicate_rank': 'descending MASt3R confidence, ascending initial reprojection error, stable reference and LiDAR point IDs',
        'optimizer': 'L-BFGS-B on right-composed normalized SE(3) increment',
        'translation_bound_m_per_axis': TRANSLATION_BOUND_M,
        'rotation_bound_deg_per_axis': 1.0,
        'huber_scale_px': HUBER_SCALE_PX,
        'visual_objective_reduction': 'sum over unique correspondences; no division by correspondence count',
        'optimizer_iterations': MAX_ITERATIONS,
        'degeneracy_condition_limit': DEGENERACY_CONDITION_LIMIT,
    }


def audit_training_pair(data_root, split_path, raw_manifest_path, map_path, map_metadata_path,
                        official_root, checkpoint_path, query_key, reference_key,
                        output_path, device='cuda'):
    split_path = Path(split_path).resolve()
    raw_manifest_path = Path(raw_manifest_path).resolve()
    raw_manifest = load_json(raw_manifest_path)
    if query_key not in raw_manifest['frames'] or reference_key not in raw_manifest['frames']:
        raise ValueError('Training-pair audit keys are absent from the raw manifest')
    if not query_key.startswith('scans/2012-') or not reference_key.startswith('scans/2012-'):
        raise ValueError('Training-pair audit requires training scan keys')
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    if split_sha != SPLIT_SHA256:
        raise ValueError('Training-pair audit split SHA mismatch')
    arrays, metadata = load_map(map_path, map_metadata_path, data_root,
                                split['splits']['train'], split_sha,
                                sha256_file(raw_manifest_path))
    reference_id = arrays['reference_keys'].tolist().index(reference_key)
    matcher = Mast3rCoarseToFineMatcher(official_root, checkpoint_path, device)
    matcher.bind_data(data_root, raw_manifest, raw_manifest_path)
    query_path = resolve_manifest_image(raw_manifest_path, data_root, raw_manifest['frames'][query_key])
    candidates, summary = matcher.match_pair(query_path, reference_id, arrays)
    output = {
        'protocol': 'mast3r_anchored_training_pair_identity_audit_v1',
        'query_training_scan': query_key,
        'reference_training_scan': reference_key,
        'training_map_sha256': metadata['map_sha256'],
        'checkpoint_sha256': matcher.checkpoint_sha256,
        'summary': summary,
        'all_reference_matches_resolve_to_exact_lidar_pixel_anchor': summary['identity_mismatch'] == 0,
        'sample_correspondences': candidates[:64],
    }
    write_json(output_path, output)
    if summary['identity_mismatch']:
        raise RuntimeError(f"{summary['identity_mismatch']} matches did not resolve to exact LiDAR anchors")


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
    run.add_argument('--mast3r-root', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--device', default='cuda')
    audit = subparsers.add_parser('audit-training-pair')
    audit.add_argument('--data-root', type=Path, required=True)
    audit.add_argument('--split', type=Path, required=True)
    audit.add_argument('--raw-manifest', type=Path, required=True)
    audit.add_argument('--map', type=Path, required=True)
    audit.add_argument('--map-metadata', type=Path, required=True)
    audit.add_argument('--mast3r-root', type=Path, required=True)
    audit.add_argument('--checkpoint', type=Path, required=True)
    audit.add_argument('--query-key', required=True)
    audit.add_argument('--reference-key', required=True)
    audit.add_argument('--out', type=Path, required=True)
    audit.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'build-map':
        build_train_map(args.data_root, args.split, args.raw_manifest, args.out_dir)
    elif args.command == 'run':
        run_subset(args.data_root, args.split, args.raw_manifest, args.map,
                   args.map_metadata, args.baseline_predictions, args.out_dir,
                   args.subset, official_root=args.mast3r_root,
                   checkpoint_path=args.checkpoint, device=args.device)
    else:
        audit_training_pair(args.data_root, args.split, args.raw_manifest, args.map,
                            args.map_metadata, args.mast3r_root, args.checkpoint,
                            args.query_key, args.reference_key, args.out, args.device)


if __name__ == '__main__':
    main()
