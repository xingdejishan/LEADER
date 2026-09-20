import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_lidar, load_module, project_world


def load_lidar_online(cache_dir, frame_id):
    with np.load(Path(cache_dir) / (frame_id + ".npz")) as data:
        return {key: np.asarray(data[key]) for key in ("source", "prediction", "center")}


def pose_from_baseline_online(row, cache_dir, matcher, full_pool, device, seed, return_lidar_evidence=False):
    import torch

    cached = load_lidar_online(cache_dir, row["frame_id"])
    source = torch.as_tensor(cached["source"], dtype=torch.float32, device=device)
    prediction = torch.as_tensor(cached["prediction"], dtype=torch.float32, device=device)
    keep_count = max(min(50, len(prediction)), int(0.5 * len(prediction)))
    torch.manual_seed(seed)
    keep = prediction[:, 3].topk(keep_count).indices
    initial = matcher.estimator(source[keep][None], prediction[keep, :3][None])[0]
    output = full_pool(initial, source, prediction[:, :3], return_evidence=return_lidar_evidence)
    refined = output[0] if isinstance(output, (tuple, list)) else output
    pose = refined.detach().cpu().numpy().astype(np.float64)
    center = np.asarray(cached["center"], dtype=np.float64)
    pose[:3, 3] += center
    if not return_lidar_evidence:
        return pose, None
    evidence = output[2]
    if evidence is None:
        return pose, None
    lidar_evidence = {key: value.detach().cpu().numpy().astype(np.float64)
                      for key, value in evidence.items() if key != "threshold"}
    lidar_evidence["target"] += center
    residual = np.asarray(lidar_evidence["residual"], dtype=np.float64)
    weights = np.asarray(lidar_evidence["weights"], dtype=np.float64)
    lidar_evidence["residual_scale_m"] = max(float(np.sqrt(np.sum(weights * residual ** 2) /
                                                           max(3. * float(weights.sum()) - 6., 1.))), .005)
    return pose, lidar_evidence


def load_gray_mask(view):
    from PIL import Image

    image = np.asarray(Image.open(view["image"]).convert("L"), dtype=np.float64) / 255.0
    mask = np.asarray(np.load(view["mask"]), dtype=bool)
    if image.shape != mask.shape:
        raise ValueError("image/mask size mismatch for camera %s" % view["camera"])
    return image, mask


def normalize_patch(values, floor=1e-3):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    scale = max(float(np.sqrt(np.mean(centered * centered))), floor)
    return centered / scale, float(np.sqrt(np.mean(centered * centered)))


def sample_bilinear(image, uv):
    uv = np.asarray(uv, dtype=np.float64)
    height, width = image.shape[:2]
    x, y = uv[:, 0], uv[:, 1]
    valid = np.isfinite(uv).all(axis=1) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    safe_x = np.clip(np.nan_to_num(x, nan=0.0), 0, width - 1)
    safe_y = np.clip(np.nan_to_num(y, nan=0.0), 0, height - 1)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = safe_x - x0
    wy = safe_y - y0
    values = ((1 - wx) * (1 - wy) * image[y0, x0] + wx * (1 - wy) * image[y0, x1] +
              (1 - wx) * wy * image[y1, x0] + wx * wy * image[y1, x1])
    return values, valid


def apply_rotation_delta(initial, delta):
    return apply_pose_delta(initial, np.r_[np.zeros(3, dtype=np.float64), np.asarray(delta, dtype=np.float64)])


def apply_pose_delta(initial, delta):
    pose = np.asarray(initial, dtype=np.float64).copy()
    delta = np.asarray(delta, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ pose[:3, :3]
    pose[:3, 3] += delta[:3]
    return pose


def fit_surface_plane(anchor, surface_tree, surface_points, radius, min_neighbors):
    count = min(max(min_neighbors * 4, 32), len(surface_points))
    if count < min_neighbors:
        return None
    distances, indices = surface_tree.query(anchor, k=count)
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)
    keep = np.isfinite(distances) & (distances <= radius)
    if int(keep.sum()) < min_neighbors:
        return None
    points = surface_points[indices[keep]]
    centroid = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - centroid[None], full_matrices=False)
    if len(singular) < 3 or singular[1] <= 1e-4:
        return None
    normal = vh[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    return normal, float(normal @ centroid), int(len(points)), singular


def bind_reference_patch(anchor, reference_pose, reference_view, reference_uv, plane, patch_size, calibration):
    half = patch_size // 2
    offsets = np.asarray([(du, dv) for dv in range(-half, half) for du in range(-half, half)], dtype=np.float64)
    pixels = np.asarray(reference_uv, dtype=np.float64)[None] + offsets
    height, width = reference_view["shape"]
    if (pixels[:, 0].min() < half or pixels[:, 0].max() > width - 1 - half or
            pixels[:, 1].min() < half or pixels[:, 1].max() > height - 1 - half):
        return None
    normal, distance, _, _ = plane
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    rays_camera = homogeneous @ np.linalg.inv(calibration).T
    world_from_camera = reference_pose @ np.asarray(reference_view["camera_to_body"], dtype=np.float64)
    origin = world_from_camera[:3, 3]
    rays_world = rays_camera @ world_from_camera[:3, :3].T
    denominator = rays_world @ normal
    safe = np.abs(denominator) > 1e-8
    depth = np.full(len(pixels), np.nan, dtype=np.float64)
    depth[safe] = (distance - origin @ normal) / denominator[safe]
    safe &= np.isfinite(depth) & (depth > 0.1)
    if not safe.all():
        return None
    points = origin[None] + depth[:, None] * rays_world
    return points, pixels


def downsample_surface_points(world_xyz, voxel_size):
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    keys = np.floor(world_xyz / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return world_xyz[np.sort(first)]


def prepare_patch_context(references, rows_by_frame, lidar_cache, surface_voxel):
    surface_points = downsample_surface_points(references["world_xyz"], surface_voxel)
    surface_tree = cKDTree(surface_points)
    reference_indices = {}
    reference_trees = {}
    for camera in range(6):
        indices = np.where(np.asarray(references["camera_ids"]) == camera)[0]
        reference_indices[camera] = indices
        reference_trees[camera] = cKDTree(references["world_xyz"][indices]) if len(indices) else None
    reference_poses = {}
    for frame_id in np.unique(references["frame_ids"]).tolist():
        reference_poses[str(frame_id)] = load_lidar(lidar_cache, str(frame_id))["GT"].astype(np.float64)
    return surface_points, surface_tree, reference_indices, reference_trees, reference_poses


def build_visibility_depth_buffer(surface_points, pose, camera_to_body, calibration, image_shape, cell_px):
    height, width = image_shape
    depth_buffer = np.full(((height + cell_px - 1) // cell_px, (width + cell_px - 1) // cell_px),
                           np.inf, dtype=np.float64)
    if not len(surface_points):
        return depth_buffer
    uv, depth = project_world(surface_points, pose, camera_to_body, calibration)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.5)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if valid.any():
        cell_u = np.floor(uv[valid, 0] / cell_px).astype(np.int64)
        cell_v = np.floor(uv[valid, 1] / cell_px).astype(np.int64)
        np.minimum.at(depth_buffer, (cell_v, cell_u), depth[valid])
    return depth_buffer


def check_patch_visibility(query_uv, patch_depth, depth_buffer, cell_px, window_cells,
                           depth_margin_m, min_coverage, min_visible_fraction):
    if not len(query_uv):
        return False, {"coverage": 0., "visible_fraction": 0.}
    height, width = depth_buffer.shape
    cell_u = np.floor(query_uv[:, 0] / cell_px).astype(np.int64)
    cell_v = np.floor(query_uv[:, 1] / cell_px).astype(np.int64)
    nearest = np.full(len(query_uv), np.inf, dtype=np.float64)
    for dv in range(-window_cells, window_cells + 1):
        for du in range(-window_cells, window_cells + 1):
            uu = np.clip(cell_u + du, 0, width - 1)
            vv = np.clip(cell_v + dv, 0, height - 1)
            nearest = np.minimum(nearest, depth_buffer[vv, uu])
    covered = np.isfinite(nearest)
    coverage = float(covered.mean())
    if not covered.any():
        return False, {"coverage": coverage, "visible_fraction": 0.}
    visible = patch_depth[covered] <= nearest[covered] + depth_margin_m
    visible_fraction = float(visible.mean())
    accepted = coverage >= min_coverage and visible_fraction >= min_visible_fraction
    return accepted, {"coverage": coverage, "visible_fraction": visible_fraction}


def select_surface_patches(row, initial, references, rows_by_frame, reference_context, crop_radius,
                            min_view_cosine, plane_radius, min_plane_neighbors, patch_size,
                            max_patches_per_camera, grid_cell, min_contrast, visibility_check=False,
                            visibility_cell_px=4, visibility_window_cells=1, visibility_depth_margin_m=.5,
                            min_visibility_coverage=.5, min_visible_fraction=.9,
                            shared_plane_cache=None, shared_query_images=None):
    surface_points, surface_tree, reference_indices, reference_trees, reference_poses = reference_context
    view_by_frame_camera = {
        frame_id: {int(view["camera"]): view for view in frame["views"]}
        for frame_id, frame in rows_by_frame.items()
    }
    plane_cache = {}
    image_cache = {}
    calibration_cache = {}
    query_images = shared_query_images if shared_query_images is not None else {}
    candidates = []
    diagnostics = []
    visibility_buffers = {}
    if visibility_check:
        local_positions = surface_tree.query_ball_point(initial[:3, 3], crop_radius)
        local_surface = surface_points[np.asarray(local_positions, dtype=np.int64)] if local_positions else np.empty((0, 3))
        for query_view in row["views"]:
            camera = int(query_view["camera"])
            if camera in query_images:
                image_shape = query_images[camera][0].shape
            else:
                image_shape = load_gray_mask(query_view)[0].shape
            visibility_buffers[camera] = build_visibility_depth_buffer(
                local_surface, initial, np.asarray(query_view["camera_to_body"], dtype=np.float64),
                np.loadtxt(query_view["calibration"]).astype(np.float64), image_shape, visibility_cell_px)
    for query_view in sorted(row["views"], key=lambda value: int(value["camera"])):
        camera = int(query_view["camera"])
        query_image, query_mask = load_gray_mask(query_view)
        query_images[camera] = (query_image, query_mask)
        query_view_with_shape = dict(query_view, shape=query_image.shape)
        query_extrinsic = np.asarray(query_view["camera_to_body"], dtype=np.float64)
        calibration = np.loadtxt(query_view["calibration"]).astype(np.float64)
        query_center = (initial @ query_extrinsic)[:3, 3]
        indices = reference_indices[camera]
        tree = reference_trees[camera]
        if tree is None or not len(indices):
            diagnostics.append({"camera": camera, "candidates": 0, "selected": 0})
            continue
        local_positions = tree.query_ball_point(initial[:3, 3], crop_radius)
        local_indices = indices[np.asarray(local_positions, dtype=np.int64)] if local_positions else np.empty(0, dtype=np.int64)
        local_candidates = []
        for ref_index in local_indices:
            ref_frame = str(references["frame_ids"][ref_index])
            anchor = np.asarray(references["world_xyz"][ref_index], dtype=np.float64)
            ref_pose = reference_poses[ref_frame]
            ref_view = view_by_frame_camera[ref_frame][camera]
            ref_extrinsic = np.asarray(ref_view["camera_to_body"], dtype=np.float64)
            ref_center = (ref_pose @ ref_extrinsic)[:3, 3]
            query_ray = anchor - query_center
            ref_ray = anchor - ref_center
            query_norm = np.linalg.norm(query_ray)
            ref_norm = np.linalg.norm(ref_ray)
            if min(query_norm, ref_norm) <= 1e-6 or float(query_ray @ ref_ray) / (query_norm * ref_norm) < min_view_cosine:
                continue
            map_id = int(references["map_ids"][ref_index])
            if map_id not in plane_cache:
                shared_key = (map_id, tuple(anchor.tolist()))
                if shared_plane_cache is not None and shared_key in shared_plane_cache:
                    plane_cache[map_id] = shared_plane_cache[shared_key]
                else:
                    plane_cache[map_id] = fit_surface_plane(anchor, surface_tree, surface_points, plane_radius, min_plane_neighbors)
                    if shared_plane_cache is not None:
                        shared_plane_cache[shared_key] = plane_cache[map_id]
            plane = plane_cache[map_id]
            if plane is None:
                continue
            if ref_frame not in image_cache:
                image_cache[ref_frame] = {}
            if camera not in image_cache[ref_frame]:
                ref_image, ref_mask = load_gray_mask(ref_view)
                image_cache[ref_frame][camera] = (ref_image, ref_mask)
                calibration_cache[(ref_frame, camera)] = np.loadtxt(ref_view["calibration"]).astype(np.float64)
            ref_image, ref_mask = image_cache[ref_frame][camera]
            ref_view_with_shape = dict(ref_view, shape=ref_image.shape)
            bound = bind_reference_patch(anchor, ref_pose, ref_view_with_shape, references["ref_uv"][ref_index],
                                         plane, patch_size, calibration_cache[(ref_frame, camera)])
            if bound is None:
                continue
            points, ref_pixels = bound
            ref_integer = np.rint(ref_pixels).astype(np.int64)
            if not ref_mask[ref_integer[:, 1], ref_integer[:, 0]].all():
                continue
            ref_values, ref_valid = sample_bilinear(ref_image, ref_pixels)
            if not ref_valid.all():
                continue
            ref_normalized, contrast = normalize_patch(ref_values)
            if contrast < min_contrast:
                continue
            query_uv, depth = project_world(points, initial, query_extrinsic, calibration)
            q_values, q_valid = sample_bilinear(query_image, query_uv)
            rounded = np.rint(query_uv).astype(np.int64)
            q_mask_valid = np.zeros(len(query_uv), dtype=bool)
            safe = q_valid & (rounded[:, 0] >= 0) & (rounded[:, 0] < query_image.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < query_image.shape[0])
            q_mask_valid[safe] = query_mask[rounded[safe, 1], rounded[safe, 0]]
            if not (q_valid & (depth > 0.5) & q_mask_valid).all():
                continue
            query_normalized, _ = normalize_patch(q_values)
            visibility_stats = {"coverage": float("nan"), "visible_fraction": float("nan")}
            if visibility_check:
                visible, visibility_stats = check_patch_visibility(
                    query_uv, depth, visibility_buffers[camera], visibility_cell_px,
                    visibility_window_cells, visibility_depth_margin_m, min_visibility_coverage,
                    min_visible_fraction)
                if not visible:
                    continue
            cell = tuple(np.floor(query_uv.mean(axis=0) / grid_cell).astype(np.int64))
            local_candidates.append({
                "points": points,
                "reference": ref_normalized,
                "camera": camera,
                "query_center": query_uv.mean(axis=0),
                "cell": cell,
                "contrast": float(contrast),
                "initial_patch_rmse": float(np.sqrt(np.mean((query_normalized - ref_normalized) ** 2))),
                "reference_frame": ref_frame,
                "reference_index": int(ref_index),
                "map_id": map_id,
                "anchor": anchor.copy(),
                "plane": plane,
                "reference_pose": ref_pose.copy(),
                "reference_camera_to_body": ref_extrinsic.copy(),
                "reference_calibration": calibration_cache[(ref_frame, camera)].copy(),
                "reference_uv": np.asarray(references["ref_uv"][ref_index], dtype=np.float64).copy(),
                "reference_shape": tuple(ref_image.shape),
                "patch_size": int(patch_size),
                "visibility": visibility_stats,
            })
        by_cell = {}
        for item in sorted(local_candidates, key=lambda value: (-value["contrast"], value["reference_frame"], value["reference_index"])):
            by_cell.setdefault(item["cell"], item)
        ordered = sorted(by_cell.values(), key=lambda value: (value["cell"][1], value["cell"][0]))
        if len(ordered) > max_patches_per_camera:
            selected_indices = np.linspace(0, len(ordered) - 1, max_patches_per_camera, dtype=np.int64)
            selected = [ordered[int(index)] for index in selected_indices]
        else:
            selected = ordered
        candidates.extend(selected)
        diagnostics.append({"camera": camera, "candidates": len(local_candidates), "grid_cells": len(by_cell),
                            "selected": len(selected), "visibility_check": bool(visibility_check)})
    return candidates, query_images, diagnostics


def patch_visual_residual_blocks(pose, patches, query_images, camera_data, invalid_penalty=2.0):
    values = []
    for patch in patches:
        camera = patch["camera"]
        image, mask = query_images[camera]
        uv, depth = project_world(patch["points"], pose, *camera_data[camera])
        sampled, valid = sample_bilinear(image, uv)
        rounded = np.rint(uv).astype(np.int64)
        mask_valid = np.zeros(len(uv), dtype=bool)
        safe = valid & (rounded[:, 0] >= 0) & (rounded[:, 0] < image.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < image.shape[0])
        mask_valid[safe] = mask[rounded[safe, 1], rounded[safe, 0]]
        if not (valid & (depth > 0.5) & mask_valid).all():
            values.append(np.full(len(uv), invalid_penalty, dtype=np.float64))
            continue
        normalized, _ = normalize_patch(sampled)
        values.append(normalized - patch["reference"])
    return values


def patch_visual_residual(pose, patches, query_images, camera_data, invalid_penalty=2.0):
    blocks = patch_visual_residual_blocks(pose, patches, query_images, camera_data, invalid_penalty)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float64)


def prepare_patch_batch(patches, query_images, camera_data):
    grouped = {}
    for index, patch in enumerate(patches):
        grouped.setdefault(int(patch["camera"]), []).append((index, patch))
    batches = []
    for camera in sorted(grouped):
        items = grouped[camera]
        sizes = np.asarray([len(patch["points"]) for _, patch in items], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes)]
        batches.append({
            "indices": [index for index, _ in items],
            "points": np.concatenate([patch["points"] for _, patch in items], axis=0),
            "references": [patch["reference"] for _, patch in items],
            "offsets": offsets,
            "image": query_images[camera][0],
            "mask": query_images[camera][1],
            "camera_data": camera_data[camera],
        })
    return batches


def patch_visual_residual_blocks_batched(pose, patch_batch, patch_count, invalid_penalty=2.0,
                                         block_transforms=None):
    blocks = [None] * patch_count
    for batch in patch_batch:
        uv, depth = project_world(batch["points"], pose, *batch["camera_data"])
        sampled, valid = sample_bilinear(batch["image"], uv)
        rounded = np.rint(uv).astype(np.int64)
        mask_valid = np.zeros(len(uv), dtype=bool)
        safe = valid & (rounded[:, 0] >= 0) & (rounded[:, 0] < batch["image"].shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < batch["image"].shape[0])
        mask_valid[safe] = batch["mask"][rounded[safe, 1], rounded[safe, 0]]
        for local, index in enumerate(batch["indices"]):
            start, end = batch["offsets"][local:local + 2]
            patch_valid = valid[start:end] & (depth[start:end] > 0.5) & mask_valid[start:end]
            if not patch_valid.all():
                blocks[index] = np.full(end - start, invalid_penalty, dtype=np.float64)
                continue
            normalized, _ = normalize_patch(sampled[start:end])
            block = normalized - batch["references"][local]
            if block_transforms is not None:
                block = block_transforms[index] @ block
            blocks[index] = block
    return blocks


def patch_visual_residual_batched(pose, patch_batch, patch_count, invalid_penalty=2.0,
                                  block_transforms=None):
    blocks = patch_visual_residual_blocks_batched(
        pose, patch_batch, patch_count, invalid_penalty, block_transforms)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float64)


def huber_patch_factor(rms, scale, count, pixel_count=1):
    rms = np.asarray(rms, dtype=np.float64)
    scale = max(float(scale), 1e-12)
    pixel_count = np.maximum(np.asarray(pixel_count, dtype=np.float64), 1.)
    normalized = rms / scale
    rho = np.where(normalized <= 1., .5 * normalized ** 2, normalized - .5)
    return np.sqrt(pixel_count) * scale * np.sqrt(2. * rho) / math.sqrt(max(int(count), 1))


def patch_robust_visual_residual(pose, patches, query_images, camera_data, scale,
                                 invalid_penalty=2.0):
    blocks = patch_visual_residual_blocks(pose, patches, query_images, camera_data, invalid_penalty)
    if not blocks:
        return np.empty(0, dtype=np.float64)
    rms = np.asarray([np.sqrt(np.mean(block ** 2)) for block in blocks], dtype=np.float64)
    pixel_count = np.asarray([block.size for block in blocks], dtype=np.float64)
    return huber_patch_factor(rms, scale, len(blocks), pixel_count)


def patch_robust_visual_residual_batched(pose, patch_batch, patch_count, scale,
                                         invalid_penalty=2.0, block_transforms=None):
    blocks = patch_visual_residual_blocks_batched(
        pose, patch_batch, patch_count, invalid_penalty, block_transforms)
    if not blocks:
        return np.empty(0, dtype=np.float64)
    rms = np.asarray([np.sqrt(np.mean(block ** 2)) for block in blocks], dtype=np.float64)
    pixel_count = np.asarray([block.size for block in blocks], dtype=np.float64)
    return huber_patch_factor(rms, scale, len(blocks), pixel_count)


def patch_points_for_plane_delta(patch, delta):
    normal, distance, count, singular = patch["plane"]
    anchor = np.asarray(patch["anchor"], dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64) + np.asarray(delta[:3], dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    local_distance = float(distance - np.asarray(patch["plane"][0], dtype=np.float64) @ anchor + delta[3])
    plane = (normal, float(normal @ anchor + local_distance), count, singular)
    reference_view = {"shape": patch["reference_shape"],
                      "camera_to_body": patch["reference_camera_to_body"]}
    bound = bind_reference_patch(anchor, patch["reference_pose"], reference_view,
                                 patch["reference_uv"], plane, patch["patch_size"],
                                 patch["reference_calibration"])
    return None if bound is None else bound[0]


def patch_query_residual(pose, patch, points, query_images, camera_data):
    if points is None:
        return None
    image, mask = query_images[patch["camera"]]
    uv, depth = project_world(points, pose, *camera_data[patch["camera"]])
    sampled, valid = sample_bilinear(image, uv)
    rounded = np.rint(uv).astype(np.int64)
    safe = valid & (rounded[:, 0] >= 0) & (rounded[:, 0] < image.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < image.shape[0])
    mask_valid = np.zeros(len(uv), dtype=bool)
    mask_valid[safe] = mask[rounded[safe, 1], rounded[safe, 0]]
    if not (valid & (depth > 0.5) & mask_valid).all():
        return None
    normalized, _ = normalize_patch(sampled)
    return normalized - patch["reference"]


def patch_geometry_transform(initial, patch, query_images, camera_data, covariance, steps):
    nominal = patch_query_residual(initial, patch, patch["points"], query_images, camera_data)
    if nominal is None:
        return np.eye(len(patch["reference"]), dtype=np.float64), 1., 1.
    jacobian = np.zeros((len(nominal), len(steps)), dtype=np.float64)
    invalid_columns = 0
    for axis, step in enumerate(steps):
        delta = np.zeros(len(steps), dtype=np.float64)
        delta[axis] = step
        plus = patch_query_residual(
            initial, patch, patch_points_for_plane_delta(patch, delta), query_images, camera_data)
        minus = patch_query_residual(
            initial, patch, patch_points_for_plane_delta(patch, -delta), query_images, camera_data)
        if plus is not None and minus is not None:
            jacobian[:, axis] = (plus - minus) / (2. * step)
        else:
            invalid_columns += 1
    propagated = jacobian @ covariance @ jacobian.T
    propagated = .5 * (propagated + propagated.T)
    values, vectors = np.linalg.eigh(np.eye(len(nominal)) + propagated)
    values = np.maximum(values, 1e-8)
    transform = (vectors * (1. / np.sqrt(values))[None, :]) @ vectors.T
    norm = np.linalg.norm(nominal)
    ratio = float(np.linalg.norm(transform @ nominal) / norm) if norm > 1e-12 else 1.
    return transform, ratio, invalid_columns / max(len(steps), 1)


def prepare_patch_geometry_transforms(initial, patches, query_images, camera_data, model):
    covariance = np.asarray(model["parameter_covariance"], dtype=np.float64)
    steps = np.asarray(model["parameter_steps"], dtype=np.float64)
    transforms = []
    ratios = []
    invalid_rates = []
    for patch in patches:
        transform, ratio, invalid_rate = patch_geometry_transform(
            initial, patch, query_images, camera_data, covariance, steps)
        transforms.append(transform)
        ratios.append(ratio)
        invalid_rates.append(invalid_rate)
    return transforms, np.asarray(ratios, dtype=np.float64), np.asarray(invalid_rates, dtype=np.float64)


def finite_difference_jacobian(residual, steps):
    steps = np.asarray(steps, dtype=np.float64)

    def jacobian(x):
        x = np.asarray(x, dtype=np.float64)
        base_size = len(residual(x))
        output = np.empty((base_size, len(x)), dtype=np.float64)
        for axis, step in enumerate(steps):
            offset = np.zeros_like(x)
            offset[axis] = step
            output[:, axis] = (residual(x + offset) - residual(x - offset)) / (2. * step)
        return output

    return jacobian


def lidar_pose_residual(pose, evidence):
    if evidence is None:
        return np.empty(0, dtype=np.float64)
    source = np.asarray(evidence["source"], dtype=np.float64)
    target = np.asarray(evidence["target"], dtype=np.float64)
    weights = np.asarray(evidence["weights"], dtype=np.float64)
    residual_scale = max(float(evidence["residual_scale_m"]), 1e-4)
    raw = source @ pose[:3, :3].T + pose[:3, 3] - target
    return (np.sqrt(np.maximum(weights, 0.))[:, None] * raw / residual_scale).ravel()


def refine_pose(initial, patches, query_images, row, variant, lidar_evidence,
                max_translation, max_rotation, prior_sigma, robust_scale,
                visual_weight, lidar_weight, translation_step_m, rotation_step_rad, max_nfev,
                patch_robust_scale, geometry_model=None, weak_visual_scale=1.):
    camera_data = {int(view["camera"]): (np.asarray(view["camera_to_body"], dtype=np.float64),
                                          np.loadtxt(view["calibration"]).astype(np.float64))
                   for view in row["views"]}
    patch_count = max(len(patches), 1)
    patch_batch = prepare_patch_batch(patches, query_images, camera_data)
    block_transforms = None
    geometry_ratios = np.empty(0, dtype=np.float64)
    geometry_invalid_rates = np.empty(0, dtype=np.float64)
    if variant == "patch_geometry":
        if geometry_model is None:
            raise ValueError("patch_geometry requires a geometry uncertainty model")
        block_transforms, geometry_ratios, geometry_invalid_rates = prepare_patch_geometry_transforms(
            initial, patches, query_images, camera_data, geometry_model)
    before_visual = patch_visual_residual_batched(
        initial, patch_batch, len(patches), block_transforms=block_transforms)
    before_lidar = lidar_pose_residual(initial, lidar_evidence)

    def residual(delta):
        if variant == "rotation":
            pose = apply_rotation_delta(initial, delta)
        else:
            pose = apply_pose_delta(initial, delta)
        if variant == "lidar":
            return lidar_weight * lidar_pose_residual(pose, lidar_evidence)
        if variant == "patch_robust":
            visual = patch_robust_visual_residual_batched(
                pose, patch_batch, len(patches), patch_robust_scale)
        elif variant == "patch_geometry":
            visual = patch_robust_visual_residual_batched(
                pose, patch_batch, len(patches), patch_robust_scale,
                block_transforms=block_transforms)
        elif variant == "patch_weak":
            visual = weak_visual_scale * patch_robust_visual_residual_batched(
                pose, patch_batch, len(patches), patch_robust_scale)
        else:
            visual = patch_visual_residual_batched(pose, patch_batch, len(patches)) / math.sqrt(patch_count)
        if variant == "rotation":
            prior = np.asarray(delta, dtype=np.float64) / prior_sigma
            return np.concatenate([visual_weight * visual, prior])
        lidar = lidar_pose_residual(pose, lidar_evidence)
        return np.concatenate([visual_weight * visual, lidar_weight * lidar])

    if variant == "rotation":
        x0 = np.zeros(3, dtype=np.float64)
        bounds = (-np.full(3, max_rotation), np.full(3, max_rotation))
        steps = np.full(3, rotation_step_rad, dtype=np.float64)
    else:
        x0 = np.zeros(6, dtype=np.float64)
        bounds = (-np.r_[np.full(3, max_translation), np.full(3, max_rotation)],
                  np.r_[np.full(3, max_translation), np.full(3, max_rotation)])
        steps = np.r_[np.full(3, translation_step_m), np.full(3, rotation_step_rad)]
    result = least_squares(residual, x0, jac=finite_difference_jacobian(residual, steps), bounds=bounds,
                           method="trf", loss="soft_l1", f_scale=robust_scale,
                           x_scale="jac", max_nfev=max_nfev)
    candidate = apply_rotation_delta(initial, result.x) if variant == "rotation" else apply_pose_delta(initial, result.x)
    after_visual = patch_visual_residual_batched(
        candidate, patch_batch, len(patches), block_transforms=block_transforms)
    after_lidar = lidar_pose_residual(candidate, lidar_evidence)
    return candidate, result, before_visual, after_visual, before_lidar, after_lidar, geometry_ratios, geometry_invalid_rates


def digest_json(value):
    return __import__("hashlib").sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--reference-lidar-cache", default=None)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default="rotation", choices=("rotation", "joint", "patch_robust", "patch_geometry", "patch_weak", "lidar", "visibility", "all"))
    parser.add_argument("--evaluate-split", default="validation", choices=("validation", "train"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--surface-voxel-size", type=float, default=.2)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--min-view-cosine", type=float, default=.7)
    parser.add_argument("--plane-radius", type=float, default=.8)
    parser.add_argument("--min-plane-neighbors", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-patches-per-camera", type=int, default=48)
    parser.add_argument("--grid-cell", type=int, default=32)
    parser.add_argument("--min-contrast", type=float, default=.03)
    parser.add_argument("--max-rotation-deg", type=float, default=2.)
    parser.add_argument("--max-translation-m", type=float, default=.5)
    parser.add_argument("--prior-sigma-deg", type=float, default=1.)
    parser.add_argument("--robust-scale", type=float, default=.2)
    parser.add_argument("--patch-robust-scale", type=float, default=.2)
    parser.add_argument("--geometry-uncertainty-model", default=None)
    parser.add_argument("--visual-weight", type=float, default=1.)
    parser.add_argument("--lidar-weight", type=float, default=1.)
    parser.add_argument("--translation-step-m", type=float, default=1e-4)
    parser.add_argument("--rotation-step-rad", type=float, default=1e-5)
    parser.add_argument("--visibility-cell-px", type=int, default=4)
    parser.add_argument("--visibility-window-cells", type=int, default=1)
    parser.add_argument("--visibility-depth-margin-m", type=float, default=.5)
    parser.add_argument("--min-visibility-coverage", type=float, default=.5)
    parser.add_argument("--min-visible-fraction", type=float, default=.9)
    parser.add_argument("--collect-visibility-diagnostic", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if args.variant in ("patch_geometry", "patch_weak", "all") and not args.geometry_uncertainty_model:
        parser.error("patch_geometry, patch_weak, and all require --geometry-uncertainty-model")
    if args.patch_size < 4 or args.patch_size % 2 or args.min_plane_neighbors < 3:
        parser.error("patch-size must be even and >= 4; min-plane-neighbors must be >= 3")
    if args.visibility_cell_px < 1 or args.visibility_window_cells < 0:
        parser.error("visibility cell size must be positive and window must be non-negative")
    if not (0. < args.min_visibility_coverage <= 1. and 0. < args.min_visible_fraction <= 1.):
        parser.error("visibility fractions must be in (0, 1]")
    if min(args.map_voxel_size, args.surface_voxel_size, args.crop_radius, args.plane_radius,
           args.min_view_cosine, args.min_contrast, args.max_rotation_deg, args.prior_sigma_deg,
           args.robust_scale, args.patch_robust_scale, args.max_translation_m, args.visual_weight,
           args.lidar_weight, args.translation_step_m, args.rotation_step_rad,
           args.visibility_depth_margin_m) <= 0:
        parser.error("geometric and optimization scales must be positive")

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    reference_lidar_cache = args.reference_lidar_cache or args.lidar_cache
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = train_rows if args.evaluate_split == "train" else [row for row in rows if row["split"] in ("val", "validation", "test")]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
        rows_for_map = train_rows + [row for row in rows if row["split"] != "train"]
    else:
        rows_for_map = rows
    if args.frames:
        eval_rows = eval_rows[:args.frames]

    references = build_reference_observations(rows_for_map, reference_lidar_cache, args.projection_cache,
                                              args.map_voxel_size, Path(args.map_cache), 0)
    rows_by_frame = {row["frame_id"]: row for row in rows_for_map}
    reference_context = prepare_patch_context(references, rows_by_frame, reference_lidar_cache, args.surface_voxel_size)
    geometry_model = None
    weak_visual_scale = 1.
    if args.geometry_uncertainty_model:
        geometry_model = json.loads(Path(args.geometry_uncertainty_model).read_text(encoding="utf-8"))
        weak_visual_scale = float(geometry_model["weak_visual_scale"])
    matcher_module = load_module("surface_patch_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("surface_patch_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    records = []
    started = time.time()
    shared_plane_cache = {}
    for index, row in enumerate(eval_rows):
        frame_started = time.perf_counter()
        timing = {}
        stage_started = time.perf_counter()
        initial, lidar_evidence = pose_from_baseline_online(
            row, args.lidar_cache, matcher, full_pool_module.full_pool_refine,
            args.device, args.seed + index, return_lidar_evidence=True)
        timing["baseline_s"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        patches, query_images, patch_diagnostics = select_surface_patches(
            row, initial, references, rows_by_frame, reference_context, args.crop_radius,
            args.min_view_cosine, args.plane_radius, args.min_plane_neighbors, args.patch_size,
            args.max_patches_per_camera, args.grid_cell, args.min_contrast,
            shared_plane_cache=shared_plane_cache)
        timing["patch_build_s"] = time.perf_counter() - stage_started
        need_visibility = args.variant == "visibility" or args.collect_visibility_diagnostic
        if need_visibility:
            stage_started = time.perf_counter()
            visible_patches, _, visibility_diagnostics = select_surface_patches(
                row, initial, references, rows_by_frame, reference_context, args.crop_radius,
                args.min_view_cosine, args.plane_radius, args.min_plane_neighbors, args.patch_size,
                args.max_patches_per_camera, args.grid_cell, args.min_contrast, visibility_check=True,
                visibility_cell_px=args.visibility_cell_px, visibility_window_cells=args.visibility_window_cells,
                visibility_depth_margin_m=args.visibility_depth_margin_m,
                min_visibility_coverage=args.min_visibility_coverage,
                min_visible_fraction=args.min_visible_fraction,
                shared_plane_cache=shared_plane_cache, shared_query_images=query_images)
            timing["visibility_patch_build_s"] = time.perf_counter() - stage_started
        else:
            visible_patches, visibility_diagnostics = [], []
            timing["visibility_patch_build_s"] = 0.
        patch_sets = {"rotation": patches, "joint": patches, "patch_robust": patches,
                      "patch_geometry": patches, "patch_weak": patches,
                      "lidar": [], "visibility": visible_patches}
        variants = ("lidar", "patch_robust", "patch_geometry", "patch_weak") if args.variant == "all" else (args.variant,)
        variant_records = {}
        refine_started = time.perf_counter()
        for variant in variants:
            variant_started = time.perf_counter()
            variant_patches = patch_sets[variant]
            candidate = np.full((4, 4), np.nan, dtype=np.float64)
            solver = {"success": False, "status": -1, "nfev": 0, "cost": float("nan"),
                      "message": "no patches or no lidar evidence"}
            needs_visual = variant in ("rotation", "joint", "patch_robust", "patch_geometry", "patch_weak", "visibility")
            run_failed = (needs_visual and not variant_patches) or lidar_evidence is None
            before_rmse = after_rmse = float("nan")
            before_lidar = after_lidar = np.empty(0, dtype=np.float64)
            geometry_ratios = np.empty(0, dtype=np.float64)
            geometry_invalid_rates = np.empty(0, dtype=np.float64)
            delta = np.full(3 if variant == "rotation" else 6, np.nan, dtype=np.float64)
            if not run_failed:
                candidate, result, before_visual, after_visual, before_lidar, after_lidar, geometry_ratios, geometry_invalid_rates = refine_pose(
                    initial, variant_patches, query_images, row, variant, lidar_evidence,
                    args.max_translation_m, math.radians(args.max_rotation_deg),
                    math.radians(args.prior_sigma_deg), args.robust_scale,
                    args.visual_weight, args.lidar_weight, args.translation_step_m,
                    args.rotation_step_rad, args.max_nfev, args.patch_robust_scale,
                    geometry_model, weak_visual_scale)
                delta = np.asarray(result.x, dtype=np.float64)
                before_rmse = float(np.sqrt(np.mean(before_visual ** 2))) if len(before_visual) else float("nan")
                after_rmse = float(np.sqrt(np.mean(after_visual ** 2))) if len(after_visual) else float("nan")
                run_failed = not bool(result.success) or not np.isfinite(candidate).all() or not np.isfinite(delta).all()
                solver = {"success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev),
                          "cost": float(result.cost), "optimality": float(result.optimality),
                          "message": str(result.message)}
            variant_records[variant] = {
                "final_pose": candidate.tolist() if not run_failed else np.full((4, 4), np.nan).tolist(),
                "delta": delta.tolist(),
                "translation_delta_m": delta[:3].tolist() if np.isfinite(delta).all() else [float("nan")] * 3,
                "rotation_delta_rad": delta[-3:].tolist() if np.isfinite(delta).all() else [float("nan")] * 3,
                "rotation_delta_deg": float(np.degrees(np.linalg.norm(delta[-3:]))) if np.isfinite(delta).all() else float("nan"),
                "patch_count": int(len(variant_patches)),
                "visibility_check": bool(variant == "visibility"),
                "visual_rmse_before": before_rmse,
                "visual_rmse_after": after_rmse,
                "geometry_ratio_median": float(np.median(geometry_ratios)) if len(geometry_ratios) else float("nan"),
                "geometry_invalid_rate_median": float(np.median(geometry_invalid_rates)) if len(geometry_invalid_rates) else float("nan"),
                "geometry_invalid_rate_max": float(np.max(geometry_invalid_rates)) if len(geometry_invalid_rates) else float("nan"),
                "lidar_rmse_before": float(np.sqrt(np.mean(before_lidar ** 2))) if len(before_lidar) else float("nan"),
                "lidar_rmse_after": float(np.sqrt(np.mean(after_lidar ** 2))) if len(after_lidar) else float("nan"),
                "solver": solver,
                "run_failed": bool(run_failed),
            }
            timing["refine_%s_s" % variant] = time.perf_counter() - variant_started
        timing["refine_total_s"] = time.perf_counter() - refine_started
        timing["frame_total_s"] = time.perf_counter() - frame_started
        record = {
            "frame_id": row["frame_id"],
            "initial_pose": initial.tolist(),
            "patch_count": int(len(patches)),
            "patch_diagnostics": patch_diagnostics,
            "visibility_patch_count": int(len(visible_patches)),
            "visibility_diagnostics": visibility_diagnostics,
            "reference_frames": sorted({patch["reference_frame"] for patch in patches}),
            "reference_indices": [int(patch["reference_index"]) for patch in patches],
            "lidar_evidence_count": int(len(lidar_evidence["source"])) if lidar_evidence is not None else 0,
            "timing_s": timing,
            "variants": variant_records,
        }
        if args.variant != "all":
            record.update(variant_records[args.variant])
        records.append(record)
        status = " ".join("%s=%s/%.4fdeg" % (name, data["run_failed"], data["rotation_delta_deg"])
                          for name, data in variant_records.items())
        print("%s %d/%d %s patches=%d %s" % (
            args.evaluate_split, index + 1, len(eval_rows), row["frame_id"], len(patches), status), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "surface-bound real-image patch refinement",
        "map_source": "train split only",
        "query_gt_in_runner": False,
        "reference_patch": "8x8 grayscale pixels bound to frozen local fitted surface planes",
        "online_observation": "reference and query image bilinear samples at candidate-pose projections",
        "visual_objectives": {
            "joint": "mean normalized pixel residual squared with shared outer soft_l1",
            "patch_robust": "mean Huber(||patch_residual||_2/sqrt(m)) with fixed training scale and sqrt(m) small-error scale alignment",
            "patch_geometry": "patch_robust residual whitened by fixed initial-pose plane-uncertainty propagation",
            "patch_weak": "patch_robust residual with training-only scalar visual weakening",
        },
        "patch_robust_scale": args.patch_robust_scale,
        "patch_robust_scale_alignment": "sqrt(per-patch pixel count), fixed by patch_size",
        "geometry_uncertainty_model": str(args.geometry_uncertainty_model) if args.geometry_uncertainty_model else None,
        "weak_visual_scale": weak_visual_scale,
        "query_mask_enforced_during_optimization": True,
        "visibility_diagnostic_collected": bool(args.collect_visibility_diagnostic or args.variant == "visibility"),
        "visibility_filter": "frozen map-point z-buffer at LEADER initial pose for visibility variant",
        "pose_updates": {"rotation": "R=Exp(delta_rotation) R_LEADER; t=t_LEADER",
                         "joint": "R=Exp(delta_rotation) R_LEADER; t=t_LEADER+delta_translation"},
        "optimizer": {"method": "trf", "loss": "soft_l1", "jacobian": "central finite difference",
                      "translation_step_m": args.translation_step_m, "rotation_step_rad": args.rotation_step_rad,
                      "x_scale": "jac"},
        "no_baseline_fallback": True,
        "parameters": {key: value for key, value in vars(args).items() if key not in ("manifest", "lidar_cache", "projection_cache", "map_cache", "output", "full_pool")},
        "input_manifest": str(args.manifest),
        "online_lidar_cache": str(args.lidar_cache),
        "reference_lidar_cache": str(reference_lidar_cache),
        "reference_map_cache": str(args.map_cache),
        "reference_observation_count": int(len(references["world_xyz"])),
        "surface_point_count": int(len(reference_context[0])),
        "manifest_sha256": digest_json(rows),
    }
    timing_totals = {}
    for record in records:
        for key, value in record["timing_s"].items():
            timing_totals[key] = timing_totals.get(key, 0.) + value
    result = {"protocol": protocol, "evaluation_split": args.evaluate_split,
              "evaluation_frames": len(records), "records": records,
              "timing_totals_s": timing_totals, "elapsed_s": time.time() - started}
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
