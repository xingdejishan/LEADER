import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nre_scoremap_pose_runner as nre


SEARCH_RADIUS = 24
QUERY_PATCH_SIZE = 2 * SEARCH_RADIUS + 17
REFERENCE_PATCH_SIZE = 49
TRAIN_EPOCHS = 8
POSE_STEPS = 5
POSE_WEIGHT = 1.0
SEED = 2089


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backend(baseline_code_root, device):
    baseline_code = Path(baseline_code_root)
    sys.path.insert(0, str(baseline_code))
    matcher_spec = importlib.util.spec_from_file_location(
        "highres_pose_matcher", baseline_code / "models" / "sc2pcr.py")
    matcher_module = importlib.util.module_from_spec(matcher_spec)
    matcher_spec.loader.exec_module(matcher_module)
    pool_spec = importlib.util.spec_from_file_location(
        "highres_pose_full_pool", baseline_code / "tools" / "full_pool_robust_v1.py")
    pool_module = importlib.util.module_from_spec(pool_spec)
    pool_spec.loader.exec_module(pool_module)
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    return matcher, pool_module.full_pool_refine


def reference_view(rows_by_frame, frame_id, camera):
    row = rows_by_frame[str(frame_id)]
    return next(view for view in row["views"] if int(view["camera"]) == int(camera))


def target_pixels(points, pose, cameras, views):
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = np.full(len(points), np.nan, dtype=np.float64)
    for camera in np.unique(cameras):
        indices = np.where(cameras == camera)[0]
        view = next(view for view in views if int(view["camera"]) == int(camera))
        uv, z = nre.project_world(points[indices], pose,
                                  np.asarray(view["camera_to_body"], dtype=np.float64),
                                  np.loadtxt(view["calibration"]).astype(np.float64))
        pixels[indices] = uv
        depth[indices] = z
    return pixels, depth


def prepare_match_frame(row, rows_by_frame, cache_path, lidar_cache, matcher,
                        full_pool_refine, roma_features, device, seed):
    frame_id = str(row["frame_id"])
    with np.load(cache_path) as cache:
        required = {"points", "pixels", "reference_pixels", "cameras", "reference_frames",
                    "reference_cameras", "scores", "precisions"}
        if not required.issubset(cache.files):
            raise RuntimeError("match cache is missing required fields: " + frame_id)
        points = np.asarray(cache["points"], dtype=np.float64)
        reference_pixels = np.asarray(cache["reference_pixels"], dtype=np.float64)
        cameras = np.asarray(cache["cameras"], dtype=np.int64)
        reference_frames = np.asarray(cache["reference_frames"]).astype(str)
        reference_cameras = np.asarray(cache["reference_cameras"], dtype=np.int64)
        cache_scores = np.asarray(cache["scores"], dtype=np.float32)
        cache_precisions = np.asarray(cache["precisions"], dtype=np.float32)

    baseline, _, _, _, lidar_information, lidar_details, lidar_hash, baseline_support = \
        nre.build_baseline(row, lidar_cache, matcher, full_pool_refine, device, seed)
    n = len(points)
    centers = np.full((n, 2), np.nan, dtype=np.float64)
    depths = np.full(n, np.nan, dtype=np.float64)
    peaks = np.full((n, 2), np.nan, dtype=np.float64)
    peak_precisions = np.zeros((n, 2, 2), dtype=np.float64)
    peak_scores = np.full(n, np.nan, dtype=np.float32)
    cost_maps = np.full((n, 25, 25), 2., dtype=np.float32)
    map_valid = np.zeros((n, 25, 25), dtype=bool)
    keep = np.zeros(n, dtype=bool)
    query_views = {int(view["camera"]): view for view in row["views"]}

    for camera in np.unique(cameras):
        camera_indices = np.where(cameras == camera)[0]
        query_view = query_views[int(camera)]
        from PIL import Image
        with Image.open(query_view["image"]) as image:
            query_hw = (image.height, image.width)
        query_mask = np.asarray(np.load(query_view["mask"]), dtype=bool)
        query_features = roma_features.image_features(query_view["image"])
        camera_to_body = np.asarray(query_view["camera_to_body"], dtype=np.float64)
        calibration = np.loadtxt(query_view["calibration"]).astype(np.float64)
        base_uv, depth, _ = nre.pixel_pose_jacobian(points[camera_indices], baseline,
                                                   camera_to_body, calibration)
        centers[camera_indices] = base_uv
        depths[camera_indices] = depth
        groups = sorted(set(zip(reference_frames[camera_indices],
                                reference_cameras[camera_indices])))
        for ref_frame, ref_camera in groups:
            global_indices = camera_indices[(reference_frames[camera_indices] == ref_frame) &
                                            (reference_cameras[camera_indices] == ref_camera)]
            local_indices = np.searchsorted(camera_indices, global_indices)
            view = reference_view(rows_by_frame, ref_frame, ref_camera)
            with Image.open(view["image"]) as image:
                reference_hw = (image.height, image.width)
            reference_features = roma_features.image_features(view["image"])
            score_maps, valid_maps, _ = roma_features.similarity_maps(
                reference_features, query_features, reference_pixels[global_indices], reference_hw,
                base_uv[local_indices], query_hw, query_mask, depth[local_indices])
            for local_index, global_index in enumerate(global_indices):
                scores = score_maps[local_index].astype(np.float32)
                valid = valid_maps[local_index]
                if not valid[12, 12] or not valid[11:14, 11:14].all():
                    continue
                peak, precision, best_score, _ = nre.peak_quadratic(
                    scores, valid, base_uv[local_indices[local_index]])
                cost = np.full((25, 25), 2., dtype=np.float32)
                cost[valid] = np.maximum(best_score - scores[valid], 0.)
                peaks[global_index] = peak
                peak_precisions[global_index] = precision
                peak_scores[global_index] = best_score
                cost_maps[global_index] = cost
                map_valid[global_index] = valid
                keep[global_index] = True
            del reference_features
        del query_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not keep.any():
        raise RuntimeError("no valid local RoMa peaks for frame " + frame_id)
    return {
        "frame_id": frame_id,
        "points": points[keep],
        "cameras": cameras[keep],
        "reference_pixels": reference_pixels[keep],
        "reference_frames": reference_frames[keep],
        "reference_cameras": reference_cameras[keep],
        "cache_scores": cache_scores[keep],
        "cache_precisions": cache_precisions[keep],
        "centers": centers[keep],
        "depths": depths[keep],
        "peaks": peaks[keep],
        "peak_precisions": peak_precisions[keep],
        "peak_scores": peak_scores[keep],
        "cost_maps": cost_maps[keep],
        "map_valid": map_valid[keep],
        "baseline_pose": baseline,
        "lidar_information": lidar_information,
        "lidar_details": lidar_details,
        "lidar_input_sha256": lidar_hash,
        "baseline_support": baseline_support,
        "match_cache_sha256": sha256_file(cache_path),
        "match_cache_rows": int(n),
        "usable_local_peaks": int(keep.sum()),
    }


def add_training_targets(frame, row, lidar_cache):
    if row["split"] != "train":
        raise RuntimeError("training labels are only allowed for manifest train rows")
    with np.load(Path(lidar_cache) / (frame["frame_id"] + ".npz")) as cache:
        gt_pose = np.asarray(cache["GT"], dtype=np.float64)
    gt_pixels, gt_depth = target_pixels(frame["points"], gt_pose, frame["cameras"], row["views"])
    visible = np.isfinite(gt_pixels).all(axis=1) & (gt_depth > 0)
    for camera in np.unique(frame["cameras"]):
        indices = np.where(frame["cameras"] == camera)[0]
        view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
        mask = np.asarray(np.load(view["mask"]), dtype=bool)
        from PIL import Image
        with Image.open(view["image"]) as image:
            nonblack = np.asarray(image.convert("RGB"), dtype=np.uint8).max(axis=2) > 0
        h, w = mask.shape
        uv = gt_pixels[indices]
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w - 1) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
        x = np.clip(np.floor(uv[:, 0]).astype(np.int64), 0, w - 1)
        y = np.clip(np.floor(uv[:, 1]).astype(np.int64), 0, h - 1)
        visible[indices] &= inside & mask[y, x] & nonblack[y, x]
    frame["gt_pixels"] = gt_pixels.astype(np.float32)
    frame["gt_visible"] = visible
    frame["train_gt_pose_sha256"] = sha256_file(row["pose"])
    return frame


def save_match_frame(frame, path):
    metadata = {key: value for key, value in frame.items()
                if key in {"frame_id", "lidar_details", "lidar_input_sha256", "baseline_support",
                           "match_cache_sha256", "match_cache_rows", "usable_local_peaks",
                           "train_gt_pose_sha256", "training_lidar_cache_sha256",
                           "manifest_sha256", "roma_feature_state_sha256"}}
    arrays = {key: value for key, value in frame.items()
              if isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_match_frame(path):
    with np.load(path, allow_pickle=False) as archive:
        frame = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"]))
    frame.update(metadata)
    frame["frame_id"] = str(frame["frame_id"])
    return frame


def load_rgb(path, device):
    from PIL import Image

    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).to(device=device).permute(2, 0, 1).float().div_(255.)
    return ((tensor - .5) / .25).unsqueeze(0)


def crop_batch(image, pixels, size):
    count = len(pixels)
    if not count:
        return torch.empty((0, 3, size, size), dtype=image.dtype, device=image.device)
    offsets = torch.arange(size, dtype=pixels.dtype, device=pixels.device) - (size - 1) / 2
    xs = pixels[:, 0, None, None] + offsets[None, None, :]
    ys = pixels[:, 1, None, None] + offsets[None, :, None]
    xs = xs.expand(count, size, size)
    ys = ys.expand(count, size, size)
    height, width = image.shape[-2:]
    grid = torch.stack((2 * (xs + .5) / width - 1,
                        2 * (ys + .5) / height - 1), dim=-1)
    sampled = F.grid_sample(image.expand(count, -1, -1, -1), grid,
                            mode="bilinear", padding_mode="border", align_corners=False)
    return sampled


def build_patch_batch(frame, row, rows_by_frame, device):
    n = len(frame["points"])
    ref_patches = torch.empty((n, 3, REFERENCE_PATCH_SIZE, REFERENCE_PATCH_SIZE),
                              dtype=torch.float32, device=device)
    query_patches = torch.empty((n, 3, QUERY_PATCH_SIZE, QUERY_PATCH_SIZE),
                                dtype=torch.float32, device=device)
    candidate_valid = torch.zeros((n, 2 * SEARCH_RADIUS + 1, 2 * SEARCH_RADIUS + 1),
                                  dtype=torch.bool, device=device)
    shifts = torch.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1, dtype=torch.float32,
                          device=device)
    shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
    shift_grid = torch.stack((shift_x, shift_y), dim=-1)

    for camera in np.unique(frame["cameras"]):
        indices_np = np.where(frame["cameras"] == camera)[0]
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        query_view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
        query_image = load_rgb(query_view["image"], device)
        peaks = torch.as_tensor(frame["peaks"][indices_np], dtype=torch.float32, device=device)
        query_patches[indices] = crop_batch(query_image, peaks, QUERY_PATCH_SIZE)
        query_mask = np.asarray(np.load(query_view["mask"]), dtype=bool)
        height, width = query_mask.shape
        candidate_pixels = peaks[:, None, None, :] + shift_grid[None]
        x, y = candidate_pixels[..., 0], candidate_pixels[..., 1]
        in_image = ((x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1))
        ix = torch.floor(x).long().clamp(0, width - 1).cpu().numpy()
        iy = torch.floor(y).long().clamp(0, height - 1).cpu().numpy()
        mask_valid = torch.as_tensor(query_mask[iy, ix], dtype=torch.bool, device=device)
        candidate_valid[indices] = in_image & mask_valid
        for ref_frame, ref_camera in sorted(set(zip(frame["reference_frames"][indices_np],
                                                    frame["reference_cameras"][indices_np]))):
            local_np = indices_np[(frame["reference_frames"][indices_np] == ref_frame) &
                                  (frame["reference_cameras"][indices_np] == ref_camera)]
            local_indices = torch.as_tensor(local_np, dtype=torch.long, device=device)
            ref_view = reference_view(rows_by_frame, ref_frame, ref_camera)
            ref_image = load_rgb(ref_view["image"], device)
            uv = torch.as_tensor(frame["reference_pixels"][local_np], dtype=torch.float32, device=device)
            ref_patches[local_indices] = crop_batch(ref_image, uv, REFERENCE_PATCH_SIZE)

    center = SEARCH_RADIUS
    candidate_valid[:, center, center] |= ~candidate_valid.flatten(1).any(dim=1)
    return ref_patches, query_patches, candidate_valid


def build_ro_ma_channels(frame, device):
    n = len(frame["points"])
    radius = SEARCH_RADIUS
    shifts = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
    shift_grid = torch.stack((shift_x, shift_y), dim=-1)
    relative = (torch.as_tensor(frame["peaks"] - frame["centers"],
                                dtype=torch.float32, device=device)[:, None, None, :] +
                shift_grid[None])
    grid = torch.stack((2 * (relative[..., 0] + 12.) / 24. - 1,
                        2 * (relative[..., 1] + 12.) / 24. - 1), dim=-1)
    maps = torch.as_tensor(frame["cost_maps"], dtype=torch.float32, device=device)[:, None]
    valid = torch.as_tensor(frame["map_valid"], dtype=torch.float32, device=device)[:, None]
    costs = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
    support = F.grid_sample(valid, grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0] > .5
    in_map = ((relative[..., 0] >= -12.) & (relative[..., 0] <= 12.) &
              (relative[..., 1] >= -12.) & (relative[..., 1] <= 12.))
    support &= in_map
    similarity = torch.exp(-5. * torch.clamp(costs, min=0., max=2.))
    similarity = torch.where(support, similarity, torch.zeros_like(similarity))
    peak_score = torch.as_tensor(frame["peak_scores"], dtype=torch.float32, device=device)
    peak_score = peak_score[:, None, None].expand(-1, 2 * radius + 1, 2 * radius + 1)
    dx = (shift_x / radius).expand(n, -1, -1)
    dy = (shift_y / radius).expand(n, -1, -1)
    return torch.stack((similarity, support.float(), peak_score, dx, dy), dim=1)


class PatchEncoder(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class HighResPatchPoseRefiner(nn.Module):
    def __init__(self, radius=SEARCH_RADIUS):
        super().__init__()
        self.radius = int(radius)
        if self.radius != SEARCH_RADIUS:
            raise ValueError("checkpoint radius differs from frozen experiment radius")
        self.encoder = PatchEncoder()
        self.fusion = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.normal_(self.fusion[-1].weight, mean=0., std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, reference_patches, query_patches, roma_channels, candidate_valid,
                return_features=False):
        n = len(reference_patches)
        ref = self.encoder(reference_patches)
        query = self.encoder(query_patches)
        ref_center = F.normalize(ref[:, :, REFERENCE_PATCH_SIZE // 2,
                                     REFERENCE_PATCH_SIZE // 2], dim=1)
        offsets = torch.arange(-self.radius, self.radius + 1,
                               dtype=query.dtype, device=query.device)
        shift_y, shift_x = torch.meshgrid(offsets, offsets, indexing="ij")
        center = QUERY_PATCH_SIZE // 2
        x = center + shift_x
        y = center + shift_y
        grid = torch.stack((2 * x / (QUERY_PATCH_SIZE - 1) - 1,
                            2 * y / (QUERY_PATCH_SIZE - 1) - 1), dim=-1)
        query_candidates = F.grid_sample(query, grid[None].expand(n, -1, -1, -1),
                                         mode="bilinear", padding_mode="border", align_corners=True)
        query_candidates = F.normalize(query_candidates, dim=1)
        correlation = (query_candidates * ref_center[:, :, None, None]).sum(dim=1, keepdim=True)
        features = torch.cat((correlation, roma_channels, candidate_valid[:, None].to(correlation.dtype)), dim=1)
        logits = self.fusion(features)[:, 0] / .25
        valid = candidate_valid.bool()
        invalid_rows = ~valid.flatten(1).any(dim=1)
        if invalid_rows.any():
            valid = valid.clone()
            center_index = self.radius
            valid[invalid_rows, center_index, center_index] = True
        logits = logits.masked_fill(~valid, -1e4)
        probabilities = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        delta = torch.stack(((probabilities * shift_x).sum(dim=(1, 2)),
                             (probabilities * shift_y).sum(dim=(1, 2))), dim=1)
        if return_features:
            flat_probability = probabilities.flatten(1)
            flat_query = query_candidates.flatten(2).transpose(1, 2)
            query_mean = torch.bmm(flat_probability[:, None], flat_query).squeeze(1)
            query_second = torch.bmm(flat_probability[:, None], flat_query.square()).squeeze(1)
            query_variance = (query_second - query_mean.square()).clamp_min(0.)
            flat_correlation = correlation.flatten(1)
            correlation_mean = (flat_probability * flat_correlation).sum(dim=1, keepdim=True)
            correlation_variance = (flat_probability *
                                    (flat_correlation - correlation_mean).square()).sum(
                                        dim=1, keepdim=True)
            flat_roma = roma_channels.flatten(2).transpose(1, 2)
            roma_mean = torch.bmm(flat_probability[:, None], flat_roma).squeeze(1)
            entropy = -(flat_probability * flat_probability.clamp_min(1e-12).log()).sum(
                dim=1, keepdim=True) / math.log(flat_probability.shape[1])
            normalized_x = shift_x / max(self.radius, 1)
            normalized_y = shift_y / max(self.radius, 1)
            mean_x = delta[:, 0, None, None] / max(self.radius, 1)
            mean_y = delta[:, 1, None, None] / max(self.radius, 1)
            cov_xx = (probabilities * (normalized_x - mean_x).square()).sum((1, 2))
            cov_yy = (probabilities * (normalized_y - mean_y).square()).sum((1, 2))
            cov_xy = (probabilities * (normalized_x - mean_x) *
                      (normalized_y - mean_y)).sum((1, 2))
            features = {
                "reference_descriptor": ref_center,
                "query_descriptor_mean": query_mean,
                "query_descriptor_variance": query_variance,
                "roma_mean": roma_mean,
                "probability_entropy": entropy,
                "correlation_moments": torch.cat((correlation_mean, correlation_variance), dim=1),
                "offset_covariance": torch.stack((cov_xx, cov_yy, cov_xy), dim=1),
                "normalized_delta": delta / max(self.radius, 1),
            }
            return delta, probabilities, features
        return delta, probabilities


def skew_torch(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vector.shape[:-1], 3, 3)


def rotation_exp_torch(vector):
    theta2 = (vector * vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    a = torch.where(theta2 > 1e-8, torch.sin(theta) / theta,
                    1. - theta2 / 6. + theta2 * theta2 / 120.)
    b = torch.where(theta2 > 1e-8, (1. - torch.cos(theta)) / theta2.clamp_min(1e-16),
                    .5 - theta2 / 24. + theta2 * theta2 / 720.)
    matrix = skew_torch(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + a[..., None] * matrix + b[..., None] * (matrix @ matrix)


def project_and_jacobian(points, delta, base_pose, camera_to_body, calibration):
    base_pose = base_pose.to(dtype=points.dtype, device=points.device)
    camera_to_body = camera_to_body.to(dtype=points.dtype, device=points.device)
    calibration = calibration.to(dtype=points.dtype, device=points.device)
    rotation = rotation_exp_torch(delta[3:]) @ base_pose[:3, :3]
    translation = base_pose[:3, 3] + delta[:3]
    a = points - translation
    r_wb_t = rotation.T
    r_bc_t = camera_to_body[:, :3, :3].transpose(1, 2)
    world_to_camera_body = r_bc_t @ r_wb_t[None]
    camera = (world_to_camera_body @ a.unsqueeze(-1)).squeeze(-1) - \
        (r_bc_t @ camera_to_body[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    homogeneous = (calibration @ camera.unsqueeze(-1)).squeeze(-1)
    denominator = homogeneous[:, 2].clamp_min(1e-8)
    pixels = homogeneous[:, :2] / denominator[:, None]
    row0 = (calibration[:, 0, :] * homogeneous[:, 2:3] -
            homogeneous[:, 0:1] * calibration[:, 2, :]) / denominator[:, None].square()
    row1 = (calibration[:, 1, :] * homogeneous[:, 2:3] -
            homogeneous[:, 1:2] * calibration[:, 2, :]) / denominator[:, None].square()
    projection_jacobian = torch.stack((row0, row1), dim=1)
    translation_jacobian = -world_to_camera_body
    rotation_jacobian = world_to_camera_body @ skew_torch(a)
    camera_jacobian = torch.cat((translation_jacobian, rotation_jacobian), dim=2)
    jacobian = projection_jacobian @ camera_jacobian
    return pixels, jacobian


def differentiable_pose_solve(points, pixels, precision, camera_to_body, calibration,
                              baseline_pose, lidar_information, steps=POSE_STEPS):
    dtype = torch.float64
    points = points.to(dtype=dtype)
    pixels = pixels.to(dtype=dtype)
    precision = precision.to(dtype=dtype)
    camera_to_body = camera_to_body.to(dtype=dtype)
    calibration = calibration.to(dtype=dtype)
    baseline_pose = baseline_pose.to(dtype=dtype)
    information = lidar_information.to(dtype=dtype)
    delta = torch.zeros(6, dtype=dtype, device=points.device)
    eye = torch.eye(6, dtype=dtype, device=points.device)
    bounds = torch.tensor([.1, .1, .1, math.radians(1.), math.radians(1.), math.radians(1.)],
                          dtype=dtype, device=points.device)
    for _ in range(int(steps)):
        projected, jacobian = project_and_jacobian(points, delta, baseline_pose,
                                                   camera_to_body, calibration)
        residual = projected - pixels
        hessian = torch.einsum("nki,nkl,nlj->ij", jacobian, precision, jacobian) + information
        gradient = torch.einsum("nki,nkl,nl->i", jacobian, precision, residual) + information @ delta
        damping = (hessian.diagonal().mean().clamp_min(1e-8) * 1e-8)
        step = torch.linalg.solve(hessian + damping * eye, gradient)
        delta = torch.maximum(torch.minimum(delta - step, bounds), -bounds)
    rotation = rotation_exp_torch(delta[3:]) @ baseline_pose[:3, :3]
    pose = torch.eye(4, dtype=dtype, device=points.device)
    pose = pose.clone()
    pose[:3, :3] = rotation
    pose[:3, 3] = baseline_pose[:3, 3] + delta[:3]
    return pose, delta


def frame_geometry_tensors(frame, row, device):
    n = len(frame["points"])
    cameras = frame["cameras"].astype(np.int64)
    camera_to_body = np.empty((n, 4, 4), dtype=np.float64)
    calibration = np.empty((n, 3, 3), dtype=np.float64)
    for camera in np.unique(cameras):
        indices = np.where(cameras == camera)[0]
        view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
        camera_to_body[indices] = np.asarray(view["camera_to_body"], dtype=np.float64)
        calibration[indices] = np.loadtxt(view["calibration"]).astype(np.float64)
    return {
        "points": torch.as_tensor(frame["points"], dtype=torch.float64, device=device),
        "camera_to_body": torch.as_tensor(camera_to_body, dtype=torch.float64, device=device),
        "calibration": torch.as_tensor(calibration, dtype=torch.float64, device=device),
        "baseline_pose": torch.as_tensor(frame["baseline_pose"], dtype=torch.float64, device=device),
        "lidar_information": torch.as_tensor(frame["lidar_information"], dtype=torch.float64, device=device),
        "precision": torch.as_tensor(frame["peak_precisions"], dtype=torch.float64, device=device),
    }
