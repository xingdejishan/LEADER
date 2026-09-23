import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def skew(value):
    x, y, z = value
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)), dtype=np.float64)


def apply_delta(pose, delta):
    from scipy.spatial.transform import Rotation

    result = pose.copy()
    result[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ pose[:3, :3]
    result[:3, 3] = pose[:3, 3] + delta[:3]
    return result


def project_world(points, pose, camera_to_body, calibration):
    camera_pose = pose @ camera_to_body
    camera_points = (points - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    homogeneous = camera_points @ calibration.T
    depth = camera_points[:, 2]
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-12)
    return pixels, depth


def pixel_pose_jacobian(points, pose, camera_to_body, calibration):
    base, depth = project_world(points, pose, camera_to_body, calibration)
    jacobian = np.empty((len(points), 2, 6), dtype=np.float64)
    for axis in range(6):
        delta = np.zeros(6, dtype=np.float64)
        delta[axis] = 1e-4 if axis < 3 else 1e-5
        shifted, _ = project_world(points, apply_delta(pose, delta), camera_to_body, calibration)
        jacobian[:, :, axis] = (shifted - base) / delta[axis]
    return base, depth, jacobian


def lidar_pose_information(pose_local, source, target):
    rotated = source @ pose_local[:3, :3].T
    residual = rotated + pose_local[:3, 3] - target
    distances = np.linalg.norm(residual, axis=1)
    keep = distances < .6
    if int(keep.sum()) < 6:
        raise RuntimeError("two-stage full-pool baseline has fewer than six final-threshold points")
    scaled = distances[keep] / .6
    weights = (1. - scaled ** 2) ** 2
    jacobian = np.concatenate((np.broadcast_to(np.eye(3), (int(keep.sum()), 3, 3)),
                               -np.stack([skew(value) for value in rotated[keep]])), axis=2)
    hessian = np.einsum("n,nki,nkj->ij", weights, jacobian, jacobian)
    regularization = 1e-6 * max(float(np.trace(hessian) / 6.), 1e-12)
    weighted_sse = float(np.sum(weights * np.sum(residual[keep] ** 2, axis=1)))
    scale = max(math.sqrt(weighted_sse / max(3. * float(weights.sum()) - 6., 1.)), .005)
    covariance = scale ** 2 * np.linalg.inv(hessian + regularization * np.eye(6))
    information = np.linalg.inv(.5 * (covariance + covariance.T))
    return information, {"support": int(keep.sum()), "residual_scale_m": scale,
                         "covariance_diagonal": np.diag(covariance).tolist()}


class RoMaFeatures:
    def __init__(self, device, setting, stride):
        import torch
        import torch.nn.functional as F
        from romav2 import RoMaV2

        self.torch = torch
        self.functional = F
        self.device = torch.device(device)
        self.model = RoMaV2(RoMaV2.Cfg(setting=setting))
        self.stride = stride
        torch.set_float32_matmul_precision("highest")

    def image_features(self, image_path):
        F = self.functional
        image = self.model._load_image(str(image_path))
        height, width = ((self.model.H_hr, self.model.W_hr) if self.model.H_hr is not None else
                         (self.model.H_lr, self.model.W_lr))
        resized = F.interpolate(image, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        with self.torch.inference_mode():
            feature = self.model.refiner_features(resized)[self.stride].float().permute(0, 3, 1, 2)
        return feature

    def model_sha256(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            tensor = value.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            if tensor.dtype == self.torch.bfloat16:
                tensor = tensor.to(self.torch.float32)
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def normalized_grid(self, pixels, image_hw, dtype):
        pixels = self.torch.as_tensor(pixels, dtype=dtype, device=self.device)
        height, width = image_hw
        return self.torch.stack((2 * (pixels[:, 0] + .5) / width - 1,
                                 2 * (pixels[:, 1] + .5) / height - 1), dim=-1)

    def similarity_maps(self, reference_features, query_features, reference_uv, reference_hw,
                        center_uv, query_hw, query_mask, depth, batch_size=24):
        torch = self.torch
        F = self.functional
        count = len(reference_uv)
        offsets_y, offsets_x = np.mgrid[-12:13, -12:13]
        offsets = np.column_stack((offsets_x.ravel(), offsets_y.ravel())).astype(np.float32)
        height, width = query_hw
        maps = np.full((count, len(offsets)), -np.inf, dtype=np.float32)
        valid_maps = np.zeros((count, len(offsets)), dtype=bool)
        query_mask = np.asarray(query_mask, dtype=bool)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            local_centers = np.asarray(center_uv[start:stop], dtype=np.float32)
            candidates = local_centers[:, None, :] + offsets[None]
            in_image = ((candidates[..., 0] >= 0) & (candidates[..., 0] < width - 1) &
                        (candidates[..., 1] >= 0) & (candidates[..., 1] < height - 1))
            mask_x = np.clip(np.floor(candidates[..., 0]).astype(np.int64), 0, width - 1)
            mask_y = np.clip(np.floor(candidates[..., 1]).astype(np.int64), 0, height - 1)
            valid = in_image & query_mask[mask_y, mask_x] & (depth[start:stop, None] > 0)
            valid &= (np.abs(offsets[None, :, 0]) <= 12.) & (np.abs(offsets[None, :, 1]) <= 12.)
            reference_grid = self.normalized_grid(reference_uv[start:stop], reference_hw,
                                                 reference_features.dtype)
            candidate_grid = self.normalized_grid(candidates.reshape(-1, 2), query_hw,
                                                  query_features.dtype)
            with torch.inference_mode():
                reference = F.grid_sample(reference_features, reference_grid[None, :, None],
                                           mode="bilinear", padding_mode="border", align_corners=False)
                query = F.grid_sample(query_features, candidate_grid[None, :, None],
                                      mode="bilinear", padding_mode="border", align_corners=False)
                reference = F.normalize(reference[0, :, :, 0].T, dim=1)
                query = F.normalize(query[0, :, :, 0].T, dim=1).reshape(stop - start, len(offsets), -1)
                similarity = (query * reference[:, None]).sum(dim=2).float().cpu().numpy()
            maps[start:stop] = similarity
            valid_maps[start:stop] = valid
        return maps.reshape(count, 25, 25), valid_maps.reshape(count, 25, 25), offsets.reshape(25, 25, 2)


def peak_quadratic(scores, valid, base_uv):
    from numpy.linalg import lstsq

    masked = np.where(valid, scores, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    peak_uv = base_uv + np.array((ix - 12., iy - 12.))
    precision = np.zeros((2, 2), dtype=np.float64)
    if 0 < ix < 24 and 0 < iy < 24 and valid[iy - 1:iy + 2, ix - 1:ix + 2].all():
        y, x = np.mgrid[-1:2, -1:2]
        design = np.column_stack((x.ravel() ** 2, y.ravel() ** 2, (x * y).ravel(),
                                  x.ravel(), y.ravel(), np.ones(9)))
        coeff, _, _, _ = lstsq(design, scores[iy - 1:iy + 2, ix - 1:ix + 2].ravel(), rcond=None)
        hessian = np.array(((2 * coeff[0], coeff[2]), (coeff[2], 2 * coeff[1])))
        eigenvalues, eigenvectors = np.linalg.eigh(-hessian)
        precision = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.)) @ eigenvectors.T
        if np.linalg.eigvalsh(precision).min() > 0:
            offset = -np.linalg.solve(hessian, coeff[3:5])
            if np.isfinite(offset).all() and np.max(np.abs(offset)) <= 1.:
                peak_uv += offset
    return peak_uv, precision, float(scores[iy, ix]), bool(np.linalg.eigvalsh(precision).min() > 0)


def bilinear_map_value(cost, valid, offset):
    x, y = float(offset[0] + 12.), float(offset[1] + 12.)
    outside = max(-x, 0.) ** 2 + max(x - 24., 0.) ** 2 + max(-y, 0.) ** 2 + max(y - 24., 0.) ** 2
    x = min(max(x, 0.), 24.)
    y = min(max(y, 0.), 24.)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, 24), min(y0 + 1, 24)
    dx, dy = x - x0, y - y0
    indices = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    weights = ((1. - dx) * (1. - dy), dx * (1. - dy), (1. - dx) * dy, dx * dy)
    values = []
    for index in indices:
        value = float(cost[index]) if valid[index] else 2.
        values.append(value)
    return float(sum(weight * value for weight, value in zip(weights, values)) + outside)


def optimize_pose(initial, points, cameras, views, maps, map_valid, centers, peaks, peak_precisions,
                  lidar_information, mode):
    from scipy.optimize import minimize

    camera_data = {int(view["camera"]): (np.asarray(view["camera_to_body"], dtype=np.float64),
                                           np.loadtxt(view["calibration"]).astype(np.float64)) for view in views}
    index_by_camera = {camera: np.where(cameras == camera)[0] for camera in np.unique(cameras)}
    def objective(delta):
        pose = apply_delta(initial, delta)
        total = .5 * float(delta @ lidar_information @ delta)
        for camera, indices in index_by_camera.items():
            if mode == "geometry":
                continue
            camera_to_body, calibration = camera_data[int(camera)]
            pixels, depth = project_world(points[indices], pose, camera_to_body, calibration)
            if mode == "scoremap":
                for local_index, global_index in enumerate(indices):
                    value = bilinear_map_value(maps[global_index], map_valid[global_index],
                                               pixels[local_index] - centers[global_index])
                    total += value
            else:
                for local_index, global_index in enumerate(indices):
                    error = pixels[local_index] - peaks[global_index]
                    total += .5 * float(error @ peak_precisions[global_index] @ error)
                    if depth[local_index] <= 0:
                        total += 2.
        return total

    bounds = [(-.1, .1)] * 3 + [(-math.radians(1.), math.radians(1.))] * 3
    result = minimize(objective, np.zeros(6), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8})
    return apply_delta(initial, result.x), result, float(objective(np.zeros(6))), float(result.fun)


def build_baseline(row, lidar_cache, matcher, full_pool_refine, device, seed):
    import torch

    with np.load(Path(lidar_cache) / (row["frame_id"] + ".npz")) as data:
        source_np = np.asarray(data["source"], dtype=np.float32)
        prediction_np = np.asarray(data["prediction"], dtype=np.float32)
        center = np.asarray(data["center"], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(source_np.tobytes())
    digest.update(prediction_np.tobytes())
    digest.update(center.tobytes())
    source = torch.as_tensor(source_np, dtype=torch.float32, device=device)
    prediction = torch.as_tensor(prediction_np, dtype=torch.float32, device=device)
    keep_count = max(min(50, len(prediction_np)), int(.5 * len(prediction_np)))
    torch.manual_seed(seed)
    keep = prediction[:, 3].topk(keep_count).indices
    initial = matcher.estimator(source[keep][None], prediction[keep, :3][None])[0]
    output = full_pool_refine(initial, source, prediction[:, :3])
    refined = output[0] if isinstance(output, tuple) else output
    support = int(output[1]) if isinstance(output, tuple) and len(output) > 1 else None
    pose_local = refined.detach().cpu().numpy().astype(np.float64)
    pose = pose_local.copy()
    pose[:3, 3] += center
    target_local = prediction_np[:, :3].astype(np.float64)
    information, lidar_details = lidar_pose_information(pose_local, source_np.astype(np.float64), target_local)
    return pose, pose_local, source_np.astype(np.float64), target_local, information, lidar_details, digest.hexdigest(), support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--baseline-code-root", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/glace-local/code")
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--match-cache-dir", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/"
                                                       "research/prevoxel_multiview/results/roma_controlled_validation_matches"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--feature-stride", type=int, default=4)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    started = time.time()
    repo = Path(args.repo_root)
    sys.path.insert(0, str(repo))
    import torch
    import importlib.util

    baseline_code = Path(args.baseline_code_root)
    sys.path.insert(0, str(baseline_code))
    matcher_spec = importlib.util.spec_from_file_location("scoremap_baseline_matcher",
                                                          baseline_code / "models" / "sc2pcr.py")
    matcher_module = importlib.util.module_from_spec(matcher_spec)
    matcher_spec.loader.exec_module(matcher_module)
    pool_spec = importlib.util.spec_from_file_location("scoremap_baseline_full_pool",
                                                       baseline_code / "tools" / "full_pool_robust_v1.py")
    pool_module = importlib.util.module_from_spec(pool_spec)
    pool_spec.loader.exec_module(pool_module)
    full_pool_refine = pool_module.full_pool_refine
    torch.set_num_threads(16)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    eval_rows = [row for row in rows if row["split"] in ("val", "validation")]
    match_cache_dir = Path(args.match_cache_dir)
    eval_rows = [row for row in eval_rows if (match_cache_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if args.frames:
        eval_rows = eval_rows[:args.frames]
    if len(eval_rows) != args.frames and args.frames:
        raise RuntimeError("requested smoke frame count does not match available caches")
    if not args.frames and len(eval_rows) != args.expected_frames:
        raise RuntimeError("frozen full-run denominator differs from expected frame count")
    if not eval_rows:
        raise RuntimeError("no validation rows have frozen match caches")
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                                     nms_radius=.1, max_points=3000, k1=30)
    features = RoMaFeatures(args.device, args.roma_setting, args.feature_stride)
    feature_model_sha256 = features.model_sha256()
    records = []
    lidar_input_hashes = {}
    for row_index, row in enumerate(eval_rows):
        frame_id = str(row["frame_id"])
        cache_path = match_cache_dir / (frame_id + ".npz")
        with np.load(cache_path) as cache:
            required = {"points", "pixels", "reference_pixels", "cameras", "reference_frames", "reference_cameras"}
            if not required.issubset(cache.files):
                raise RuntimeError("match cache is missing required fields: " + frame_id)
            points = np.asarray(cache["points"], dtype=np.float64)
            matched_pixels = np.asarray(cache["pixels"], dtype=np.float64)
            reference_pixels = np.asarray(cache["reference_pixels"], dtype=np.float64)
            cameras = np.asarray(cache["cameras"], dtype=np.int64)
            reference_frames = np.asarray(cache["reference_frames"]).astype(str)
            reference_cameras = np.asarray(cache["reference_cameras"], dtype=np.int64)
        pose, pose_local, lidar_source, lidar_target, information, lidar_details, lidar_hash, baseline_support = build_baseline(
            row, args.lidar_cache, matcher, full_pool_refine, args.device, args.seed + row_index)
        lidar_input_hashes[frame_id] = lidar_hash
        query_views = {int(view["camera"]): view for view in row["views"]}
        maps = np.full((len(points), 25, 25), np.nan, dtype=np.float32)
        map_valid = np.zeros((len(points), 25, 25), dtype=bool)
        centers = np.full((len(points), 2), np.nan, dtype=np.float64)
        peaks = np.full((len(points), 2), np.nan, dtype=np.float64)
        peak_precisions = np.zeros((len(points), 2, 2), dtype=np.float64)
        keep_points = np.zeros(len(points), dtype=bool)
        peak_fits = 0
        boundary_peaks = 0
        map_digest = hashlib.sha256()
        match_offset_norms = []
        for camera in np.unique(cameras):
            camera_positions = np.where(cameras == camera)[0]
            query_view = query_views[int(camera)]
            from PIL import Image
            with Image.open(query_view["image"]) as image:
                query_hw = (image.height, image.width)
            query_mask = np.asarray(np.load(query_view["mask"]), dtype=bool)
            query_features = features.image_features(query_view["image"])
            camera_to_body = np.asarray(query_view["camera_to_body"], dtype=np.float64)
            calibration = np.loadtxt(query_view["calibration"]).astype(np.float64)
            base_uv, depth, jacobian = pixel_pose_jacobian(points[camera_positions], pose,
                                                           camera_to_body, calibration)
            match_offset_norms.extend(np.linalg.norm(matched_pixels[camera_positions] - base_uv, axis=1).tolist())
            for ref_frame, ref_camera in sorted(set(zip(reference_frames[camera_positions],
                                                        reference_cameras[camera_positions]))):
                positions = camera_positions[(reference_frames[camera_positions] == ref_frame) &
                                              (reference_cameras[camera_positions] == ref_camera)]
                local_positions = np.searchsorted(camera_positions, positions)
                reference_view = next(view for view in rows_by_frame[str(ref_frame)]["views"]
                                      if int(view["camera"]) == int(ref_camera))
                with Image.open(reference_view["image"]) as image:
                    reference_hw = (image.height, image.width)
                reference_features = features.image_features(reference_view["image"])
                local_maps, local_valid, _ = features.similarity_maps(
                    reference_features, query_features, reference_pixels[positions], reference_hw,
                    base_uv[local_positions], query_hw, query_mask, depth[local_positions])
                for local_index, global_index in enumerate(positions):
                    score_map = local_maps[local_index]
                    valid_map = local_valid[local_index]
                    score_map = score_map.astype(np.float32)
                    if not valid_map[12, 12] or not valid_map[11:14, 11:14].all():
                        continue
                    peak_uv, precision, best_score, fit_ok = peak_quadratic(
                        score_map, valid_map, base_uv[local_positions[local_index]])
                    cost_map = np.full((25, 25), 2., dtype=np.float32)
                    cost_map[valid_map] = np.maximum(best_score - score_map[valid_map], 0.)
                    maps[global_index] = cost_map
                    map_valid[global_index] = valid_map
                    centers[global_index] = base_uv[local_positions[local_index]]
                    peaks[global_index] = peak_uv
                    peak_precisions[global_index] = precision
                    keep_points[global_index] = True
                    peak_fits += int(fit_ok)
                    boundary_peaks += int(not fit_ok)
                    map_digest.update(cost_map.tobytes())
                    map_digest.update(valid_map.tobytes())
                del reference_features
            del query_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        points, cameras = points[keep_points], cameras[keep_points]
        maps, map_valid = maps[keep_points], map_valid[keep_points]
        centers, peaks, peak_precisions = centers[keep_points], peaks[keep_points], peak_precisions[keep_points]
        if len(points):
            score_pose, score_result, score_start, score_end = optimize_pose(
                pose, points, cameras, row["views"], maps, map_valid, centers, peaks,
                peak_precisions, information, "scoremap")
            peak_pose, peak_result, peak_start, peak_end = optimize_pose(
                pose, points, cameras, row["views"], maps, map_valid, centers, peaks,
                peak_precisions, information, "peak")
        else:
            score_pose = peak_pose = pose.copy()
            score_result = peak_result = None
            score_start = score_end = peak_start = peak_end = None
        geometry_pose = pose.copy()
        geometry_result = {"success": True, "status": 0, "nit": 0,
                           "message": "analytic minimum at zero pose correction"}
        geometry_start = geometry_end = 0.
        records.append({
            "frame_id": frame_id,
            "correspondences_in_cache": int(len(keep_points)),
            "usable_score_maps": int(keep_points.sum()),
            "active_cameras": int(len(np.unique(cameras))) if len(cameras) else 0,
            "peak_quadratic_fits": int(peak_fits),
            "peak_without_positive_curvature": int(boundary_peaks),
            "match_offset_median_px": float(np.median(match_offset_norms)) if match_offset_norms else None,
            "match_offset_p90_px": float(np.quantile(match_offset_norms, .9)) if match_offset_norms else None,
            "score_map_sha256": map_digest.hexdigest(),
            "lidar_information": lidar_details,
            "baseline_support": baseline_support,
            "baseline_pose": pose.tolist(),
            "geometry_only_pose": geometry_pose.tolist(),
            "peak_pose": peak_pose.tolist(),
            "scoremap_pose": score_pose.tolist(),
            "geometry_solver": {"success": geometry_result["success"], "status": geometry_result["status"],
                                "iterations": geometry_result["nit"], "message": geometry_result["message"],
                                "objective_initial": geometry_start, "objective_final": geometry_end},
            "peak_solver": None if peak_result is None else {"success": bool(peak_result.success),
                                                               "status": int(peak_result.status),
                                                               "iterations": int(peak_result.nit),
                                                               "message": str(peak_result.message),
                                                               "objective_initial": peak_start,
                                                               "objective_final": peak_end},
            "scoremap_solver": None if score_result is None else {"success": bool(score_result.success),
                                                                   "status": int(score_result.status),
                                                                   "iterations": int(score_result.nit),
                                                                   "message": str(score_result.message),
                                                                   "objective_initial": score_start,
                                                                   "objective_final": score_end},
        })
        print("scoremap %d/%d %s usable=%d cameras=%d map_obj=%.6f->%.6f peak_obj=%.6f->%.6f" % (
            row_index + 1, len(eval_rows), frame_id, int(keep_points.sum()),
            int(len(np.unique(cameras))) if len(cameras) else 0,
            score_start if score_start is not None else float("nan"),
            score_end if score_end is not None else float("nan"),
            peak_start if peak_start is not None else float("nan"),
            peak_end if peak_end is not None else float("nan")), flush=True)
    result = {
        "protocol": {
            "name": "local dense RoMa cost maps with shared pose refinement",
            "dataset": "2012-02-18 validation development set; not an independent sequence test",
            "baseline": "SC2-PCR followed by two-stage full-pool refinement at thresholds 1.2 m and 0.6 m",
            "correspondence_set": "frozen train-map RoMa v2 controlled validation cache; shared by peak and score-map paths",
            "visual_map": "cosine similarity of frozen RoMa precise stride-4 refiner features; local 25x25 pixel grid centered at baseline projection; no per-point LiDAR score",
            "peak_control": "one maximum per map, local quadratic curvature as the coordinate precision, then shared-pose reprojection optimization",
            "scoremap_method": "sum of per-map relative negative-log costs (maximum similarity minus sampled similarity) plus one LiDAR-derived 6D pose prior",
            "lidar_prior": "one covariance from the final 0.6 m full-pool Tukey evidence; no duplicate pixel-wise LiDAR prior",
            "geometry_control": "same 6D optimizer and LiDAR prior with visual factors disabled",
            "pose_update": "R=Exp(delta_rotation) R_A; t=t_A+delta_translation",
            "bounds": {"translation_each_axis_m": 0.1, "rotation_each_axis_deg": 1.0},
            "gt_in_runner": False,
        },
        "settings": vars(args),
        "inputs": {
            "manifest_sha256": sha256_file(args.manifest),
            "match_cache_sha256": {str(row["frame_id"]): sha256_file(match_cache_dir / (str(row["frame_id"]) + ".npz"))
                                    for row in eval_rows},
            "lidar_inputs_sha256": lidar_input_hashes,
            "runner_sha256": sha256_file(__file__),
            "branch_model_source_sha256": sha256_file(repo / "models" / "sc2pcr.py"),
            "baseline_matcher_source_sha256": sha256_file(baseline_code / "models" / "sc2pcr.py"),
            "full_pool_source_sha256": sha256_file(baseline_code / "tools" / "full_pool_robust_v1.py"),
            "roma_feature_state_sha256": feature_model_sha256,
        },
        "frames": len(eval_rows),
        "records": records,
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
