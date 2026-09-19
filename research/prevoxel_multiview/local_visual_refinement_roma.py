"""LEADER-guided RoMa v2 local correspondence refinement."""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement import visible_map_points
from oracle_pose_refinement import (
    bootstrap_ci,
    geometry_diagnostics,
    load_lidar,
    load_module,
    metrics,
    pose_error,
    pose_from_baseline,
    project_world,
)


def build_reference_observations(rows, lidar_cache, projection_cache,
                                 voxel_size, output, max_history):
    required = {"map_ids", "world_xyz", "frame_ids", "camera_ids", "ref_uv", "geometry_only"}
    if output.exists():
        with np.load(output) as data:
            if required.issubset(data.files):
                return {key: np.asarray(data[key]) for key in data.files}
    point_ids, observations = {}, defaultdict(list)
    train_rows = [row for row in rows if row["split"] == "train"]
    for index, row in enumerate(train_rows):
        cached = load_lidar(lidar_cache, row["frame_id"])
        with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as mapping:
            projection_xyz = np.asarray(mapping["projection_xyz"], dtype=np.float32)
            localization_xyz = np.asarray(mapping["localization_xyz"], dtype=np.float32)
        if not np.array_equal(localization_xyz, cached["source"]):
            raise ValueError("projection/localization order mismatch: %s" % row["frame_id"])
        world = projection_xyz @ cached["GT"][:3, :3].T + cached["GT"][:3, 3]
        map_for_raw = np.empty(len(world), dtype=np.int32)
        for raw_index, key_array in enumerate(np.floor(world / voxel_size).astype(np.int64)):
            key = tuple(int(value) for value in key_array)
            map_index = point_ids.get(key)
            if map_index is None:
                map_index = len(point_ids)
                point_ids[key] = map_index
            map_for_raw[raw_index] = map_index
        for view in row["views"]:
            camera = int(view["camera"])
            from PIL import Image
            image = np.asarray(Image.open(view["image"]).convert("RGB"))
            image_mask = np.asarray(np.load(view["mask"]))
            uv, raw_indices = visible_map_points(world, cached["GT"], view, image, image_mask)
            if not len(raw_indices):
                continue
            for raw_index in raw_indices:
                key = (int(map_for_raw[raw_index]), row["frame_id"], camera)
                if max_history <= 0 or len(observations[key]) < max_history:
                    observations[key].append((world[raw_index], uv[raw_index]))
        print("reference observations %d/%d" % (index + 1, len(train_rows)), flush=True)
    map_ids, frame_ids, camera_ids, world_xyz, ref_uv = [], [], [], [], []
    for (map_id, frame_id, camera_id), history in observations.items():
        for xyz, uv in history:
            map_ids.append(map_id)
            frame_ids.append(frame_id)
            camera_ids.append(camera_id)
            world_xyz.append(xyz)
            ref_uv.append(uv)
    result = {
        "map_ids": np.asarray(map_ids, dtype=np.int32),
        "frame_ids": np.asarray(frame_ids, dtype="U32"),
        "camera_ids": np.asarray(camera_ids, dtype=np.int8),
        "world_xyz": np.asarray(world_xyz, dtype=np.float32),
        "ref_uv": np.asarray(ref_uv, dtype=np.float32),
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
        "geometry_only": np.asarray(True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


class RoMaField:
    def __init__(self, device, setting):
        import torch
        from romav2 import RoMaV2

        self.torch = torch
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device(device)
        self.model = RoMaV2(RoMaV2.Cfg(setting=setting))
        self.cache = {}

    def match(self, ref_path, query_path):
        key = (str(ref_path), str(query_path))
        if key not in self.cache:
            self.cache[key] = self.model.match(ref_path, query_path)
        return self.cache[key]

    def clear_cache(self):
        self.cache.clear()

    def sample(self, prediction, ref_uv, ref_hw, query_hw):
        import torch.nn.functional as F

        ref_height, ref_width = ref_hw
        query_height, query_width = query_hw
        uv = self.torch.as_tensor(ref_uv, dtype=self.torch.float32, device=self.device)
        grid = self.torch.stack((2 * uv[:, 0] / ref_width, 2 * uv[:, 1] / ref_height), dim=-1) - 1
        grid = grid[None, :, None]
        def sample_bhwc(values):
            channels = values.shape[-1]
            result = F.grid_sample(values.permute(0, 3, 1, 2), grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=False)
            return result[0, :, :, 0].T.reshape(len(uv), *values.shape[3:]) if values.ndim > 4 else result[0, :, :, 0].T
        warp = sample_bhwc(prediction["warp_AB"])
        overlap = sample_bhwc(prediction["overlap_AB"]).reshape(-1)
        precision = sample_bhwc(prediction["precision_AB"].reshape(*prediction["precision_AB"].shape[:3], 4)).reshape(-1, 2, 2)
        model_height, model_width = prediction["precision_AB"].shape[1:3]
        ratio = self.torch.tensor([model_width / query_width, model_height / query_height], device=self.device)
        precision = precision * ratio[None, :, None] * ratio[None, None, :]
        query_uv = self.torch.stack(((warp[:, 0] + 1) * query_width / 2,
                                     (warp[:, 1] + 1) * query_height / 2), dim=-1)
        return (query_uv.detach().cpu().numpy(), overlap.detach().cpu().numpy(),
                precision.detach().cpu().numpy())


def stabilize_precision(precision, min_precision, max_precision):
    precision = .5 * (precision + np.swapaxes(precision, -1, -2))
    values, vectors = np.linalg.eigh(precision)
    values = np.clip(values, min_precision, max_precision)
    return (vectors * values[:, None, :]) @ np.swapaxes(vectors, -1, -2)


def apply_local_delta(initial, delta):
    from scipy.spatial.transform import Rotation

    pose = np.asarray(initial, dtype=np.float64).copy()
    pose[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ pose[:3, :3]
    pose[:3, 3] += delta[:3]
    return pose


def effective_precision(precisions, precision_scale, precision_floor_px):
    covariance = np.linalg.inv(precisions)
    covariance *= precision_scale ** 2
    covariance += (precision_floor_px ** 2) * np.eye(2)[None]
    return np.linalg.inv(.5 * (covariance + np.swapaxes(covariance, -1, -2)))


def reprojection_blocks(pose, points, pixels, cameras, camera_data):
    raw = np.empty((len(points), 2), dtype=np.float64)
    for camera in range(6):
        keep = cameras == camera
        if not keep.any():
            continue
        uv, depth = project_world(points[keep], pose, *camera_data[camera])
        raw[keep] = uv - pixels[keep]
        bad = (~np.isfinite(raw[keep]).all(axis=1)) | (depth <= 1e-4)
        raw[np.where(keep)[0][bad]] = 1000.
    return raw


def whitened_reprojection(pose, points, pixels, cameras, precisions, views,
                          precision_scale, precision_floor_px):
    camera_data = {int(view["camera"]): (np.asarray(view["camera_to_body"], dtype=np.float64),
                                           np.loadtxt(view["calibration"]).astype(np.float64)) for view in views}
    effective = effective_precision(precisions, precision_scale, precision_floor_px)
    cholesky = np.linalg.cholesky(effective)
    raw = reprojection_blocks(pose, points, pixels, cameras, camera_data)
    return np.einsum("nij,nj->ni", np.swapaxes(cholesky, 1, 2), raw)


def block_soft_l1_cost(whitened, robust_scale):
    norms = np.linalg.norm(whitened, axis=1)
    return float(np.sum(2 * robust_scale ** 2 * (np.sqrt(1 + (norms / robust_scale) ** 2) - 1)))


def refine_pose_protected(initial, points, pixels, cameras, precisions, views,
                          max_translation, max_rotation, max_nfev,
                          prior_sigma_translation, prior_sigma_rotation,
                          visual_lambda, precision_scale, precision_floor_px,
                          robust_scale, irls_iterations):
    from scipy.optimize import least_squares

    camera_data = {int(view["camera"]): (np.asarray(view["camera_to_body"], dtype=np.float64),
                                           np.loadtxt(view["calibration"]).astype(np.float64)) for view in views}
    effective = effective_precision(precisions, precision_scale, precision_floor_px)
    cholesky = np.linalg.cholesky(effective)
    visual_scale = math.sqrt(visual_lambda / max(len(points), 1))
    prior_scales = np.array([prior_sigma_translation] * 3 + [prior_sigma_rotation] * 3, dtype=np.float64)
    weights = np.ones(len(points), dtype=np.float64)
    bounds = np.concatenate([np.full(3, max_translation), np.full(3, max_rotation)])
    result = None
    for _ in range(irls_iterations):
        def residual(delta):
            raw = reprojection_blocks(apply_local_delta(initial, delta), points, pixels, cameras, camera_data)
            whitened = np.einsum("nij,nj->ni", np.swapaxes(cholesky, 1, 2), raw)
            visual = visual_scale * np.sqrt(weights)[:, None] * whitened
            return np.concatenate((delta / prior_scales, visual.reshape(-1)))
        result = least_squares(residual, np.zeros(6) if result is None else result.x,
                               bounds=(-bounds, bounds), method="trf", loss="linear", max_nfev=max_nfev)
        raw = reprojection_blocks(apply_local_delta(initial, result.x), points, pixels, cameras, camera_data)
        whitened = np.einsum("nij,nj->ni", np.swapaxes(cholesky, 1, 2), raw)
        norms = np.linalg.norm(whitened, axis=1)
        weights = 1. / np.sqrt(1. + (norms / robust_scale) ** 2)
    return apply_local_delta(initial, result.x), result, whitened, effective, weights


def query_matches(row, initial, references, rows_by_frame, lidar_cache, roma, crop_radius, local_radius,
                  min_overlap, max_reference_images, grid_cell, max_per_camera,
                  min_precision, max_precision, min_view_cosine, gate_free_cache=False):
    all_points, all_pixels, all_cameras, all_scores, all_precisions = [], [], [], [], []
    all_anchor_ids, all_reference_frames, all_reference_cameras, diagnostics = [], [], [], []
    for query_view in sorted(row["views"], key=lambda value: value["camera"]):
        from PIL import Image
        query_image = np.asarray(Image.open(query_view["image"]).convert("RGB"))
        query_mask = np.asarray(np.load(query_view["mask"]))
        query_height, query_width = query_image.shape[:2]
        observation_world = references["world_xyz"].astype(np.float64)
        local_observations = np.where(np.linalg.norm(observation_world - initial[:3, 3], axis=1) <= crop_radius)[0]
        _, visible_local = visible_map_points(observation_world[local_observations], initial, query_view, query_image, query_mask)
        if not len(visible_local):
            diagnostics.append({"camera": int(query_view["camera"]), "visible_map_points": 0, "reference_images": 0, "accepted": 0})
            continue
        record_rows = local_observations[visible_local]
        record_rows = record_rows[references["frame_ids"][record_rows] != row["frame_id"]]
        pair_groups = defaultdict(list)
        for record in record_rows:
            pair_groups[(str(references["frame_ids"][record]), int(references["camera_ids"][record]))].append(record)
        query_camera_center = (initial @ np.asarray(query_view["camera_to_body"], dtype=np.float64))[:3, 3]
        pair_scores = {}
        for pair, pair_rows in pair_groups.items():
            reference_frame, reference_camera = pair
            reference_row = rows_by_frame[reference_frame]
            reference_view = next(view for view in reference_row["views"] if int(view["camera"]) == reference_camera)
            reference_pose = load_lidar(lidar_cache, reference_frame)["GT"]
            reference_center = (reference_pose @ np.asarray(reference_view["camera_to_body"], dtype=np.float64))[:3, 3]
            anchor = references["world_xyz"][np.asarray(pair_rows, dtype=np.int64)].astype(np.float64)
            query_ray = anchor - query_camera_center
            reference_ray = anchor - reference_center
            query_ray /= np.maximum(np.linalg.norm(query_ray, axis=1, keepdims=True), 1e-9)
            reference_ray /= np.maximum(np.linalg.norm(reference_ray, axis=1, keepdims=True), 1e-9)
            compatible = (query_ray * reference_ray).sum(axis=1) >= min_view_cosine
            pair_scores[pair] = int(compatible.sum())
        selected_pairs = sorted(pair_groups, key=lambda pair: (-pair_scores[pair], -len(pair_groups[pair]), pair))[:max_reference_images]
        candidates, stage_counts = [], defaultdict(int)
        for reference_frame, reference_camera in selected_pairs:
            ref_row = rows_by_frame[reference_frame]
            ref_view = next(view for view in ref_row["views"] if int(view["camera"]) == reference_camera)
            ref_rows = np.asarray(pair_groups[(reference_frame, reference_camera)], dtype=np.int64)
            ref_uv = references["ref_uv"][ref_rows]
            with Image.open(ref_view["image"]) as image:
                ref_width, ref_height = image.size
            prediction = roma.match(ref_view["image"], query_view["image"])
            query_uv, overlap, precision = roma.sample(prediction, ref_uv, (ref_height, ref_width), (query_height, query_width))
            world = references["world_xyz"][ref_rows].astype(np.float64)
            base, base_depth = project_world(world, initial, np.asarray(query_view["camera_to_body"], dtype=np.float64),
                                             np.loadtxt(query_view["calibration"]).astype(np.float64))
            integer = np.rint(query_uv).astype(np.int64)
            valid = np.isfinite(query_uv).all(axis=1) & np.isfinite(precision).all(axis=(1, 2))
            stage_counts["finite"] += int(valid.sum())
            valid &= (query_uv[:, 0] >= 0) & (query_uv[:, 0] < query_width) & (query_uv[:, 1] >= 0) & (query_uv[:, 1] < query_height)
            stage_counts["in_image"] += int(valid.sum())
            integer[:, 0] = np.clip(integer[:, 0], 0, query_width - 1)
            integer[:, 1] = np.clip(integer[:, 1], 0, query_height - 1)
            valid &= query_mask[integer[:, 1], integer[:, 0]] > 0
            stage_counts["mask"] += int(valid.sum())
            valid &= query_image[integer[:, 1], integer[:, 0]].max(axis=1) > 10
            stage_counts["nonblack"] += int(valid.sum())
            valid &= np.isfinite(base).all(axis=1) & (base_depth > .5)
            stage_counts["observation_projection"] += int(valid.sum())
            if not gate_free_cache:
                valid &= np.abs(query_uv - base).max(axis=1) <= local_radius
                stage_counts["local"] += int(valid.sum())
                valid &= overlap >= min_overlap
                stage_counts["overlap"] += int(valid.sum())
            precision = stabilize_precision(precision, min_precision, max_precision)
            for position in np.where(valid)[0]:
                candidates.append((float(overlap[position]), world[position], query_uv[position], precision[position],
                                   int(references["map_ids"][ref_rows[position]]), reference_frame, reference_camera))
        if gate_free_cache:
            selected = candidates
        else:
            candidates.sort(key=lambda value: -value[0])
            occupied, selected = set(), []
            for candidate in candidates:
                cell = tuple(np.floor(candidate[2] / grid_cell).astype(np.int64))
                if cell in occupied:
                    continue
                occupied.add(cell)
                selected.append(candidate)
                if max_per_camera > 0 and len(selected) >= max_per_camera:
                    break
        if selected:
            all_scores.extend(value[0] for value in selected)
            all_points.append(np.asarray([value[1] for value in selected]))
            all_pixels.append(np.asarray([value[2] for value in selected]))
            all_precisions.append(np.asarray([value[3] for value in selected]))
            all_cameras.append(np.full(len(selected), int(query_view["camera"]), dtype=np.int64))
            all_anchor_ids.append(np.asarray([value[4] for value in selected], dtype=np.int64))
            all_reference_frames.append(np.asarray([value[5] for value in selected], dtype="U32"))
            all_reference_cameras.append(np.asarray([value[6] for value in selected], dtype=np.int8))
        diagnostics.append({"camera": int(query_view["camera"]), "visible_observations": int(len(record_rows)),
                            "reference_images": int(len(selected_pairs)), "raw_candidates": int(len(candidates)),
                            "accepted": int(len(selected)), "stages": dict(stage_counts),
                            "reference_pair_covisible_anchors": {"%s:%d" % pair: pair_scores[pair] for pair in selected_pairs},
                            "reference_pairs": [list(pair) for pair in selected_pairs]})
    if not all_points:
        return (np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int64), np.empty(0),
                np.empty((0, 2, 2)), np.empty(0, dtype=np.int64), np.empty(0, dtype="U32"),
                np.empty(0, dtype=np.int8), diagnostics)
    return (np.concatenate(all_points), np.concatenate(all_pixels), np.concatenate(all_cameras),
            np.asarray(all_scores), np.concatenate(all_precisions), np.concatenate(all_anchor_ids),
            np.concatenate(all_reference_frames), np.concatenate(all_reference_cameras), diagnostics)


def match_cache_path(cache_dir, frame_id):
    return Path(cache_dir) / (frame_id + ".npz")


def save_match_cache(path, points, pixels, cameras, scores, precisions, anchor_ids,
                     reference_frames, reference_cameras, diagnostics):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, pixels=pixels, cameras=cameras, scores=scores,
                        precisions=precisions, anchor_ids=anchor_ids,
                        reference_frames=reference_frames, reference_cameras=reference_cameras,
                        diagnostics_json=np.asarray(json.dumps(diagnostics)))


def load_match_cache(path):
    with np.load(path) as data:
        return (np.asarray(data["points"]), np.asarray(data["pixels"]), np.asarray(data["cameras"]),
                np.asarray(data["scores"]), np.asarray(data["precisions"]), np.asarray(data["anchor_ids"]),
                np.asarray(data["reference_frames"]), np.asarray(data["reference_cameras"]),
                json.loads(str(data["diagnostics_json"])))


def holdout_mask(group_ids, modulus, min_count=0):
    if modulus < 2:
        return np.zeros(len(group_ids), dtype=bool)
    unique, inverse, counts = np.unique(np.asarray(group_ids).astype(str), return_inverse=True, return_counts=True)
    hashes = np.asarray([int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little") for value in unique], dtype=np.uint64)
    order = np.argsort(hashes)
    selected = np.zeros(len(unique), dtype=bool)
    count = 0
    for group in order:
        if count >= max(int(math.ceil(len(group_ids) / modulus)), min_count):
            break
        selected[group] = True
        count += int(counts[group])
    return selected[inverse]


def reference_pair_ids(query_cameras, reference_frames, reference_cameras):
    return np.asarray(["%d|%s|%d" % (camera, frame, reference_camera) for camera, frame, reference_camera in
                       zip(query_cameras, reference_frames, reference_cameras)], dtype="U64")


def gate_diagnostics(records):
    rows, good, bad = [], [], []
    for record in records:
        delta = np.asarray(record["candidate_delta"], dtype=np.float64)
        acceptance = record["acceptance"]
        holdout_before = acceptance["holdout_cost_before"]
        holdout_candidate = acceptance["holdout_cost_candidate"]
        holdout_improvement = float("nan")
        if np.isfinite(holdout_before + holdout_candidate) and holdout_before > 0:
            holdout_improvement = 1. - holdout_candidate / holdout_before
        label = "mixed"
        if np.isfinite(delta).all() and (delta < 0).all():
            label, good = "both_improved", good + [acceptance["accepted"]]
        elif np.isfinite(delta).all() and (delta > 0).any():
            label, bad = "any_degraded", bad + [acceptance["accepted"]]
        rows.append({"frame_id": record["frame_id"], "label_from_train_gt": label,
                     "candidate_delta_translation_m_rotation_deg": delta.tolist(),
                     "holdout_improvement_ratio": holdout_improvement,
                     "accepted": acceptance["accepted"], "acceptance_reason": acceptance["reason"]})
    return {"good_candidates": len(good), "bad_candidates": len(bad),
            "good_acceptance_rate": float(np.mean(good)) if good else float("nan"),
            "bad_rejection_rate": float(1. - np.mean(bad)) if bad else float("nan"), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise", choices=("turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast"))
    parser.add_argument("--evaluate-split", default="validation", choices=("train", "validation"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--local-radius", type=float, default=8.)
    parser.add_argument("--min-overlap", type=float, default=.2)
    parser.add_argument("--max-reference-images", type=int, default=2)
    parser.add_argument("--min-reference-view-cosine", type=float, default=.7)
    parser.add_argument("--grid-cell", type=int, default=4)
    parser.add_argument("--max-per-camera", type=int, default=300)
    parser.add_argument("--min-precision", type=float, default=1e-4)
    parser.add_argument("--max-precision", type=float, default=100.)
    parser.add_argument("--max-translation", type=float, default=2.)
    parser.add_argument("--max-rotation-deg", type=float, default=10.)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--prior-sigma-translation-m", type=float, default=.10)
    parser.add_argument("--prior-sigma-rotation-deg", type=float, default=1.)
    parser.add_argument("--visual-lambda", type=float, default=1.)
    parser.add_argument("--precision-scale", type=float, default=1.)
    parser.add_argument("--precision-floor-px", type=float, default=1.)
    parser.add_argument("--robust-scale-whitened", type=float, default=1.)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-group", default="reference-pair", choices=("reference-pair", "anchor"))
    parser.add_argument("--min-holdout", type=int, default=6)
    parser.add_argument("--holdout-accept-ratio", type=float, default=.95)
    parser.add_argument("--match-cache-dir")
    parser.add_argument("--replay-match-cache", action="store_true")
    parser.add_argument("--gate-free-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if min(args.prior_sigma_translation_m, args.prior_sigma_rotation_deg, args.visual_lambda,
           args.precision_scale, args.precision_floor_px, args.robust_scale_whitened) <= 0:
        parser.error("prior, precision, visual, and robust scales must be positive")
    if args.irls_iterations < 1 or args.min_holdout < 1 or not 0 < args.holdout_accept_ratio <= 1:
        parser.error("IRLS iterations and minimum holdout must be positive; holdout ratio must be in (0, 1]")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] in ("val", "validation", "test")]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
        rows = train_rows + val_rows
    eval_rows = train_rows if args.evaluate_split == "train" else val_rows
    if args.frames:
        eval_rows = eval_rows[:args.frames]
    references = build_reference_observations(rows, args.lidar_cache, args.projection_cache,
                                              args.map_voxel_size, Path(args.map_cache), args.max_history)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("roma_local_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("roma_local_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    roma = None if args.replay_match_cache else RoMaField(args.device, args.roma_setting)
    records, started = [], time.time()
    for index, row in enumerate(eval_rows):
        initial, gt, support = pose_from_baseline(row, args.lidar_cache, matcher, full_pool_module.full_pool_refine,
                                                  args.device, args.seed + index)
        before = pose_error(initial, gt)
        cache_path = match_cache_path(args.match_cache_dir, row["frame_id"]) if args.match_cache_dir else None
        if args.replay_match_cache:
            if cache_path is None or not cache_path.exists():
                raise FileNotFoundError("missing match cache for %s" % row["frame_id"])
            points, pixels, cameras, scores, precisions, anchor_ids, reference_frames, reference_cameras, matching = load_match_cache(cache_path)
        else:
            points, pixels, cameras, scores, precisions, anchor_ids, reference_frames, reference_cameras, matching = query_matches(
                row, initial, references, rows_by_frame, args.lidar_cache, roma, args.crop_radius, args.local_radius, args.min_overlap,
                args.max_reference_images, args.grid_cell, args.max_per_camera, args.min_precision, args.max_precision,
                args.min_reference_view_cosine, args.gate_free_cache)
            if cache_path is not None:
                save_match_cache(cache_path, points, pixels, cameras, scores, precisions, anchor_ids,
                                 reference_frames, reference_cameras, matching)
        candidate = initial.copy()
        candidate_after = before
        accepted = False
        acceptance_reason = "insufficient_correspondences"
        solver = {"success": False, "status": -1, "nfev": 0, "cost": float("nan"),
                  "whitened_residual_rmse": float("nan"), "irls_iterations": 0}
        group_ids = reference_pair_ids(cameras, reference_frames, reference_cameras) if args.holdout_group == "reference-pair" else anchor_ids
        holdout = holdout_mask(group_ids, args.holdout_modulus, args.min_holdout)
        fit = ~holdout
        holdout_before = holdout_after = float("nan")
        correction_translation = correction_rotation = float("nan")
        if not args.cache_only and len(points) >= 6 and len(np.unique(cameras)) and int(fit.sum()) >= 6 and len(np.unique(cameras[fit])):
            candidate, optimizer, fit_residual, _, _ = refine_pose_protected(
                initial, points[fit], pixels[fit], cameras[fit], precisions[fit], row["views"], args.max_translation,
                math.radians(args.max_rotation_deg), args.max_nfev, args.prior_sigma_translation_m,
                math.radians(args.prior_sigma_rotation_deg), args.visual_lambda, args.precision_scale,
                args.precision_floor_px, args.robust_scale_whitened, args.irls_iterations)
            delta = np.asarray(optimizer.x, dtype=np.float64)
            if np.isfinite(candidate).all():
                candidate_after = pose_error(candidate, gt)
            else:
                candidate_after = (float("nan"), float("nan"))
            correction_translation = float(np.linalg.norm(candidate[:3, 3] - initial[:3, 3]))
            correction_rotation = float(np.linalg.norm(delta[3:]))
            solver = {"success": bool(optimizer.success), "status": int(optimizer.status), "nfev": int(optimizer.nfev),
                      "cost": float(optimizer.cost), "whitened_residual_rmse": float(np.sqrt(np.mean(fit_residual ** 2))),
                      "irls_iterations": args.irls_iterations}
            if int(holdout.sum()) >= args.min_holdout:
                holdout_before = block_soft_l1_cost(whitened_reprojection(
                    initial, points[holdout], pixels[holdout], cameras[holdout], precisions[holdout], row["views"],
                    args.precision_scale, args.precision_floor_px), args.robust_scale_whitened)
                holdout_after = block_soft_l1_cost(whitened_reprojection(
                    candidate, points[holdout], pixels[holdout], cameras[holdout], precisions[holdout], row["views"],
                    args.precision_scale, args.precision_floor_px), args.robust_scale_whitened)
            if not optimizer.success:
                acceptance_reason = "solver_failed"
            elif not np.isfinite(candidate).all() or not np.isfinite(delta).all():
                acceptance_reason = "nonfinite_candidate"
            elif correction_translation > args.max_translation + 1e-9:
                acceptance_reason = "translation_bound"
            elif correction_rotation > math.radians(args.max_rotation_deg) + 1e-9:
                acceptance_reason = "rotation_bound"
            elif int(holdout.sum()) < args.min_holdout:
                acceptance_reason = "insufficient_holdout"
            elif not np.isfinite(holdout_before + holdout_after) or holdout_after > args.holdout_accept_ratio * holdout_before:
                acceptance_reason = "holdout_not_improved"
            else:
                accepted, acceptance_reason = True, "accepted"
        elif len(points) and int(holdout.sum()) >= len(points) - 5:
            acceptance_reason = "insufficient_fit_correspondences"
        refined = candidate if accepted else initial.copy()
        after = candidate_after if accepted else before
        acceptance = {"accepted": accepted, "reason": acceptance_reason, "fit_correspondences": int(fit.sum()),
                      "holdout_correspondences": int(holdout.sum()), "holdout_cost_before": holdout_before,
                      "holdout_cost_candidate": holdout_after, "correction_translation_m": correction_translation,
                      "correction_rotation_deg": math.degrees(correction_rotation) if np.isfinite(correction_rotation) else float("nan"),
                      "fit_groups": int(len(np.unique(group_ids[fit]))), "holdout_groups": int(len(np.unique(group_ids[holdout]))),
                      "holdout_group_ids": np.unique(group_ids[holdout]).astype(str).tolist()}
        record = {"frame_id": row["frame_id"], "before": list(before), "after": list(after),
                  "candidate_after": list(candidate_after), "delta": [after[0] - before[0], after[1] - before[1]],
                  "candidate_delta": [candidate_after[0] - before[0], candidate_after[1] - before[1]],
                  "leader_success": bool(before[0] < 1. and before[1] < 2.), "n_correspondences": int(len(points)),
                  "n_cameras": int(len(np.unique(cameras))) if len(cameras) else 0,
                  "overlap_mean": float(scores.mean()) if len(scores) else float("nan"),
                  "per_camera_correspondences": [int((cameras == camera).sum()) for camera in range(6)],
                  "matching": matching, "solver": solver, "baseline_support": support,
                  "geometry": geometry_diagnostics(points, cameras, initial, row["views"]),
                  "leader_pose": initial.tolist(), "candidate_pose": candidate.tolist(), "refined_pose": refined.tolist(),
                  "acceptance": acceptance, "gt_pose": gt.tolist()}
        records.append(record)
        print("%s %d/%d %s cams=%d corr=%d overlap=%.3f accepted=%s before=(%.3f,%.3f) after=(%.3f,%.3f)" %
              (args.evaluate_split, index + 1, len(eval_rows), row["frame_id"], record["n_cameras"], len(points), record["overlap_mean"],
               accepted, before[0], before[1], after[0], after[1]), flush=True)
        if roma is not None:
            roma.clear_cache()
    success = [record for record in records if record["leader_success"]]
    paired, paired_success = np.asarray([record["delta"] for record in records]), np.asarray([record["delta"] for record in success])
    correspondence_counts = np.asarray([record["n_correspondences"] for record in records], dtype=np.int64)
    active_camera_counts = np.asarray([record["n_cameras"] for record in records], dtype=np.int64)
    translation_improved = paired[:, 0] < 0 if len(paired) else np.empty(0, dtype=bool)
    rotation_improved = paired[:, 1] < 0 if len(paired) else np.empty(0, dtype=bool)
    result = {"protocol": {"map_source": "train split only", "front_end": "RoMa v2 dense reference-to-query fields",
                             "matching": {"max_reference_images_per_query_camera": args.max_reference_images,
                                          "local_radius_px": args.local_radius, "min_overlap": args.min_overlap,
                                          "min_reference_view_cosine": args.min_reference_view_cosine,
                                          "grid_cell_px": args.grid_cell, "max_per_camera": args.max_per_camera,
                                          "gate_free_cache": args.gate_free_cache, "cache_only": args.cache_only},
                             "optimizer": "LiDAR-prior-protected block-IRLS Trust Region Reflective optimization",
                             "local_pose_update": "R=Exp(delta_rotation) R_LEADER; t=t_LEADER+delta_translation",
                             "prior": {"sigma_translation_m": args.prior_sigma_translation_m,
                                       "sigma_rotation_deg": args.prior_sigma_rotation_deg,
                                       "visual_lambda": args.visual_lambda},
                             "precision": {"scale": args.precision_scale, "floor_px": args.precision_floor_px},
                             "robust_scale_whitened": args.robust_scale_whitened,
                             "acceptance": {"holdout_group": args.holdout_group, "holdout_modulus": args.holdout_modulus,
                                            "min_holdout": args.min_holdout, "maximum_holdout_cost_ratio": args.holdout_accept_ratio},
                             "match_cache": {"directory": args.match_cache_dir, "replay": args.replay_match_cache},
                             "gt_in_correspondence_path": False},
              "reference_map": {"path": str(args.map_cache), "observations": int(len(references["map_ids"])),
                                "train_frames": len(train_rows), "geometry_only": bool(references["geometry_only"])},
              "evaluation_split": args.evaluate_split, "evaluation_frames": len(records), "records": records,
              "metrics": {"all_before": metrics(records, "before"), "candidate_after": metrics(records, "candidate_after"), "all_after": metrics(records, "after"),
                          "leader_success_before": metrics(success, "before"), "leader_success_after": metrics(success, "after")},
              "coverage": {"translation_improved_frames": int(translation_improved.sum()),
                           "rotation_improved_frames": int(rotation_improved.sum()),
                           "both_improved_frames": int((translation_improved & rotation_improved).sum()),
                           "accepted_candidates": int(sum(record["acceptance"]["accepted"] for record in records)),
                           "median_correspondences_per_frame": float(np.median(correspondence_counts)) if len(correspondence_counts) else float("nan"),
                           "median_active_cameras_per_frame": float(np.median(active_camera_counts)) if len(active_camera_counts) else float("nan"),
                           "mean_active_cameras_per_frame": float(active_camera_counts.mean()) if len(active_camera_counts) else float("nan")},
              "gate_diagnostics_train_only": gate_diagnostics(records) if args.evaluate_split == "train" else None,
              "paired": {"all_mean_delta_translation_m_rotation_deg": paired.mean(axis=0).tolist() if len(paired) else [],
                         "all_bootstrap_95ci": bootstrap_ci(paired),
                         "leader_success_mean_delta_translation_m_rotation_deg": paired_success.mean(axis=0).tolist() if len(paired_success) else [],
                         "leader_success_bootstrap_95ci": bootstrap_ci(paired_success)}, "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
