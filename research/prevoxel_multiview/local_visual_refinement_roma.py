"""LEADER-guided RoMa v2 local correspondence refinement."""
import argparse
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

from local_visual_refinement import descriptor_row, visible_map_points
from oracle_pose_refinement import (
    bootstrap_ci,
    geometry_diagnostics,
    load_lidar,
    load_module,
    metrics,
    pose_error,
    pose_from_baseline,
    project_world,
    se3_exp,
)


def build_reference_observations(rows, lidar_cache, feature_cache, projection_cache,
                                 voxel_size, output, max_history):
    required = {"points", "map_ids", "world_xyz", "frame_ids", "camera_ids", "ref_uv"}
    if output.exists():
        with np.load(output) as data:
            if required.issubset(data.files):
                return {key: np.asarray(data[key]) for key in data.files}
    point_ids, points, observations = {}, [], defaultdict(list)
    train_rows = [row for row in rows if row["split"] == "train"]
    for index, row in enumerate(train_rows):
        cached = load_lidar(lidar_cache, row["frame_id"])
        _, valid = descriptor_row(Path(feature_cache) / (row["frame_id"] + ".npz"), len(cached["source"]))
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
                map_index = len(points)
                point_ids[key] = map_index
                points.append(world[raw_index])
            map_for_raw[raw_index] = map_index
        for view in row["views"]:
            camera = int(view["camera"])
            raw_indices = np.where(valid[:, camera])[0]
            if not len(raw_indices):
                continue
            K = np.loadtxt(view["calibration"]).astype(np.float64)
            uv, depth = project_world(world[raw_indices], cached["GT"],
                                      np.asarray(view["camera_to_body"], dtype=np.float64), K)
            image_mask = np.asarray(np.load(view["mask"]))
            height, width = image_mask.shape
            integer = np.rint(uv).astype(np.int64)
            in_image = (depth > .5) & np.isfinite(uv).all(axis=1)
            in_image &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            integer[:, 0] = np.clip(integer[:, 0], 0, width - 1)
            integer[:, 1] = np.clip(integer[:, 1], 0, height - 1)
            in_image &= image_mask[integer[:, 1], integer[:, 0]] > 0
            for position in np.where(in_image)[0]:
                raw_index = raw_indices[position]
                key = (int(map_for_raw[raw_index]), row["frame_id"], camera)
                if len(observations[key]) < max_history:
                    observations[key].append((world[raw_index], uv[position]))
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
        "points": np.asarray(points, dtype=np.float32),
        "map_ids": np.asarray(map_ids, dtype=np.int32),
        "frame_ids": np.asarray(frame_ids, dtype="U32"),
        "camera_ids": np.asarray(camera_ids, dtype=np.int8),
        "world_xyz": np.asarray(world_xyz, dtype=np.float32),
        "ref_uv": np.asarray(ref_uv, dtype=np.float32),
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
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


def refine_pose_weighted(initial, points, pixels, cameras, precisions, views,
                         max_translation, max_rotation, max_nfev, f_scale):
    from scipy.optimize import least_squares

    camera_data = {int(view["camera"]): (np.asarray(view["camera_to_body"], dtype=np.float64),
                                           np.loadtxt(view["calibration"]).astype(np.float64)) for view in views}
    cholesky = np.linalg.cholesky(precisions)
    def residual(delta):
        pose = se3_exp(delta) @ initial
        raw = np.empty((len(points), 2), dtype=np.float64)
        for camera in range(6):
            keep = cameras == camera
            if not keep.any():
                continue
            uv, depth = project_world(points[keep], pose, *camera_data[camera])
            raw[keep] = uv - pixels[keep]
            bad = (~np.isfinite(raw[keep]).all(axis=1)) | (depth <= 1e-4)
            raw[np.where(keep)[0][bad]] = 1000.
        return np.einsum("nij,nj->ni", np.swapaxes(cholesky, 1, 2), raw).reshape(-1)
    bounds = np.concatenate([np.full(3, max_translation), np.full(3, max_rotation)])
    result = least_squares(residual, np.zeros(6), bounds=(-bounds, bounds), method="trf",
                           loss="soft_l1", f_scale=f_scale, max_nfev=max_nfev)
    return se3_exp(result.x) @ initial, result, residual(result.x)


def query_matches(row, initial, references, rows_by_frame, lidar_cache, roma, crop_radius, local_radius,
                  min_overlap, max_reference_images, grid_cell, max_per_camera,
                  min_precision, max_precision, min_view_cosine):
    map_points = references["points"].astype(np.float64)
    local_ids = np.where(np.linalg.norm(map_points - initial[:3, 3], axis=1) <= crop_radius)[0]
    local_set = set(int(value) for value in local_ids)
    all_points, all_pixels, all_cameras, all_scores, all_precisions, diagnostics = [], [], [], [], [], []
    for query_view in sorted(row["views"], key=lambda value: value["camera"]):
        from PIL import Image
        query_image = np.asarray(Image.open(query_view["image"]).convert("RGB"))
        query_mask = np.asarray(np.load(query_view["mask"]))
        query_height, query_width = query_image.shape[:2]
        local_uv, visible_local = visible_map_points(map_points[local_ids], initial, query_view, query_image, query_mask)
        if not len(visible_local):
            diagnostics.append({"camera": int(query_view["camera"]), "visible_map_points": 0, "reference_images": 0, "accepted": 0})
            continue
        visible_ids = local_ids[visible_local]
        visible_set = set(int(value) for value in visible_ids)
        base_by_map = {int(map_id): local_uv[position] for position, map_id in enumerate(visible_ids)}
        eligible = np.fromiter((int(map_id) in visible_set for map_id in references["map_ids"]), dtype=bool,
                               count=len(references["map_ids"]))
        eligible &= references["frame_ids"] != row["frame_id"]
        record_rows = np.where(eligible)[0]
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
            base = np.asarray([base_by_map[int(map_id)] for map_id in references["map_ids"][ref_rows]], dtype=np.float64)
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
            valid &= np.abs(query_uv - base).max(axis=1) <= local_radius
            stage_counts["local"] += int(valid.sum())
            valid &= overlap >= min_overlap
            stage_counts["overlap"] += int(valid.sum())
            precision = stabilize_precision(precision, min_precision, max_precision)
            for position in np.where(valid)[0]:
                candidates.append((float(overlap[position]), world[position], query_uv[position], precision[position],
                                   reference_frame, reference_camera))
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
        diagnostics.append({"camera": int(query_view["camera"]), "visible_map_points": int(len(visible_ids)),
                            "reference_images": int(len(selected_pairs)), "raw_candidates": int(len(candidates)),
                            "accepted": int(len(selected)), "stages": dict(stage_counts),
                            "reference_pair_covisible_anchors": {"%s:%d" % pair: pair_scores[pair] for pair in selected_pairs},
                            "reference_pairs": [list(pair) for pair in selected_pairs]})
    if not all_points:
        return (np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int64), np.empty(0),
                np.empty((0, 2, 2)), diagnostics)
    return (np.concatenate(all_points), np.concatenate(all_pixels), np.concatenate(all_cameras),
            np.asarray(all_scores), np.concatenate(all_precisions), diagnostics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise", choices=("turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--max-history", type=int, default=4)
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
    parser.add_argument("--f-scale-whitened", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] in ("val", "validation", "test")]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
        rows = train_rows + val_rows
    if args.frames:
        val_rows = val_rows[:args.frames]
    references = build_reference_observations(rows, args.lidar_cache, args.feature_cache, args.projection_cache,
                                              args.map_voxel_size, Path(args.map_cache), args.max_history)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("roma_local_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("roma_local_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    roma = RoMaField(args.device, args.roma_setting)
    records, started = [], time.time()
    for index, row in enumerate(val_rows):
        initial, gt, support = pose_from_baseline(row, args.lidar_cache, matcher, full_pool_module.full_pool_refine,
                                                  args.device, args.seed + index)
        before = pose_error(initial, gt)
        points, pixels, cameras, scores, precisions, matching = query_matches(
            row, initial, references, rows_by_frame, args.lidar_cache, roma, args.crop_radius, args.local_radius, args.min_overlap,
            args.max_reference_images, args.grid_cell, args.max_per_camera, args.min_precision, args.max_precision,
            args.min_reference_view_cosine)
        if len(points) >= 6 and len(np.unique(cameras)):
            refined, optimizer, residual = refine_pose_weighted(initial, points, pixels, cameras, precisions, row["views"],
                                                                  args.max_translation, math.radians(args.max_rotation_deg),
                                                                  args.max_nfev, args.f_scale_whitened)
            after = pose_error(refined, gt)
            solver = {"success": bool(optimizer.success), "status": int(optimizer.status), "nfev": int(optimizer.nfev),
                      "cost": float(optimizer.cost), "whitened_residual_rmse": float(np.sqrt(np.mean(residual ** 2)))}
        else:
            refined, after = initial.copy(), before
            solver = {"success": False, "status": -1, "nfev": 0, "cost": float("nan"), "whitened_residual_rmse": float("nan")}
        record = {"frame_id": row["frame_id"], "before": list(before), "after": list(after),
                  "delta": [after[0] - before[0], after[1] - before[1]],
                  "leader_success": bool(before[0] < 1. and before[1] < 2.), "n_correspondences": int(len(points)),
                  "n_cameras": int(len(np.unique(cameras))) if len(cameras) else 0,
                  "overlap_mean": float(scores.mean()) if len(scores) else float("nan"),
                  "per_camera_correspondences": [int((cameras == camera).sum()) for camera in range(6)],
                  "matching": matching, "solver": solver, "baseline_support": support,
                  "geometry": geometry_diagnostics(points, cameras, initial, row["views"]),
                  "leader_pose": initial.tolist(), "refined_pose": refined.tolist(), "gt_pose": gt.tolist()}
        records.append(record)
        print("validation %d/%d %s cams=%d corr=%d overlap=%.3f before=(%.3f,%.3f) after=(%.3f,%.3f)" %
              (index + 1, len(val_rows), row["frame_id"], record["n_cameras"], len(points), record["overlap_mean"],
               before[0], before[1], after[0], after[1]), flush=True)
    success = [record for record in records if record["leader_success"]]
    paired, paired_success = np.asarray([record["delta"] for record in records]), np.asarray([record["delta"] for record in success])
    result = {"protocol": {"map_source": "train split only", "front_end": "RoMa v2 dense reference-to-query fields",
                             "matching": {"max_reference_images_per_query_camera": args.max_reference_images,
                                          "local_radius_px": args.local_radius, "min_overlap": args.min_overlap,
                                          "min_reference_view_cosine": args.min_reference_view_cosine,
                                          "grid_cell_px": args.grid_cell, "max_per_camera": args.max_per_camera},
                             "optimizer": "bounded robust LM on RoMa pixel-precision-whitened residuals",
                             "f_scale_whitened": args.f_scale_whitened, "gt_in_correspondence_path": False},
              "reference_map": {"path": str(args.map_cache), "points": int(len(references["points"])),
                                "observations": int(len(references["map_ids"])), "train_frames": len(train_rows)},
              "validation_frames": len(records), "records": records,
              "metrics": {"all_before": metrics(records, "before"), "all_after": metrics(records, "after"),
                          "leader_success_before": metrics(success, "before"), "leader_success_after": metrics(success, "after")},
              "paired": {"all_mean_delta_translation_m_rotation_deg": paired.mean(axis=0).tolist() if len(paired) else [],
                         "all_bootstrap_95ci": bootstrap_ci(paired),
                         "leader_success_mean_delta_translation_m_rotation_deg": paired_success.mean(axis=0).tolist() if len(paired_success) else [],
                         "leader_success_bootstrap_95ci": bootstrap_ci(paired_success)}, "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
