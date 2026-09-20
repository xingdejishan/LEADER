import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_lidar, project_world
from visibility import VisibilityChecker


def image_and_mask(view):
    from PIL import Image

    image = np.asarray(Image.open(view["image"]).convert("RGB"))
    mask = np.asarray(np.load(view["mask"]))
    if image.shape[:2] != mask.shape:
        raise ValueError("image/mask size mismatch for camera %s" % view["camera"])
    return image, mask


def world_points(row, lidar_cache, projection_cache):
    cached = load_lidar(lidar_cache, row["frame_id"])
    with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as mapping:
        projection_xyz = np.asarray(mapping["projection_xyz"], dtype=np.float64)
        localization_xyz = np.asarray(mapping["localization_xyz"], dtype=np.float64)
    if not np.array_equal(localization_xyz, cached["source"]):
        raise ValueError("projection/localization order mismatch: %s" % row["frame_id"])
    gt = np.asarray(cached["GT"], dtype=np.float64)
    return projection_xyz @ gt[:3, :3].T + gt[:3, 3], projection_xyz, gt


def project_body(points, view):
    camera_to_body = np.asarray(view["camera_to_body"], dtype=np.float64)
    body_to_camera = np.linalg.inv(camera_to_body)
    camera = points @ body_to_camera[:3, :3].T + body_to_camera[:3, 3]
    K = np.loadtxt(view["calibration"]).astype(np.float64)
    homogeneous = camera @ K.T
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, depth


def valid_image_pixels(uv, depth, image, mask, min_depth, max_depth):
    height, width = image.shape[:2]
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    valid &= np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    rounded = np.rint(uv).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    valid &= mask[rounded[:, 1], rounded[:, 0]] > 0
    valid &= image[rounded[:, 1], rounded[:, 0]].max(axis=1) > 10
    return valid


def point_visibility(checker, image, mask, depth_uv, depth, source_valid):
    grid, width_cells, height_cells = checker.build_zbuffer(depth_uv, depth, source_valid)
    occluded, _, evidence = checker.occlusion_query(grid, width_cells, height_cells, depth_uv, depth, source_valid)
    return source_valid & ~occluded, grid, width_cells, height_cells, evidence


def reference_visibility(checker, predicted_uv, predicted_depth, image, mask, grid, width_cells, height_cells):
    valid = valid_image_pixels(predicted_uv, predicted_depth, image, mask, checker.min_depth, checker.max_depth)
    occluded, _, evidence = checker.occlusion_query(grid, width_cells, height_cells, predicted_uv, predicted_depth, valid)
    return valid & ~occluded, evidence


def summarize(errors, delta):
    errors = np.asarray(errors, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    count = int(len(errors))
    if not count:
        return {"count": 0, "median_reprojection_error_px": None, "p90_reprojection_error_px": None,
                "mean_du_px": None, "mean_dv_px": None, "lt_1px_fraction": None,
                "lt_2px_fraction": None, "lt_5px_fraction": None}
    return {
        "count": count,
        "median_reprojection_error_px": float(np.median(errors)),
        "p90_reprojection_error_px": float(np.quantile(errors, .90)),
        "mean_du_px": float(delta[:, 0].mean()),
        "mean_dv_px": float(delta[:, 1].mean()),
        "lt_1px_fraction": float((errors < 1.).mean()),
        "lt_2px_fraction": float((errors < 2.).mean()),
        "lt_5px_fraction": float((errors < 5.).mean()),
    }


def audit_frame(row, references, lidar_cache, projection_cache, checker, max_world_match_m):
    from scipy.spatial import cKDTree

    query_world, query_body, gt = world_points(row, lidar_cache, projection_cache)
    historical = references["frame_ids"] != row["frame_id"]
    ref_world = np.asarray(references["world_xyz"][historical], dtype=np.float64)
    if not len(ref_world):
        raise ValueError("no historical observations for %s" % row["frame_id"])
    nearest_distance, nearest_query = cKDTree(query_world).query(ref_world, distance_upper_bound=max_world_match_m)
    matched = np.isfinite(nearest_distance) & (nearest_query < len(query_world))
    ref_world = ref_world[matched]
    query_indices = nearest_query[matched].astype(np.int64)
    reference_frames = np.asarray(references["frame_ids"][historical][matched])
    reference_cameras = np.asarray(references["camera_ids"][historical][matched])
    frame = {"frame_id": row["frame_id"], "historical_observations": int(historical.sum()),
             "world_matches": int(matched.sum()), "world_match_distance_median": float(np.median(nearest_distance[matched])) if matched.any() else None,
             "world_match_distance_p90": float(np.quantile(nearest_distance[matched], .90)) if matched.any() else None,
             "cameras": []}
    camera_errors, camera_delta = [[] for _ in range(6)], [[] for _ in range(6)]
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        image, mask = image_and_mask(view)
        observed_uv, observed_depth = project_body(query_body, view)
        observed_valid = valid_image_pixels(observed_uv, observed_depth, image, mask, checker.min_depth, checker.max_depth)
        observed_visible, grid, grid_width, grid_height, _ = point_visibility(
            checker, image, mask, observed_uv, observed_depth, observed_valid)
        predicted_uv, predicted_depth = project_world(
            ref_world, gt, np.asarray(view["camera_to_body"], dtype=np.float64),
            np.loadtxt(view["calibration"]).astype(np.float64))
        predicted_visible, predicted_evidence = reference_visibility(
            checker, predicted_uv, predicted_depth, image, mask, grid, grid_width, grid_height)
        keep = predicted_visible & observed_visible[query_indices]
        delta = predicted_uv[keep] - observed_uv[query_indices[keep]]
        errors = np.linalg.norm(delta, axis=1)
        camera_errors[int(view["camera"])].append(errors)
        camera_delta[int(view["camera"])].append(delta)
        camera = summarize(errors, delta)
        camera.update({
            "camera": int(view["camera"]),
            "query_lidar_image_valid": int(observed_valid.sum()),
            "query_lidar_visible": int(observed_visible.sum()),
            "reference_projected_visible": int(predicted_visible.sum()),
            "reference_projected_depth_evidence": int(predicted_evidence.sum()),
            "paired_observations": int(keep.sum()),
            "historical_reference_frames": int(len(np.unique(reference_frames[keep]))),
            "historical_reference_cameras": int(len(np.unique(reference_cameras[keep]))),
        })
        frame["cameras"].append(camera)
    return frame, camera_errors, camera_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, default=.2)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--max-world-match-m", type=float, default=.05)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    if args.max_world_match_m <= 0:
        raise ValueError("--max-world-match-m must be positive")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        train_rows = train_rows[:args.frames]
    if not train_rows:
        raise ValueError("no train rows selected")
    references = build_reference_observations(rows, args.lidar_cache, args.projection_cache,
                                              args.voxel_size, Path(args.map_cache), args.max_history)
    checker = VisibilityChecker(json.loads((HERE / "config.json").read_text(encoding="utf-8")))
    start = time.time()
    frames, per_camera_errors, per_camera_delta = [], [[] for _ in range(6)], [[] for _ in range(6)]
    for index, row in enumerate(train_rows):
        frame, frame_errors, frame_delta = audit_frame(
            row, references, args.lidar_cache, args.projection_cache, checker, args.max_world_match_m)
        frames.append(frame)
        for camera in range(6):
            per_camera_errors[camera].extend(frame_errors[camera])
            per_camera_delta[camera].extend(frame_delta[camera])
        print("GT geometry audit %d/%d %s matches=%d" % (index + 1, len(train_rows), row["frame_id"], frame["world_matches"]), flush=True)
    camera_summaries = []
    for camera in range(6):
        errors = np.concatenate(per_camera_errors[camera]) if per_camera_errors[camera] else np.empty(0)
        delta = np.concatenate(per_camera_delta[camera]) if per_camera_delta[camera] else np.empty((0, 2))
        summary = summarize(errors, delta)
        summary["camera"] = camera
        camera_summaries.append(summary)
    all_errors = np.concatenate([value for camera in per_camera_errors for value in camera]) if any(per_camera_errors) else np.empty(0)
    all_delta = np.concatenate([value for camera in per_camera_delta for value in camera]) if any(per_camera_delta) else np.empty((0, 2))
    result = {
        "protocol": {"name": "GT Geometry Reprojection Audit", "splits": "train only; leave-one-frame-out historical observations",
                     "models": "none: RoMa and LEADER pose are prohibited", "query_pose": "cached GT body pose",
                     "reference_observation": "observation-specific world_xyz, reference_frame, camera_id, reference_uv",
                     "observed_pixel": "query LiDAR projection_xyz point projected through official body-to-camera calibration",
                     "visibility": "image bounds, mask, black border, sparse query-LiDAR z-buffer; reject only explicit nearer evidence",
                     "max_world_match_m": args.max_world_match_m},
        "reference_map": {"path": str(args.map_cache), "observations": int(len(references["world_xyz"])),
                          "train_frames": int(sum(row["split"] == "train" for row in rows))},
        "frames": frames,
        "per_camera": camera_summaries,
        "overall": summarize(all_errors, all_delta),
        "elapsed_s": time.time() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
