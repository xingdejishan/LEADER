"""Train-only cosine threshold calibration for local visual refinement."""
import argparse
import json
import math
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

from local_visual_refinement import (
    DenseDescriptorExtractor,
    build_visual_map,
    query_matches,
)
from oracle_pose_refinement import (
    bootstrap_ci,
    load_module,
    metrics,
    pose_error,
    pose_from_baseline,
    project_world,
    refine_pose,
)


def select_candidates(points, pixels, cameras, scores, threshold, grid_cell, max_per_camera):
    selected = []
    for camera in range(6):
        indices = np.where((cameras == camera) & (scores >= threshold))[0]
        order = indices[np.argsort(-scores[indices], kind="stable")]
        occupied = set()
        for index in order:
            key = tuple(np.floor(pixels[index] / grid_cell).astype(np.int64))
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(int(index))
            if len([item for item in selected if cameras[item] == camera]) >= max_per_camera:
                break
    selected = np.asarray(selected, dtype=np.int64)
    return points[selected], pixels[selected], cameras[selected], scores[selected]


def correspondence_pixel_error(points, pixels, cameras, gt, views):
    errors = np.full(len(points), np.inf, dtype=np.float64)
    for view in views:
        camera = int(view["camera"])
        keep = cameras == camera
        if not keep.any():
            continue
        K = np.loadtxt(view["calibration"]).astype(np.float64)
        extrinsic = np.asarray(view["camera_to_body"], dtype=np.float64)
        expected, depth = project_world(points[keep], gt, extrinsic, K)
        errors[keep] = np.linalg.norm(expected - pixels[keep], axis=1)
        errors[keep & ~np.isfinite(errors)] = np.inf
    return errors


def run_threshold(records, threshold, args):
    refined_records = []
    for record in records:
        points, pixels, cameras, scores = select_candidates(
            record["points"], record["pixels"], record["cameras"], record["scores"],
            threshold, args.grid_cell, args.max_per_camera)
        before = record["before"]
        if len(points) >= 6 and len(np.unique(cameras)):
            refined, _, _ = refine_pose(
                record["initial"], points, pixels, cameras, record["views"],
                args.max_translation, math.radians(args.max_rotation_deg),
                args.max_nfev, args.f_scale_px)
            after = pose_error(refined, record["gt"])
        else:
            after = before
        pixel_error = correspondence_pixel_error(points, pixels, cameras, record["gt"], record["views"])
        refined_records.append({
            "before": before, "after": after,
            "delta": [after[0] - before[0], after[1] - before[1]],
            "n_correspondences": int(len(points)),
            "n_cameras": int(len(np.unique(cameras))) if len(cameras) else 0,
            "correct_lt5px": int((pixel_error < 5.0).sum()),
            "correct_lt10px": int((pixel_error < 10.0).sum()),
            "pixel_error_median": float(np.median(pixel_error)) if len(pixel_error) else float("nan"),
            "pixel_error_p90": float(np.percentile(pixel_error, 90)) if len(pixel_error) else float("nan"),
        })
    success = [r for r, original in zip(refined_records, records) if original["leader_success"]]
    deltas = np.asarray([r["delta"] for r in refined_records], dtype=np.float64)
    success_deltas = np.asarray([r["delta"] for r in success], dtype=np.float64)
    all_count = np.asarray([r["n_correspondences"] for r in refined_records], dtype=np.float64)
    correct = np.asarray([r["correct_lt5px"] for r in refined_records], dtype=np.float64)
    summary = {
        "threshold": threshold,
        "frames": len(refined_records),
        "mean_correspondences": float(all_count.mean()),
        "median_correspondences": float(np.median(all_count)),
        "frames_with_at_least_6": int((all_count >= 6).sum()),
        "mean_precision_lt5px": float(correct.sum() / max(all_count.sum(), 1.0)),
        "mean_frame_precision_lt5px": float(np.mean([r["correct_lt5px"] / max(r["n_correspondences"], 1) for r in refined_records])),
        "mean_pixel_error": float(np.mean([r["pixel_error_median"] for r in refined_records if np.isfinite(r["pixel_error_median"])])),
        "metrics": {"before": metrics([{ "before": r["before"] } for r in refined_records], "before"),
                    "after": metrics([{ "after": r["after"] } for r in refined_records], "after"),
                    "leader_success_after": metrics([{ "after": r["after"] } for r in success], "after")},
        "paired": {
            "all_mean_delta": deltas.mean(axis=0).tolist(),
            "all_median_delta": np.median(deltas, axis=0).tolist(),
            "all_bootstrap_95ci": bootstrap_ci(deltas),
            "leader_success_mean_delta": success_deltas.mean(axis=0).tolist() if len(success_deltas) else [],
            "leader_success_median_delta": np.median(success_deltas, axis=0).tolist() if len(success_deltas) else [],
            "leader_success_bootstrap_95ci": bootstrap_ci(success_deltas),
            "frames_improving_both": int(((deltas[:, 0] < 0) & (deltas[:, 1] < 0)).sum()),
        },
        "records": refined_records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--dedode-weights", required=True)
    parser.add_argument("--pca-weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--crop-radius", type=float, default=80.0)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--search-step", type=int, default=4)
    parser.add_argument("--grid-cell", type=int, default=4)
    parser.add_argument("--max-per-camera", type=int, default=300)
    parser.add_argument("--max-translation", type=float, default=2.0)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--f-scale-px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        train_rows = train_rows[:args.frames]
    visual_map = build_visual_map(train_rows, args.lidar_cache, args.feature_cache,
                                  args.projection_cache, args.map_voxel_size,
                                  Path(args.map_cache), args.max_history)
    matcher_module = load_module("threshold_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("threshold_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    min_threshold = min(float(value) for value in args.thresholds.split(","))
    candidates = []
    started = time.time()
    for index, row in enumerate(train_rows):
        initial, gt, support = pose_from_baseline(row, args.lidar_cache, matcher,
                                                  full_pool_module.full_pool_refine,
                                                  args.device, args.seed + index)
        before = pose_error(initial, gt)
        points, pixels, cameras, scores, matching = query_matches(
            row, initial, visual_map, extractor, args.crop_radius, args.search_radius,
            args.search_step, min_threshold,
            args.grid_cell, 0, exclude_frame_id=row["frame_id"])
        candidates.append({"frame_id": row["frame_id"], "initial": initial, "gt": gt,
                           "before": before, "leader_success": before[0] < 1.0 and before[1] < 2.0,
                           "points": points, "pixels": pixels, "cameras": cameras,
                           "scores": scores, "views": row["views"], "matching": matching,
                           "baseline_support": support})
        print("train %d/%d %s raw_candidates=%d" %
              (index + 1, len(train_rows), row["frame_id"], len(points)), flush=True)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    results = {"%.2f" % threshold: run_threshold(candidates, threshold, args) for threshold in thresholds}
    result = {
        "protocol": {"split": "train-only", "leave_one_frame_out": True,
                     "correct_match_definition": "GT reprojection pixel error <5 px",
                     "thresholds": thresholds, "map_source": "train frames excluding query descriptor observations",
                     "optimizer": "same bounded robust LM as oracle", "validation_used_for_tuning": False},
        "visual_map": {"path": str(args.map_cache), "points": int(len(visual_map["points"])),
                       "descriptor_observations": int(len(visual_map["descriptors"])), "frames": len(train_rows)},
        "elapsed_s": time.time() - started,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({key: {name: value for name, value in summary.items()
                           if name in ("mean_correspondences", "mean_precision_lt5px", "frames_with_at_least_6")}
                      for key, summary in results.items()}, indent=2))


if __name__ == "__main__":
    main()
