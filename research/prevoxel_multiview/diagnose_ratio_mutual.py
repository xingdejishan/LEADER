"""Train-only selection diagnostics for ratio and mutual correspondence filters."""
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

from local_visual_refinement import DenseDescriptorExtractor, build_visual_map, query_matches
from oracle_pose_refinement import load_module, pose_error, pose_from_baseline, project_world


def add_stage_counts(summary, stage, correct, offset_norm):
    cosine = stage["cosine_keep"]
    ratio = stage["ratio_keep"]
    mutual = stage["mutual_keep"]
    stages = {
        "cosine": cosine,
        "ratio": ratio,
        "mutual": mutual,
    }
    summary["candidates"] += int(len(correct))
    summary["offset_norm_sum"] += float(offset_norm.sum())
    summary["offset_norm_count"] += int(len(offset_norm))
    for name, keep in stages.items():
        selected = correct[keep]
        summary[name]["count"] += int(keep.sum())
        summary[name]["correct_lt5px"] += int(selected.sum())
        summary[name]["offset_norm_sum"] += float(offset_norm[keep].sum())
        summary[name]["offset_norm_count"] += int(keep.sum())
    ratio_candidates = ratio
    for name, band in (("le8", offset_norm <= 8.0), ("gt8", offset_norm > 8.0)):
        selected = ratio_candidates & band
        summary["mutual_offset"][name]["ratio_candidates"] += int(selected.sum())
        summary["mutual_offset"][name]["mutual_pass"] += int((selected & mutual).sum())
        summary["mutual_offset"][name]["correct_lt5px"] += int(correct[selected].sum())


def new_summary():
    return {
        "candidates": 0,
        "offset_norm_sum": 0.0,
        "offset_norm_count": 0,
        "cosine": {"count": 0, "correct_lt5px": 0, "offset_norm_sum": 0.0, "offset_norm_count": 0},
        "ratio": {"count": 0, "correct_lt5px": 0, "offset_norm_sum": 0.0, "offset_norm_count": 0},
        "mutual": {"count": 0, "correct_lt5px": 0, "offset_norm_sum": 0.0, "offset_norm_count": 0},
        "mutual_offset": {
            "le8": {"ratio_candidates": 0, "mutual_pass": 0, "correct_lt5px": 0},
            "gt8": {"ratio_candidates": 0, "mutual_pass": 0, "correct_lt5px": 0},
        },
    }


def finalize_summary(summary):
    result = {"candidates": summary["candidates"]}
    result["mean_forward_offset_px"] = summary["offset_norm_sum"] / max(summary["offset_norm_count"], 1)
    for name in ("cosine", "ratio", "mutual"):
        item = summary[name]
        result[name] = {
            "count": item["count"],
            "retention": item["count"] / max(summary["cosine"]["count"], 1),
            "correct_lt5px": item["correct_lt5px"],
            "precision_lt5px": item["correct_lt5px"] / max(item["count"], 1),
            "mean_forward_offset_px": item["offset_norm_sum"] / max(item["offset_norm_count"], 1),
        }
    result["mutual_offset"] = {}
    for name, item in summary["mutual_offset"].items():
        result["mutual_offset"][name] = {
            "ratio_candidates": item["ratio_candidates"],
            "mutual_pass": item["mutual_pass"],
            "mutual_retention": item["mutual_pass"] / max(item["ratio_candidates"], 1),
            "correct_lt5px": item["correct_lt5px"],
        }
    return result


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
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--crop-radius", type=float, default=80.0)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--search-step", type=int, default=4)
    parser.add_argument("--min-cosine", type=float, default=0.55)
    parser.add_argument("--ratio-threshold", type=float, default=0.8)
    parser.add_argument("--ratio-exclusion-radius", type=float, default=4.0)
    parser.add_argument("--mutual-radius", type=float, default=8.0)
    parser.add_argument("--disable-mutual", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        train_rows = train_rows[:args.frames]
    visual_map = build_visual_map(train_rows, args.lidar_cache, args.feature_cache,
                                  args.projection_cache, args.map_voxel_size,
                                  Path(args.map_cache), args.max_history)
    matcher_module = load_module("ratio_mutual_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("ratio_mutual_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    aggregate = new_summary()
    per_frame = []
    started = time.time()
    for index, row in enumerate(train_rows):
        initial, gt, _ = pose_from_baseline(row, args.lidar_cache, matcher,
                                             full_pool_module.full_pool_refine,
                                             args.device, args.seed + index)
        stages = []
        query_matches(row, initial, visual_map, extractor, args.crop_radius,
                      args.search_radius, args.search_step, args.min_cosine, 4, 0,
                      exclude_frame_id=row["frame_id"],
                      ratio_threshold=args.ratio_threshold,
                      ratio_exclusion_radius=args.ratio_exclusion_radius,
                      mutual_radius=None if args.disable_mutual else args.mutual_radius,
                      stage_records=stages)
        frame_summary = new_summary()
        for stage in stages:
            view = next(item for item in row["views"] if int(item["camera"]) == stage["camera"])
            K = np.loadtxt(view["calibration"]).astype(np.float64)
            extrinsic = np.asarray(view["camera_to_body"], dtype=np.float64)
            expected, _ = project_world(stage["points"], gt, extrinsic, K)
            error = np.linalg.norm(expected - stage["pixels"], axis=1)
            correct = np.isfinite(error) & (error < 5.0)
            offset_norm = np.linalg.norm(stage["best_offsets"], axis=1)
            add_stage_counts(aggregate, stage, correct, offset_norm)
            add_stage_counts(frame_summary, stage, correct, offset_norm)
        per_frame.append({
            "frame_id": row["frame_id"],
            "leader_before": pose_error(initial, gt),
            "summary": finalize_summary(frame_summary),
        })
        print("train %d/%d %s candidates=%d cosine=%d ratio=%d mutual=%d" % (
            index + 1, len(train_rows), row["frame_id"], frame_summary["candidates"],
            frame_summary["cosine"]["count"], frame_summary["ratio"]["count"],
            frame_summary["mutual"]["count"]), flush=True)
    result = {
        "protocol": {
            "split": "train-only",
            "leave_one_frame_out": True,
            "correct_match_definition": "GT reprojection pixel error <5 px",
            "search_radius_px": args.search_radius,
            "search_step_px": args.search_step,
            "min_cosine": args.min_cosine,
            "ratio_threshold": args.ratio_threshold,
            "ratio_exclusion_radius_px": args.ratio_exclusion_radius,
            "mutual_radius_px": None if args.disable_mutual else args.mutual_radius,
            "mutual_enabled": not args.disable_mutual,
            "mutual_window_shape": "axis-aligned square",
            "validation_used_for_tuning": False,
        },
        "visual_map": {
            "path": str(args.map_cache),
            "points": int(len(visual_map["points"])),
            "descriptor_observations": int(len(visual_map["descriptors"])),
            "train_frames": len(train_rows),
        },
        "aggregate": finalize_summary(aggregate),
        "frames": per_frame,
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
