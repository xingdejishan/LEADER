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

from local_visual_refinement import visible_map_points
from local_visual_refinement_roma import RoMaField, build_reference_observations, query_matches
from oracle_pose_refinement import load_module, pose_from_baseline


def summary(errors, delta):
    errors = np.asarray(errors, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if not len(errors):
        return {"count": 0, "median_pixel_error": None, "p90_pixel_error": None,
                "mean_du": None, "mean_dv": None, "lt_2px_fraction": None,
                "lt_5px_fraction": None, "lt_8px_fraction": None, "gt_20px_fraction": None}
    return {
        "count": int(len(errors)),
        "median_pixel_error": float(np.median(errors)),
        "p90_pixel_error": float(np.quantile(errors, .90)),
        "mean_du": float(delta[:, 0].mean()),
        "mean_dv": float(delta[:, 1].mean()),
        "lt_2px_fraction": float((errors < 2.).mean()),
        "lt_5px_fraction": float((errors < 5.).mean()),
        "lt_8px_fraction": float((errors < 8.).mean()),
        "gt_20px_fraction": float((errors > 20.).mean()),
    }


def score_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .10)), "p90": float(np.quantile(values, .90))}


def calibration_bins(values, errors):
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(errors)
    values, errors = values[valid], errors[valid]
    if not len(values):
        return {"count": 0, "spearman_error_correlation": None, "bins": []}
    from scipy.stats import spearmanr

    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    groups = (("low", values <= lower), ("medium", (values > lower) & (values < upper)), ("high", values >= upper))
    bins = []
    for name, keep in groups:
        bin_errors = errors[keep]
        if not len(bin_errors):
            bins.append({"name": name, "count": 0, "score_min": None, "score_max": None,
                         "median_pixel_error": None, "p90_pixel_error": None, "lt_2px_fraction": None,
                         "lt_5px_fraction": None, "gt_20px_fraction": None})
            continue
        bins.append({"name": name, "count": int(keep.sum()), "score_min": float(values[keep].min()),
                     "score_max": float(values[keep].max()), "median_pixel_error": float(np.median(bin_errors)),
                     "p90_pixel_error": float(np.quantile(bin_errors, .90)),
                     "lt_2px_fraction": float((bin_errors < 2.).mean()),
                     "lt_5px_fraction": float((bin_errors < 5.).mean()),
                     "gt_20px_fraction": float((bin_errors > 20.).mean())})
    correlation = spearmanr(values, errors).statistic if len(values) > 1 else float("nan")
    return {"count": int(len(values)), "spearman_error_correlation": float(correlation) if np.isfinite(correlation) else None,
            "thresholds": [float(lower), float(upper)], "bins": bins}


def truth_pixels(points, cameras, row, gt):
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    visible = np.zeros(len(points), dtype=bool)
    views = {int(view["camera"]): view for view in row["views"]}
    for camera in range(6):
        positions = np.where(cameras == camera)[0]
        if not len(positions):
            continue
        view = views[camera]
        from PIL import Image

        image = np.asarray(Image.open(view["image"]).convert("RGB"))
        mask = np.asarray(np.load(view["mask"]))
        projected, indices = visible_map_points(points[positions], gt, view, image, mask)
        pixels[positions] = projected
        visible[positions[indices]] = True
    return pixels, visible


def load_checkpoint(path):
    with np.load(path) as data:
        uncorrected = np.asarray(data["uncorrected_errors"]) if "uncorrected_errors" in data.files else np.asarray(data["errors"])
        return (json.loads(str(data["frame_records_json"])), np.asarray(data["cameras"]),
                np.asarray(data["errors"]), np.asarray(data["delta"]), np.asarray(data["overlap"]),
                np.asarray(data["precision"]), uncorrected)


def save_checkpoint(path, frame_records, cameras, errors, delta, overlap, precision, uncorrected_errors):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_records_json=np.asarray(json.dumps(frame_records, allow_nan=False)),
                        cameras=np.concatenate(cameras) if cameras else np.empty(0, dtype=np.int8),
                        errors=np.concatenate(errors) if errors else np.empty(0),
                        delta=np.concatenate(delta) if delta else np.empty((0, 2)),
                        overlap=np.concatenate(overlap) if overlap else np.empty(0),
                        precision=np.concatenate(precision) if precision else np.empty(0),
                        uncorrected_errors=np.concatenate(uncorrected_errors) if uncorrected_errors else np.empty(0))


def load_camera_bias(path):
    bias = np.zeros((6, 2), dtype=np.float64)
    if path is None:
        return bias
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    values = result.get("camera_bias_px")
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("camera bias file must contain six camera_bias_px entries")
    for item in values:
        camera = int(item["camera"])
        if camera < 0 or camera >= 6:
            raise ValueError("invalid camera in bias file")
        bias[camera] = [float(item["du_px"]), float(item["dv_px"])]
    return bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--evaluate-split", default="train", choices=("train", "val"))
    parser.add_argument("--frames", type=int, default=0)
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
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--camera-bias")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output.with_suffix(output.suffix + ".checkpoint.npz")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    eval_rows = [row for row in rows if row["split"] == args.evaluate_split]
    if args.frames:
        eval_rows = eval_rows[:args.frames]
    if not eval_rows:
        raise ValueError("manifest has no selected rows")
    references = build_reference_observations(rows, args.lidar_cache, args.projection_cache,
                                              args.map_voxel_size, Path(args.map_cache), args.max_history)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    bias = load_camera_bias(args.camera_bias)
    frame_records, raw_cameras, raw_errors, raw_delta, raw_overlap, raw_precision, raw_uncorrected_errors = [], [], [], [], [], [], []
    if args.resume and checkpoint.exists():
        completed, cameras, errors, delta, overlap, precision, uncorrected_errors = load_checkpoint(checkpoint)
        frame_records = completed
        if len(cameras):
            raw_cameras, raw_errors, raw_delta = [cameras], [errors], [delta]
            raw_overlap, raw_precision, raw_uncorrected_errors = [overlap], [precision], [uncorrected_errors]
        print("resuming %d completed frames from %s" % (len(frame_records), checkpoint), flush=True)
    completed_ids = {record["frame_id"] for record in frame_records}
    pending_rows = [row for row in eval_rows if row["frame_id"] not in completed_ids]
    matcher = pool_module = roma = None
    if pending_rows:
        matcher_module = load_module("roma_accuracy_matcher", REPO / "models" / "sc2pcr.py")
        pool_module = load_module("roma_accuracy_pool", Path(args.full_pool))
        matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                         ratio=.15, nms_radius=.1, max_points=3000, k1=30)
        roma = RoMaField(args.device, args.roma_setting)
    started = time.time()
    for index, row in enumerate(pending_rows):
        source_index = eval_rows.index(row)
        initial, gt, _ = pose_from_baseline(row, args.lidar_cache, matcher, pool_module.full_pool_refine,
                                            args.device, args.seed + source_index)
        points, predicted, cameras, overlap, precisions, _, _, _, matching = query_matches(
            row, initial, references, rows_by_frame, args.lidar_cache, roma, args.crop_radius, args.local_radius,
            args.min_overlap, args.max_reference_images, args.grid_cell, args.max_per_camera,
            args.min_precision, args.max_precision, args.min_reference_view_cosine)
        truth, truth_visible = truth_pixels(points, cameras, row, gt)
        uncorrected_delta = predicted - truth
        error_delta = uncorrected_delta - bias[cameras]
        errors = np.linalg.norm(error_delta, axis=1)
        uncorrected_errors = np.linalg.norm(uncorrected_delta, axis=1)
        precision_score = np.sqrt(np.maximum(np.linalg.det(precisions), 0.))
        frame = {"frame_id": row["frame_id"], "candidate_correspondences": int(len(points)),
                 "gt_visible_correspondences": int(truth_visible.sum()), "matching": matching, "per_camera": []}
        for camera in range(6):
            keep = (cameras == camera) & truth_visible & np.isfinite(errors)
            raw_cameras.append(np.full(int(keep.sum()), camera, dtype=np.int8))
            raw_errors.append(errors[keep])
            raw_delta.append(error_delta[keep])
            raw_overlap.append(overlap[keep])
            raw_precision.append(precision_score[keep])
            raw_uncorrected_errors.append(uncorrected_errors[keep])
            record = summary(errors[keep], error_delta[keep])
            record.update({"camera": camera, "candidate_correspondences": int((cameras == camera).sum()),
                           "gt_visible_correspondences": int(keep.sum()), "overlap": score_summary(overlap[keep]),
                           "precision_sqrt_det": score_summary(precision_score[keep]),
                           "uncorrected": summary(uncorrected_errors[keep], uncorrected_delta[keep])})
            frame["per_camera"].append(record)
        frame_records.append(frame)
        roma.clear_cache()
        save_checkpoint(checkpoint, frame_records, raw_cameras, raw_errors, raw_delta, raw_overlap, raw_precision,
                        raw_uncorrected_errors)
        print("RoMa accuracy %d/%d %s candidates=%d gt_visible=%d checkpoint=%s" %
              (len(frame_records), len(eval_rows), row["frame_id"], len(points), truth_visible.sum(), checkpoint), flush=True)
    per_camera = []
    all_cameras = np.concatenate(raw_cameras) if raw_cameras else np.empty(0, dtype=np.int8)
    all_errors = np.concatenate(raw_errors) if raw_errors else np.empty(0)
    all_delta = np.concatenate(raw_delta) if raw_delta else np.empty((0, 2))
    all_overlap = np.concatenate(raw_overlap) if raw_overlap else np.empty(0)
    all_precision = np.concatenate(raw_precision) if raw_precision else np.empty(0)
    all_uncorrected_errors = np.concatenate(raw_uncorrected_errors) if raw_uncorrected_errors else np.empty(0)
    for camera in range(6):
        keep = all_cameras == camera
        errors, delta = all_errors[keep], all_delta[keep]
        overlap, precision = all_overlap[keep], all_precision[keep]
        record = summary(errors, delta)
        record.update({"camera": camera, "overlap": score_summary(overlap), "precision_sqrt_det": score_summary(precision),
                       "overlap_calibration": calibration_bins(overlap, errors),
                       "precision_calibration": calibration_bins(precision, errors),
                       "uncorrected": summary(all_uncorrected_errors[keep], all_delta[keep] + bias[camera])})
        per_camera.append(record)
    overall = summary(all_errors, all_delta)
    overall.update({"overlap": score_summary(all_overlap), "precision_sqrt_det": score_summary(all_precision),
                    "overlap_calibration": calibration_bins(all_overlap, all_errors),
                    "precision_calibration": calibration_bins(all_precision, all_errors)})
    result = {
        "protocol": {"name": "RoMa Correspondence Accuracy Audit",
                     "split": "%s; %s" % (args.evaluate_split, "leave-one-frame-out" if args.evaluate_split == "train" else "train-only reference map"),
                     "models": "frozen RoMa v2 precise and frozen current LEADER retrieval only; no LM or fine-tuning",
                     "retrieval": "current LEADER-pose-guided ray-compatible automatic Top-2 reference images",
                     "candidate_filter": "current local gate, overlap gate, image-grid budget, and precision stabilization",
                     "truth": "query GT pose projects each selected observation-specific world_xyz",
                     "gt_visibility": "current image/mask/black-border and sparse map z-buffer checks",
                     "precision_score": "sqrt(det(RoMa 2D precision matrix)); larger is more confident",
                     "gt_used_for": "evaluation only; never retrieval, matching, or filtering",
                     "frozen_camera_bias_px": bias.tolist()},
        "settings": vars(args), "checkpoint": str(checkpoint), "reference_map": {"path": str(args.map_cache), "observations": int(len(references["world_xyz"])),
                                                       "train_frames": int(sum(row["split"] == "train" for row in rows))},
        "frames": frame_records, "per_camera": per_camera, "overall": overall,
        "uncorrected_overall": summary(all_uncorrected_errors, all_delta + bias[all_cameras]),
        "elapsed_s": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
