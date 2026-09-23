import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import nre_scoremap_pose_runner as nre


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def pose_error(pose, ground_truth):
    delta = pose[:3, :3].T @ ground_truth[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.) / 2., -1., 1.)
    return (float(np.linalg.norm(pose[:3, 3] - ground_truth[:3, 3])),
            float(np.degrees(np.arccos(cosine))))


def block_interval(deltas, block_size=4, samples=20000, seed=2089):
    deltas = np.asarray(deltas, dtype=np.float64)
    blocks = [np.arange(start, min(start + block_size, len(deltas)))
              for start in range(0, len(deltas), block_size)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        selected = np.concatenate([blocks[value] for value in chosen])
        estimates[index] = float(np.mean(deltas[selected]))
    return [float(value) for value in np.quantile(estimates, (.025, .975))]


def projection_targets(points, pose, cameras, row):
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = np.full(len(points), np.nan, dtype=np.float64)
    for camera in np.unique(cameras):
        indices = np.where(cameras == camera)[0]
        view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
        projected, z = nre.project_world(points[indices], pose,
                                         np.asarray(view["camera_to_body"], dtype=np.float64),
                                         np.loadtxt(view["calibration"]).astype(np.float64))
        uv[indices] = projected
        depth[indices] = z
    return uv, depth


def evaluate_pose(records, lidar_cache, pose_key, solver_key):
    gt_errors, base_errors, per_frame = [], [], []
    failures = 0
    for record in records:
        frame_id = str(record["frame_id"])
        with np.load(Path(lidar_cache) / (frame_id + ".npz")) as data:
            gt = np.asarray(data["GT"], dtype=np.float64)
        gt_error = pose_error(np.asarray(record[pose_key], dtype=np.float64), gt)
        baseline_error = pose_error(np.asarray(record["baseline_pose"], dtype=np.float64), gt)
        solver = record[solver_key]
        failures += int(not solver["success"])
        gt_errors.append(gt_error)
        base_errors.append(baseline_error)
        per_frame.append({
            "frame_id": frame_id,
            "baseline_mpe_m": baseline_error[0],
            "candidate_mpe_m": gt_error[0],
            "delta_mpe_m": gt_error[0] - baseline_error[0],
            "baseline_moe_deg": baseline_error[1],
            "candidate_moe_deg": gt_error[1],
            "delta_moe_deg": gt_error[1] - baseline_error[1],
            "both_improved": bool(gt_error[0] < baseline_error[0] and gt_error[1] < baseline_error[1]),
        })
    base_errors = np.asarray(base_errors, dtype=np.float64)
    gt_errors = np.asarray(gt_errors, dtype=np.float64)
    delta_mpe = gt_errors[:, 0] - base_errors[:, 0]
    delta_moe = gt_errors[:, 1] - base_errors[:, 1]
    return {
        "denominator": len(records),
        "baseline": {"mpe_m": summarize(base_errors[:, 0]), "moe_deg": summarize(base_errors[:, 1])},
        "candidate": {"mpe_m": summarize(gt_errors[:, 0]), "moe_deg": summarize(gt_errors[:, 1])},
        "paired_delta": {
            "mean_mpe_m": float(np.mean(delta_mpe)),
            "mean_moe_deg": float(np.mean(delta_moe)),
            "mpe_95ci_block4": block_interval(delta_mpe),
            "moe_95ci_block4": block_interval(delta_moe),
            "mpe_improved_frames": int(np.sum(delta_mpe < -1e-12)),
            "mpe_damaged_frames": int(np.sum(delta_mpe > 1e-12)),
            "moe_improved_frames": int(np.sum(delta_moe < -1e-12)),
            "moe_damaged_frames": int(np.sum(delta_moe > 1e-12)),
            "both_improved_frames": int(sum(row["both_improved"] for row in per_frame)),
        },
        "solver_failures_with_finite_pose": int(failures),
        "per_frame": per_frame,
    }


def evaluate_pixels(records, rows_by_frame, lidar_cache):
    peak_errors, refined_errors = [], []
    total, visible = 0, 0
    for record in records:
        frame_id = str(record["frame_id"])
        row = rows_by_frame[frame_id]
        with np.load(Path(lidar_cache) / (frame_id + ".npz")) as data:
            gt = np.asarray(data["GT"], dtype=np.float64)
        points = np.asarray(record["points"], dtype=np.float64)
        cameras = np.asarray(record["cameras"], dtype=np.int64)
        peak = np.asarray(record["peak_pixels"], dtype=np.float64)
        refined = np.asarray(record["refined_pixels"], dtype=np.float64)
        target, depth = projection_targets(points, gt, cameras, row)
        valid = np.isfinite(target).all(axis=1) & (depth > 0)
        for camera in np.unique(cameras):
            indices = np.where(cameras == camera)[0]
            view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
            mask = np.asarray(np.load(view["mask"]), dtype=bool)
            with Image.open(view["image"]) as image:
                nonblack = np.asarray(image.convert("RGB"), dtype=np.uint8).max(axis=2) > 0
            h, w = mask.shape
            uv = target[indices]
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w - 1) &
                      (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
            x = np.clip(np.floor(uv[:, 0]).astype(np.int64), 0, w - 1)
            y = np.clip(np.floor(uv[:, 1]).astype(np.int64), 0, h - 1)
            valid[indices] &= inside & mask[y, x] & nonblack[y, x]
        total += len(points)
        visible += int(valid.sum())
        if valid.any():
            peak_errors.extend(np.linalg.norm(peak[valid] - target[valid], axis=1).tolist())
            refined_errors.extend(np.linalg.norm(refined[valid] - target[valid], axis=1).tolist())
    peak_errors = np.asarray(peak_errors, dtype=np.float64)
    refined_errors = np.asarray(refined_errors, dtype=np.float64)
    delta = refined_errors - peak_errors
    return {
        "frames": len(records),
        "correspondences_total": int(total),
        "gt_visible_correspondences": int(visible),
        "peak_then_pose_pixel_error_px": summarize(peak_errors),
        "highres_refined_pixel_error_px": summarize(refined_errors),
        "paired_delta_refined_minus_peak_px": float(np.mean(delta)),
        "refined_pixel_better_count": int(np.sum(delta < -1e-12)),
        "refined_pixel_worse_count": int(np.sum(delta > 1e-12)),
        "fraction_at_most_1px": float(np.mean(refined_errors <= 1.)),
        "fraction_at_most_2px": float(np.mean(refined_errors <= 2.)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-frames", type=int, default=32)
    args = parser.parse_args()

    run_path = Path(args.run_output)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    records = run["records"]
    if run["frames"] != args.expected_frames or len(records) != args.expected_frames:
        raise RuntimeError("frozen full-denominator run has wrong frame count")
    if run["protocol"].get("gt_in_runner") is not False:
        raise RuntimeError("prediction runner did not certify GT exclusion")
    if any("GT" in record or "ground_truth" in record for record in records):
        raise RuntimeError("frozen prediction record contains a GT field")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    for row in records:
        if rows_by_frame[str(row["frame_id"])]["split"] not in ("val", "validation"):
            raise RuntimeError("non-validation frame entered evaluation")

    result = {
        "protocol": "separate post-freeze GT evaluator; query GT read only from lidar cache here",
        "run_output_sha256": sha256_file(run_path),
        "run_frames": len(records),
        "geometry_only_control": evaluate_pose(records, args.lidar_cache,
                                                "geometry_only_pose", "geometry_solver"),
        "peak_then_pose": evaluate_pose(records, args.lidar_cache,
                                        "peak_pose", "peak_solver"),
        "highres_patch_pose": evaluate_pose(records, args.lidar_cache,
                                             "refined_pose", "refined_solver"),
        "highres_patch_pose_vs_peak_then_pose": None,
        "pixel_endpoint_metrics": evaluate_pixels(records, rows_by_frame, args.lidar_cache),
        "interval_note": "descriptive 95% bootstrap intervals over contiguous blocks of four adjacent frames; all frames share one sequence",
    }
    high = {row["frame_id"]: row for row in result["highres_patch_pose"]["per_frame"]}
    peak = {row["frame_id"]: row for row in result["peak_then_pose"]["per_frame"]}
    delta_t = np.asarray([high[key]["candidate_mpe_m"] - peak[key]["candidate_mpe_m"] for key in high])
    delta_r = np.asarray([high[key]["candidate_moe_deg"] - peak[key]["candidate_moe_deg"] for key in high])
    result["highres_patch_pose_vs_peak_then_pose"] = {
        "denominator": int(len(delta_t)),
        "mean_delta_mpe_m": float(delta_t.mean()),
        "mean_delta_moe_deg": float(delta_r.mean()),
        "mpe_95ci_block4": block_interval(delta_t),
        "moe_95ci_block4": block_interval(delta_r),
        "mpe_improved_frames": int(np.sum(delta_t < -1e-12)),
        "mpe_damaged_frames": int(np.sum(delta_t > 1e-12)),
        "moe_improved_frames": int(np.sum(delta_r < -1e-12)),
        "moe_damaged_frames": int(np.sum(delta_r > 1e-12)),
        "both_improved_frames": int(np.sum((delta_t < -1e-12) & (delta_r < -1e-12))),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
