import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import nre_scoremap_pose_runner as nre


CHI2_2_THRESHOLDS = {
    "50_percent": 1.3862943611198906,
    "80_percent": 3.2188758248682006,
    "90_percent": 4.605170185988092,
    "95_percent": 5.991464547107979,
}


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


def pose_error(pose, ground_truth):
    delta = pose[:3, :3].T @ ground_truth[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.) * .5, -1., 1.)
    return (float(np.linalg.norm(pose[:3, 3] - ground_truth[:3, 3])),
            float(np.degrees(np.arccos(cosine))))


def evaluate_pose(records, lidar_cache, pose_key, solver_key):
    translations, rotations, baseline_t, baseline_r = [], [], [], []
    failures = 0
    per_frame = []
    for record in records:
        frame_id = str(record["frame_id"])
        with np.load(Path(lidar_cache) / (frame_id + ".npz")) as data:
            gt = np.asarray(data["GT"], dtype=np.float64)
        pose = np.asarray(record[pose_key], dtype=np.float64)
        baseline = np.asarray(record["baseline_pose"], dtype=np.float64)
        translation, rotation = pose_error(pose, gt)
        base_translation, base_rotation = pose_error(baseline, gt)
        solver = record.get(solver_key, {"success": True})
        failures += int(not solver["success"])
        translations.append(translation)
        rotations.append(rotation)
        baseline_t.append(base_translation)
        baseline_r.append(base_rotation)
        per_frame.append({
            "frame_id": frame_id,
            "candidate_mpe_m": translation,
            "candidate_moe_deg": rotation,
            "baseline_mpe_m": base_translation,
            "baseline_moe_deg": base_rotation,
            "delta_mpe_m": translation - base_translation,
            "delta_moe_deg": rotation - base_rotation,
        })
    delta_t = np.asarray(translations) - np.asarray(baseline_t)
    delta_r = np.asarray(rotations) - np.asarray(baseline_r)
    return {
        "denominator": len(records),
        "candidate": {"mpe_m": summarize(translations), "moe_deg": summarize(rotations)},
        "baseline": {"mpe_m": summarize(baseline_t), "moe_deg": summarize(baseline_r)},
        "paired_delta_vs_geometry": {
            "mean_mpe_m": float(delta_t.mean()),
            "mean_moe_deg": float(delta_r.mean()),
            "mpe_95ci_block4": block_interval(delta_t),
            "moe_95ci_block4": block_interval(delta_r),
            "mpe_improved_frames": int(np.sum(delta_t < -1e-12)),
            "mpe_damaged_frames": int(np.sum(delta_t > 1e-12)),
            "moe_improved_frames": int(np.sum(delta_r < -1e-12)),
            "moe_damaged_frames": int(np.sum(delta_r > 1e-12)),
        },
        "solver_failures_with_finite_pose": failures,
        "per_frame": per_frame,
    }


def target_pixels(points, pose, cameras, row):
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = np.full(len(points), np.nan, dtype=np.float64)
    for camera in np.unique(cameras):
        indices = np.where(cameras == camera)[0]
        view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
        uv, z = nre.project_world(points[indices], pose,
                                  np.asarray(view["camera_to_body"], dtype=np.float64),
                                  np.loadtxt(view["calibration"]).astype(np.float64))
        pixels[indices] = uv
        depth[indices] = z
    return pixels, depth


def regularize_precision(precision, floor):
    precision = .5 * (precision + precision.transpose(0, 2, 1))
    values, vectors = np.linalg.eigh(precision)
    values = np.maximum(values, floor)
    return (vectors * values[:, None, :]) @ vectors.transpose(0, 2, 1)


def uncertainty_summary(errors, precision):
    sign, logdet = np.linalg.slogdet(precision)
    if not np.all(sign > 0):
        raise RuntimeError("uncertainty evaluation received non-SPD precision matrices")
    mahal = np.einsum("ni,nij,nj->n", errors, precision, errors)
    nll = .5 * mahal - .5 * logdet
    return {
        "correspondences": int(len(errors)),
        "gaussian_nll_without_constant": summarize(nll),
        "mahalanobis_squared_mean": float(np.mean(mahal)),
        "mahalanobis_squared_median": float(np.median(mahal)),
        "empirical_coverage": {
            key: float(np.mean(mahal <= threshold))
            for key, threshold in CHI2_2_THRESHOLDS.items()
        },
        "expected_coverage": {key: float(value) for key, value in
                              zip(CHI2_2_THRESHOLDS, (.5, .8, .9, .95))},
    }


def evaluate_correspondences(records, rows_by_frame, lidar_cache, precision_floor):
    all_errors, base_precisions, predicted_precisions = [], [], []
    endpoint_errors, peak_endpoint_errors = [], []
    total, visible_total = 0, 0
    for record in records:
        frame_id = str(record["frame_id"])
        row = rows_by_frame[frame_id]
        with np.load(Path(lidar_cache) / (frame_id + ".npz")) as data:
            gt_pose = np.asarray(data["GT"], dtype=np.float64)
        points = np.asarray(record["points"], dtype=np.float64)
        cameras = np.asarray(record["cameras"], dtype=np.int64)
        target, depth = target_pixels(points, gt_pose, cameras, row)
        visible = np.isfinite(target).all(axis=1) & (depth > 0)
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
            visible[indices] &= inside & mask[y, x] & nonblack[y, x]
        total += len(points)
        visible_total += int(visible.sum())
        refined = np.asarray(record["refined_pixels"], dtype=np.float64)
        peak = np.asarray(record["peak_pixels"], dtype=np.float64)
        errors = refined[visible] - target[visible]
        all_errors.append(errors)
        endpoint_errors.extend(np.linalg.norm(errors, axis=1).tolist())
        peak_endpoint_errors.extend(np.linalg.norm(peak[visible] - target[visible], axis=1).tolist())
        base = np.asarray(record["original_peak_precisions"], dtype=np.float64)
        predicted = np.asarray(record["predicted_precisions"], dtype=np.float64)
        base_precisions.append(regularize_precision(base, precision_floor)[visible])
        predicted_precisions.append(predicted[visible])
    errors = np.concatenate(all_errors)
    base_precision = np.concatenate(base_precisions)
    predicted_precision = np.concatenate(predicted_precisions)
    return {
        "correspondences_total": int(total),
        "gt_visible_correspondences": int(visible_total),
        "highres_refined_endpoint_error_px": summarize(endpoint_errors),
        "peak_endpoint_error_px": summarize(peak_endpoint_errors),
        "pixel_endpoint_changed_by_covariance_training": False,
        "regularized_original_precision": uncertainty_summary(errors, base_precision),
        "learned_full_precision": uncertainty_summary(errors, predicted_precision),
        "precision_floor_px_minus2": float(precision_floor),
    }


def paired_comparison(primary, reference):
    left = {row["frame_id"]: row for row in primary["per_frame"]}
    right = {row["frame_id"]: row for row in reference["per_frame"]}
    if set(left) != set(right):
        raise RuntimeError("paired pose runs have different frame denominators")
    delta_t = np.asarray([left[key]["candidate_mpe_m"] - right[key]["candidate_mpe_m"]
                          for key in left])
    delta_r = np.asarray([left[key]["candidate_moe_deg"] - right[key]["candidate_moe_deg"]
                          for key in left])
    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--reference-pixel-run", required=True)
    parser.add_argument("--covariance-checkpoint", required=True)
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-frames", type=int, default=32)
    args = parser.parse_args()

    run_path = Path(args.run_output)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    records = run["records"]
    if run["frames"] != args.expected_frames or len(records) != args.expected_frames:
        raise RuntimeError("GT-free covariance run has wrong frame denominator")
    if run["protocol"].get("gt_in_runner") is not False:
        raise RuntimeError("prediction runner did not certify GT exclusion")
    if any("GT" in row or "ground_truth" in row for row in records):
        raise RuntimeError("prediction records contain ground truth")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    if any(rows_by_frame[str(item["frame_id"])]["split"] not in ("val", "validation")
           for item in records):
        raise RuntimeError("non-development frame entered post-freeze evaluation")

    checkpoint = __import__("torch").load(args.covariance_checkpoint,
                                            map_location="cpu", weights_only=False)
    precision_floor = float(checkpoint["config"]["precision_floor_px_minus2"])
    methods = {
        "geometry_only_control": evaluate_pose(records, args.lidar_cache,
                                                "geometry_only_pose", "peak_solver"),
        "peak_then_pose": evaluate_pose(records, args.lidar_cache,
                                         "peak_pose", "peak_solver"),
        "frozen_pixel_head_original_precision": evaluate_pose(
            records, args.lidar_cache, "frozen_pixel_pose", "frozen_pixel_solver"),
        "frozen_pixel_head_learned_full_precision": evaluate_pose(
            records, args.lidar_cache, "covariance_pose", "covariance_solver"),
    }
    result = {
        "protocol": "separate post-freeze GT evaluator; validation GT read only after GT-free predictions were frozen",
        "dataset_note": "32 frames belong to one previously used development route and are not an independent test sequence",
        "run_output_sha256": sha256_file(run_path),
        "reference_pixel_run_sha256": sha256_file(args.reference_pixel_run),
        "covariance_checkpoint_sha256": sha256_file(args.covariance_checkpoint),
        "run_frames": len(records),
        "pose": methods,
        "learned_precision_vs_frozen_pixel_original_precision": paired_comparison(
            methods["frozen_pixel_head_learned_full_precision"],
            methods["frozen_pixel_head_original_precision"]),
        "learned_precision_vs_peak_then_pose": paired_comparison(
            methods["frozen_pixel_head_learned_full_precision"],
            methods["peak_then_pose"]),
        "correspondence_uncertainty": evaluate_correspondences(
            records, rows_by_frame, args.lidar_cache, precision_floor),
        "interval_note": "descriptive 95% bootstrap intervals over contiguous blocks of four adjacent frames; all frames share one route",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"evaluation": str(output), "sha256": sha256_file(output),
                      "frames": len(records),
                      "learned_precision_vs_frozen_pixel_original_precision": result[
                          "learned_precision_vs_frozen_pixel_original_precision"]}), flush=True)


if __name__ == "__main__":
    main()
