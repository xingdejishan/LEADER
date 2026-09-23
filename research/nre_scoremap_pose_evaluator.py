import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_error(pose, ground_truth):
    delta = pose[:3, :3].T @ ground_truth[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.) / 2., -1., 1.)
    return float(np.linalg.norm(pose[:3, 3] - ground_truth[:3, 3])), float(np.degrees(np.arccos(cosine)))


def rotation_difference(first, second):
    delta = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.) / 2., -1., 1.)
    return float(np.degrees(np.arccos(cosine)))


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


def evaluate_method(records, lidar_cache, pose_key, solver_key):
    baseline_errors, candidate_errors = [], []
    per_frame, solver_failures = [], 0
    for record in records:
        frame_id = str(record["frame_id"])
        with np.load(Path(lidar_cache) / (frame_id + ".npz")) as data:
            ground_truth = np.asarray(data["GT"], dtype=np.float64)
        baseline_pose = np.asarray(record["baseline_pose"], dtype=np.float64)
        candidate_pose = np.asarray(record[pose_key], dtype=np.float64)
        baseline = pose_error(baseline_pose, ground_truth)
        candidate = pose_error(candidate_pose, ground_truth)
        baseline_errors.append(baseline)
        candidate_errors.append(candidate)
        solver = record[solver_key]
        solver_failures += int(solver is not None and not solver["success"])
        per_frame.append({"frame_id": frame_id, "baseline_mpe_m": baseline[0], "candidate_mpe_m": candidate[0],
                          "delta_mpe_m": candidate[0] - baseline[0], "baseline_moe_deg": baseline[1],
                          "candidate_moe_deg": candidate[1], "delta_moe_deg": candidate[1] - baseline[1],
                          "both_improved": bool(candidate[0] < baseline[0] and candidate[1] < baseline[1])})
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    candidate_errors = np.asarray(candidate_errors, dtype=np.float64)
    delta_translation = candidate_errors[:, 0] - baseline_errors[:, 0]
    delta_rotation = candidate_errors[:, 1] - baseline_errors[:, 1]
    return {
        "denominator": int(len(records)),
        "baseline": {"mpe_m": summarize(baseline_errors[:, 0]), "moe_deg": summarize(baseline_errors[:, 1]),
                     "success_1m_2deg": int(np.sum((baseline_errors[:, 0] < 1.) & (baseline_errors[:, 1] < 2.)))},
        "candidate": {"mpe_m": summarize(candidate_errors[:, 0]), "moe_deg": summarize(candidate_errors[:, 1]),
                      "success_1m_2deg": int(np.sum((candidate_errors[:, 0] < 1.) & (candidate_errors[:, 1] < 2.)))},
        "paired_delta": {"mean_mpe_m": float(np.mean(delta_translation)),
                         "mean_moe_deg": float(np.mean(delta_rotation)),
                         "mpe_95ci_block4": block_interval(delta_translation),
                         "moe_95ci_block4": block_interval(delta_rotation),
                         "mpe_improved_frames": int(np.sum(delta_translation < -1e-12)),
                         "mpe_damaged_frames": int(np.sum(delta_translation > 1e-12)),
                         "moe_improved_frames": int(np.sum(delta_rotation < -1e-12)),
                         "moe_damaged_frames": int(np.sum(delta_rotation > 1e-12)),
                         "both_improved_frames": int(sum(row["both_improved"] for row in per_frame))},
        "solver_failures_with_finite_pose": int(solver_failures),
        "per_frame": per_frame,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-baseline", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/"
                                                          "research/prevoxel_multiview/results/roma_controlled_validation_cache_build.json"))
    args = parser.parse_args()
    run_path = Path(args.run_output)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    records = run["records"]
    if len(records) != run["frames"]:
        raise RuntimeError("run record count differs from frozen denominator")
    result = {
        "protocol": "separate post-freeze evaluator; query GT is read only from LiDAR cache here",
        "run_output_sha256": sha256_file(run_path),
        "run_frames": len(records),
        "geometry_only_control": evaluate_method(records, args.lidar_cache, "geometry_only_pose", "geometry_solver"),
        "scoremap": evaluate_method(records, args.lidar_cache, "scoremap_pose", "scoremap_solver"),
        "peak_then_pose": evaluate_method(records, args.lidar_cache, "peak_pose", "peak_solver"),
        "interval_note": "descriptive 95% bootstrap intervals over contiguous blocks of four adjacent frames; all frames share one sequence",
    }
    scoremap_rows = {row["frame_id"]: row for row in result["scoremap"]["per_frame"]}
    peak_rows = {row["frame_id"]: row for row in result["peak_then_pose"]["per_frame"]}
    paired_translation = np.asarray([scoremap_rows[frame_id]["candidate_mpe_m"] -
                                     peak_rows[frame_id]["candidate_mpe_m"] for frame_id in scoremap_rows])
    paired_rotation = np.asarray([scoremap_rows[frame_id]["candidate_moe_deg"] -
                                  peak_rows[frame_id]["candidate_moe_deg"] for frame_id in scoremap_rows])
    result["scoremap_vs_peak_then_pose"] = {
        "denominator": int(len(paired_translation)),
        "mean_delta_mpe_m": float(np.mean(paired_translation)),
        "mean_delta_moe_deg": float(np.mean(paired_rotation)),
        "mpe_95ci_block4": block_interval(paired_translation),
        "moe_95ci_block4": block_interval(paired_rotation),
        "scoremap_mpe_better_frames": int(np.sum(paired_translation < -1e-12)),
        "scoremap_mpe_worse_frames": int(np.sum(paired_translation > 1e-12)),
        "scoremap_moe_better_frames": int(np.sum(paired_rotation < -1e-12)),
        "scoremap_moe_worse_frames": int(np.sum(paired_rotation > 1e-12)),
        "both_metrics_better_for_scoremap": int(np.sum((paired_translation < -1e-12) & (paired_rotation < -1e-12))),
        "both_metrics_worse_for_scoremap": int(np.sum((paired_translation > 1e-12) & (paired_rotation > 1e-12))),
    }
    reference = Path(args.reference_baseline)
    if reference.is_file():
        reference_data = json.loads(reference.read_text(encoding="utf-8"))
        by_id = {str(row["frame_id"]): np.asarray(row["leader_pose"], dtype=np.float64)
                 for row in reference_data["records"]}
        matrix_differences, translation_differences, rotation_differences = [], [], []
        for record in records:
            frame_id = str(record["frame_id"])
            if frame_id in by_id:
                current = np.asarray(record["baseline_pose"], dtype=np.float64)
                previous = by_id[frame_id]
                matrix_differences.append(float(np.max(np.abs(current - previous))))
                translation_differences.append(float(np.linalg.norm(current[:3, 3] - previous[:3, 3])))
                rotation_differences.append(rotation_difference(current, previous))
        result["baseline_parity"] = {"compared_frames": len(matrix_differences),
                                      "max_abs_pose_entry_difference": max(matrix_differences) if matrix_differences else None,
                                      "translation_delta_m": summarize(translation_differences),
                                      "rotation_delta_deg": summarize(rotation_differences),
                                      "reference_file_sha256": sha256_file(reference)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
