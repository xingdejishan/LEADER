import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import nre_scoremap_pose_runner as nre
from highres_patch_pose_refiner import (
    HighResPatchPoseRefiner,
    SEED,
    build_patch_batch,
    build_ro_ma_channels,
    load_backend,
    prepare_match_frame,
    sha256_file,
)
from highres_precision_covariance import (
    PrecisionCovarianceHead,
    covariance_features,
    precision_from_raw,
)


def pose_array(value):
    return np.asarray(value, dtype=np.float64)


def run(args):
    torch.set_num_threads(16)
    device = torch.device(args.device)
    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    manifest_sha = sha256_file(manifest_path)
    match_dir = Path(args.validation_match_cache)
    val_rows = [row for row in rows if row["split"] in ("val", "validation") and
                (match_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if len(val_rows) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} development frames, got {len(val_rows)}")

    initial_path = Path(args.initial_checkpoint)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if initial.get("manifest_sha256") != manifest_sha or \
            initial.get("config", {}).get("validation_gt_used_for_training") is not False:
        raise RuntimeError("frozen pixel checkpoint provenance failed")
    covariance_path = Path(args.covariance_checkpoint)
    covariance_checkpoint = torch.load(covariance_path, map_location="cpu", weights_only=False)
    if covariance_checkpoint.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("covariance checkpoint and manifest hashes differ")
    if covariance_checkpoint.get("config", {}).get("validation_gt_used_for_training") is not False:
        raise RuntimeError("covariance checkpoint provenance failed")
    if covariance_checkpoint.get("config", {}).get("frozen_pixel_model_sha256") != sha256_file(initial_path):
        raise RuntimeError("covariance checkpoint was trained from another pixel head")

    pixel_model = HighResPatchPoseRefiner().to(device)
    pixel_model.load_state_dict(initial["model_state_dict"], strict=True)
    pixel_model.eval()
    covariance_head = PrecisionCovarianceHead().to(device)
    covariance_head.load_state_dict(covariance_checkpoint["model_state_dict"], strict=True)
    covariance_head.eval()

    reference_path = Path(args.reference_pixel_run)
    reference_run = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference_run["frames"] != args.expected_frames or \
            reference_run["protocol"].get("gt_in_runner") is not False:
        raise RuntimeError("frozen pixel reference run is not a valid GT-free full-denominator run")
    if reference_run["settings"].get("checkpoint_sha256") != sha256_file(initial_path):
        raise RuntimeError("frozen reference run used another pixel checkpoint")
    reference_records = {str(item["frame_id"]): item for item in reference_run["records"]}
    matcher, full_pool_refine = load_backend(args.baseline_code_root, args.device)
    roma_features = nre.RoMaFeatures(args.device, "precise", 4)
    feature_sha = roma_features.model_sha256()
    if feature_sha != initial["roma_feature_state_sha256"]:
        raise RuntimeError("RoMa feature state differs from the frozen pixel training model")

    records = []
    for row_index, row in enumerate(val_rows):
        frame_id = str(row["frame_id"])
        cache_path = match_dir / (frame_id + ".npz")
        frame = prepare_match_frame(row, rows_by_frame, cache_path, args.lidar_cache,
                                    matcher, full_pool_refine, roma_features,
                                    args.device, SEED + row_index)
        if frame_id not in reference_records:
            raise RuntimeError("frame missing from frozen pixel reference run: " + frame_id)
        reference = reference_records[frame_id]
        if frame.get("match_cache_sha256") != reference.get("match_cache_sha256"):
            raise RuntimeError("validation match cache changed: " + frame_id)
        if not np.array_equal(frame["cameras"].astype(np.int64),
                              np.asarray(reference["cameras"], dtype=np.int64)):
            raise RuntimeError("validation camera assignment changed: " + frame_id)
        if not np.allclose(frame["points"], reference["points"], atol=2e-5, rtol=0):
            raise RuntimeError("validation correspondence geometry changed: " + frame_id)

        ref_patches, query_patches, candidate_valid = build_patch_batch(
            frame, row, rows_by_frame, device)
        roma_channels = build_ro_ma_channels(frame, device)
        with torch.no_grad():
            delta, _, summary = pixel_model(ref_patches, query_patches,
                                            roma_channels, candidate_valid,
                                            return_features=True)
            original_precision = torch.as_tensor(frame["peak_precisions"],
                                                 dtype=torch.float64, device=device)
            inputs, base_precision = covariance_features(summary, original_precision)
            raw = covariance_head(inputs)
            precision, covariance = precision_from_raw(
                raw, base_precision, return_covariance=True)
        refined_pixels = frame["peaks"] + delta.detach().cpu().numpy().astype(np.float64)
        reference_pixels = np.asarray(reference["refined_pixels"], dtype=np.float64)
        pixel_parity = float(np.max(np.abs(refined_pixels - reference_pixels)))
        if pixel_parity > args.pixel_parity_tolerance:
            raise RuntimeError(f"frozen pixel positions changed for {frame_id}: {pixel_parity:.4g}")
        precision_np = precision.detach().cpu().numpy().astype(np.float64)
        covariance_np = covariance.detach().cpu().numpy().astype(np.float64)
        minimum_eigenvalue = float(np.linalg.eigvalsh(precision_np).min())
        if minimum_eigenvalue <= 0 or not np.isfinite(precision_np).all():
            raise RuntimeError("covariance head produced a non-SPD precision: " + frame_id)

        baseline_pose = np.asarray(frame["baseline_pose"], dtype=np.float64)
        covariance_pose, covariance_solver, covariance_initial, covariance_final = nre.optimize_pose(
            baseline_pose, frame["points"], frame["cameras"], row["views"],
            frame["cost_maps"], frame["map_valid"], frame["centers"],
            refined_pixels, precision_np, frame["lidar_information"], "peak")
        reference_baseline_parity = float(np.max(np.abs(
            baseline_pose - pose_array(reference["baseline_pose"]))))
        if reference_baseline_parity > args.pose_parity_tolerance:
            raise RuntimeError("frozen LiDAR baseline differs for " + frame_id)
        if not np.isfinite(covariance_pose).all():
            raise RuntimeError("covariance backend returned a non-finite pose: " + frame_id)

        records.append({
            "frame_id": frame_id,
            "correspondences_in_cache": int(frame["match_cache_rows"]),
            "usable_peaks": int(frame["usable_local_peaks"]),
            "active_cameras": sorted(int(value) for value in np.unique(frame["cameras"])),
            "baseline_pose": baseline_pose.tolist(),
            "geometry_only_pose": reference["geometry_only_pose"],
            "peak_pose": reference["peak_pose"],
            "frozen_pixel_pose": reference["refined_pose"],
            "covariance_pose": covariance_pose.tolist(),
            "peak_solver": reference["peak_solver"],
            "geometry_solver": reference["geometry_solver"],
            "frozen_pixel_solver": reference["refined_solver"],
            "covariance_solver": {
                "success": bool(covariance_solver.success),
                "status": int(covariance_solver.status),
                "iterations": int(getattr(covariance_solver, "nit", 0)),
                "message": str(covariance_solver.message),
                "initial_objective": float(covariance_initial),
                "final_objective": float(covariance_final),
            },
            "pixel_head_parity_max_abs": pixel_parity,
            "baseline_parity_max_abs": reference_baseline_parity,
            "reference_pixels": frame["reference_pixels"].astype(np.float32).tolist(),
            "points": frame["points"].astype(np.float32).tolist(),
            "cameras": frame["cameras"].astype(int).tolist(),
            "peak_pixels": frame["peaks"].astype(np.float32).tolist(),
            "refined_pixels": refined_pixels.astype(np.float32).tolist(),
            "original_peak_precisions": frame["peak_precisions"].astype(np.float32).tolist(),
            "predicted_precisions": precision_np.astype(np.float32).tolist(),
            "predicted_covariances": covariance_np.astype(np.float32).tolist(),
            "lidar_information": frame["lidar_information"].astype(np.float64).tolist(),
            "match_cache_sha256": frame["match_cache_sha256"],
            "lidar_input_sha256": frame["lidar_input_sha256"],
            "score_map_sha256": hashlib.sha256(frame["cost_maps"].tobytes() +
                                                frame["map_valid"].tobytes()).hexdigest(),
        })
        print(json.dumps({"stage": "covariance_inference", "frame": frame_id,
                          "correspondences": len(frame["points"]),
                          "pixel_head_parity_max_abs": pixel_parity,
                          "solver_success": bool(covariance_solver.success),
                          "solver_iterations": int(getattr(covariance_solver, "nit", 0)),
                          "precision_min_eigenvalue": minimum_eigenvalue}), flush=True)

    if len(records) != args.expected_frames:
        raise RuntimeError("development inference denominator changed")
    output = {
        "protocol": {
            "name": "frozen high-resolution pixel head with learned full covariance in the deployed pose backend",
            "dataset": "2012-02-18 validation development sequence; not an independent test sequence",
            "baseline": "SC2-PCR plus two-stage full-pool refinement at 1.2 m and 0.6 m",
            "head": "frozen high-resolution local pixel head plus train-only learned full 2x2 visual precision head",
            "precision": "absolute per-correspondence SPD matrices; no per-frame normalization",
            "pose_backend": "same bounded L-BFGS-B peak-then-pose objective, LiDAR information and bounds",
            "gt_in_runner": False,
            "development_gt_used_for_training": False,
            "development_checkpoint_selection": False,
        },
        "settings": {
            "manifest_sha256": manifest_sha,
            "roma_feature_state_sha256": feature_sha,
            "frozen_pixel_checkpoint_sha256": sha256_file(initial_path),
            "covariance_checkpoint_sha256": sha256_file(covariance_path),
            "reference_pixel_run_sha256": sha256_file(reference_path),
            "frames": len(records),
            "expected_frames": args.expected_frames,
            "device": args.device,
            "pixel_parity_tolerance": args.pixel_parity_tolerance,
            "pose_parity_tolerance": args.pose_parity_tolerance,
        },
        "frames": len(records),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "complete", "run": str(output_path),
                      "sha256": sha256_file(output_path), "frames": len(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--validation-match-cache", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/prevoxel_multiview/results/roma_controlled_validation_matches")
    parser.add_argument("--baseline-code-root", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/glace-local/code")
    parser.add_argument("--initial-checkpoint", default="research/results/highres_patch_pose_backend_ft_20260923/highres_patch_pose_head.pt")
    parser.add_argument("--covariance-checkpoint", default="research/results/highres_precision_covariance_20260924/precision_covariance_head.pt")
    parser.add_argument("--reference-pixel-run", default="research/results/highres_patch_pose_backend_ft_20260923/validation_run.json")
    parser.add_argument("--output", default="research/results/highres_precision_covariance_20260924/validation_run.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-frames", type=int, default=32)
    parser.add_argument("--pixel-parity-tolerance", type=float, default=1e-5)
    parser.add_argument("--pose-parity-tolerance", type=float, default=1e-7)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
