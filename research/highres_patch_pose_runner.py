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
    load_match_frame,
    prepare_match_frame,
    sha256_file,
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
    match_cache_dir = Path(args.validation_match_cache)
    eval_rows = [row for row in rows if row["split"] in ("val", "validation") and
                 (match_cache_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if len(eval_rows) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} validation frames, got {len(eval_rows)}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"].get("validation_gt_used_for_training") is not False:
        raise RuntimeError("checkpoint provenance does not certify validation GT exclusion")
    if checkpoint["manifest_sha256"] != manifest_sha:
        raise RuntimeError("checkpoint and inference manifest hashes differ")
    model = HighResPatchPoseRefiner().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    reference_run_path = Path(args.reference_peak_run)
    reference_run = json.loads(reference_run_path.read_text(encoding="utf-8"))
    reference_records = {str(row["frame_id"]): row for row in reference_run["records"]}
    matcher, full_pool_refine = load_backend(args.baseline_code_root, args.device)
    roma_features = nre.RoMaFeatures(args.device, "precise", 4)
    feature_sha = roma_features.model_sha256()
    if feature_sha != checkpoint["roma_feature_state_sha256"]:
        raise RuntimeError("frozen RoMa feature state differs from the training materialization")

    records = []
    for row_index, row in enumerate(eval_rows):
        frame_id = str(row["frame_id"])
        cache_path = match_cache_dir / (frame_id + ".npz")
        frame = prepare_match_frame(row, rows_by_frame, cache_path, args.lidar_cache,
                                    matcher, full_pool_refine, roma_features,
                                    args.device, SEED + row_index)
        if frame_id not in reference_records:
            raise RuntimeError("frame missing from frozen peak baseline: " + frame_id)

        ref_patches, query_patches, candidate_valid = build_patch_batch(
            frame, row, rows_by_frame, device)
        roma_channels = build_ro_ma_channels(frame, device)
        with torch.inference_mode():
            delta, _ = model(ref_patches, query_patches, roma_channels, candidate_valid)
        peak_pixels = np.asarray(frame["peaks"], dtype=np.float64)
        refined_pixels = peak_pixels + delta.detach().cpu().numpy().astype(np.float64)
        baseline_pose = np.asarray(frame["baseline_pose"], dtype=np.float64)
        geometry_pose = baseline_pose.copy()
        peak_pose, peak_solver, _, _ = nre.optimize_pose(
            baseline_pose, frame["points"], frame["cameras"], row["views"],
            frame["cost_maps"], frame["map_valid"], frame["centers"],
            peak_pixels, frame["peak_precisions"], frame["lidar_information"], "peak")
        refined_pose, refined_solver, _, _ = nre.optimize_pose(
            baseline_pose, frame["points"], frame["cameras"], row["views"],
            frame["cost_maps"], frame["map_valid"], frame["centers"],
            refined_pixels, frame["peak_precisions"], frame["lidar_information"], "peak")

        prior = reference_records[frame_id]
        baseline_parity = float(np.max(np.abs(baseline_pose - pose_array(prior["baseline_pose"]))))
        peak_parity = float(np.max(np.abs(peak_pose - pose_array(prior["peak_pose"]))))
        if baseline_parity > args.parity_tolerance or peak_parity > args.parity_tolerance:
            raise RuntimeError(f"recomputed baseline/peak differs from frozen run for {frame_id}: "
                               f"{baseline_parity:.3g}/{peak_parity:.3g}")

        records.append({
            "frame_id": frame_id,
            "correspondences_in_cache": int(frame["match_cache_rows"]),
            "usable_peaks": int(frame["usable_local_peaks"]),
            "active_cameras": sorted(int(value) for value in np.unique(frame["cameras"])),
            "baseline_pose": baseline_pose.tolist(),
            "geometry_only_pose": geometry_pose.tolist(),
            "peak_pose": peak_pose.tolist(),
            "refined_pose": refined_pose.tolist(),
            "peak_solver": {"success": bool(peak_solver.success), "status": int(peak_solver.status),
                            "iterations": int(getattr(peak_solver, "nit", 0)),
                            "message": str(peak_solver.message)},
            "geometry_solver": {"success": True, "status": 0, "iterations": 0,
                                "message": "analytic identity at the LiDAR baseline"},
            "refined_solver": {"success": bool(refined_solver.success), "status": int(refined_solver.status),
                               "iterations": int(getattr(refined_solver, "nit", 0)),
                               "message": str(refined_solver.message)},
            "baseline_peak_parity_max_abs": baseline_parity,
            "reference_pixels": frame["reference_pixels"].astype(np.float32).tolist(),
            "points": frame["points"].astype(np.float32).tolist(),
            "cameras": frame["cameras"].astype(int).tolist(),
            "peak_pixels": peak_pixels.astype(np.float32).tolist(),
            "refined_pixels": refined_pixels.astype(np.float32).tolist(),
            "peak_precisions": frame["peak_precisions"].astype(np.float32).tolist(),
            "lidar_information": frame["lidar_information"].astype(np.float64).tolist(),
            "lidar_details": frame["lidar_details"],
            "baseline_support": frame["baseline_support"],
            "match_cache_sha256": frame["match_cache_sha256"],
            "lidar_input_sha256": frame["lidar_input_sha256"],
            "score_map_sha256": hashlib.sha256(frame["cost_maps"].tobytes() +
                                                frame["map_valid"].tobytes()).hexdigest(),
        })
        print(json.dumps({"frame": frame_id, "usable_peaks": len(frame["points"]),
                          "baseline_peak_parity_max_abs": max(baseline_parity, peak_parity),
                          "peak_success": bool(peak_solver.success),
                          "refined_success": bool(refined_solver.success)}), flush=True)

    if len(records) != args.expected_frames:
        raise RuntimeError("inference denominator changed during run")
    output = {
        "protocol": {
            "name": "high-resolution raw RGB patch refinement before peak-then-pose",
            "dataset": "2012-02-18 validation development set; not an independent sequence test",
            "baseline": "SC2-PCR followed by two-stage full-pool refinement at 1.2 m and 0.6 m",
            "correspondence_set": "frozen train-map RoMa controlled validation cache",
            "head": "shared full-resolution RGB patch encoder; RoMa local similarity map and candidate validity are input channels; no confidence head",
            "loss_training": "pixel endpoint smooth L1 plus beta=1 pose reprojection smooth L1 after five unrolled Gauss-Newton steps",
            "pose_backend": "same L-BFGS-B peak-then-pose objective, precision matrices, LiDAR prior and bounds for both visual methods",
            "gt_in_runner": False,
            "validation_gt_used_for_training": False,
            "val_epoch_or_checkpoint_selection": False,
        },
        "settings": {
            "manifest_sha256": manifest_sha,
            "roma_feature_state_sha256": feature_sha,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "training_report_sha256": sha256_file(args.training_report),
            "reference_peak_run_sha256": sha256_file(reference_run_path),
            "reference_peak_parity_tolerance": args.parity_tolerance,
            "search_radius_px": int(checkpoint["config"]["search_radius_px"]),
            "frames": len(records),
            "expected_frames": args.expected_frames,
            "device": args.device,
        },
        "frames": len(records),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"frozen_run": str(output_path),
                      "sha256": sha256_file(output_path), "frames": len(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--validation-match-cache", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/"
                                                               "research/prevoxel_multiview/results/roma_controlled_validation_matches"))
    parser.add_argument("--baseline-code-root", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/glace-local/code")
    parser.add_argument("--checkpoint", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923/highres_patch_pose_head.pt")
    parser.add_argument("--training-report", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923/training_report.json")
    parser.add_argument("--reference-peak-run", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/scoremap_pose_nre_20260923_full/run.json")
    parser.add_argument("--output", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923/validation_run.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-frames", type=int, default=32)
    parser.add_argument("--parity-tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
