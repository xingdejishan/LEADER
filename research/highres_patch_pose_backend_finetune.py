import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nre_scoremap_pose_runner as nre
from highres_actual_backend import _ImplicitBackendPose, solve_actual_backend
from highres_patch_pose_refiner import (
    HighResPatchPoseRefiner,
    SEED,
    build_patch_batch,
    build_ro_ma_channels,
    frame_geometry_tensors,
    load_match_frame,
    sha256_file,
)


EXPECTED_INITIAL_SHA256 = "271415f4f252b7960186b64f70adbc829c8916b2b7f16e2cdb3ca02768231399"
POSE_EPOCHS = 8
LEARNING_RATE = 1e-4
PIXEL_WEIGHT = .1
GRADIENT_CLIP = 1.


def pose_errors(predicted, target):
    translation = torch.linalg.vector_norm(predicted[:3, 3] - target[:3, 3])
    relative = predicted[:3, :3] @ target[:3, :3].transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.) * .5).clamp(-1., 1.)
    sine_vector = torch.stack((relative[2, 1] - relative[1, 2],
                               relative[0, 2] - relative[2, 0],
                               relative[1, 0] - relative[0, 1])) * .5
    sine = torch.linalg.vector_norm(sine_vector)
    rotation = torch.atan2(sine, cosine) * (180. / math.pi)
    return translation, rotation


def train_scales(frames, lidar_cache):
    translation_errors, rotation_errors, pixel_errors = [], [], []
    for item in frames:
        frame = item["frame"]
        with np.load(Path(lidar_cache) / (item["frame_id"] + ".npz")) as cache:
            gt = torch.as_tensor(np.asarray(cache["GT"], dtype=np.float64), dtype=torch.float64)
        baseline = torch.as_tensor(frame["baseline_pose"], dtype=torch.float64)
        translation, rotation = pose_errors(baseline, gt)
        translation_errors.append(float(translation))
        rotation_errors.append(float(rotation))
        visible = np.asarray(frame["gt_visible"], dtype=bool)
        if visible.any():
            delta = np.asarray(frame["peaks"], dtype=np.float64)[visible] - \
                np.asarray(frame["gt_pixels"], dtype=np.float64)[visible]
            pixel_errors.extend(np.linalg.norm(delta, axis=1).tolist())
    scales = {
        "translation_m": max(float(np.mean(translation_errors)), 1e-6),
        "rotation_deg": max(float(np.mean(rotation_errors)), 1e-6),
        "pixel_endpoint_px": max(float(np.mean(pixel_errors)), 1e-6),
    }
    return scales


def validate_materialized_frame(frame, row, lidar_cache, manifest_sha):
    if row["split"] != "train":
        raise RuntimeError("non-train frame reached fine-tuning: " + str(row["frame_id"]))
    if str(frame["frame_id"]) != str(row["frame_id"]):
        raise RuntimeError("materialized frame identity mismatch")
    if frame.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("materialized cache manifest hash mismatch: " + str(row["frame_id"]))
    lidar_path = Path(lidar_cache) / (str(row["frame_id"]) + ".npz")
    if frame.get("training_lidar_cache_sha256") != sha256_file(lidar_path):
        raise RuntimeError("materialized cache LiDAR hash mismatch: " + str(row["frame_id"]))
    if len(frame["points"]) != len(frame["gt_visible"]):
        raise RuntimeError("materialized label count mismatch: " + str(row["frame_id"]))
    if not np.asarray(frame["gt_visible"], dtype=bool).any():
        raise RuntimeError("training frame has no visible pixel labels: " + str(row["frame_id"]))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def train(args):
    torch.set_num_threads(16)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    manifest_sha = sha256_file(manifest_path)
    initial_path = Path(args.initial_checkpoint)
    initial_sha = sha256_file(initial_path)
    if initial_sha != EXPECTED_INITIAL_SHA256:
        raise RuntimeError("initial checkpoint hash differs from the frozen high-resolution head")
    initial = torch.load(initial_path, map_location="cpu")
    if initial.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("initial checkpoint was trained with another manifest")
    if initial.get("config", {}).get("validation_gt_used_for_training") is not False:
        raise RuntimeError("initial checkpoint validation isolation is not certified")

    original_inputs = initial.get("training_inputs", [])
    if len(original_inputs) != args.expected_train_frames:
        raise RuntimeError("initial checkpoint has an unexpected training input count")
    training_frames = []
    for item in original_inputs:
        frame_id = str(item["frame_id"])
        if frame_id not in rows_by_frame:
            raise RuntimeError("training frame missing from manifest: " + frame_id)
        cache_path = Path(args.train_input_dir) / (frame_id + ".npz")
        if sha256_file(cache_path) != item["cache_sha256"]:
            raise RuntimeError("materialized training input hash mismatch: " + frame_id)
        frame = load_match_frame(cache_path)
        validate_materialized_frame(frame, rows_by_frame[frame_id], args.lidar_cache, manifest_sha)
        training_frames.append({"frame_id": frame_id, "frame": frame,
                                "row": rows_by_frame[frame_id], "cache_path": str(cache_path),
                                "cache_sha256": item["cache_sha256"]})
    if len(training_frames) != args.expected_train_frames:
        raise RuntimeError("training frame denominator changed")

    scales = train_scales(training_frames, args.lidar_cache)
    model = HighResPatchPoseRefiner().to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, POSE_EPOCHS + 1):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(training_frames))
        rows_for_epoch = []
        for position in order:
            item = training_frames[int(position)]
            frame = item["frame"]
            row = item["row"]
            row_id = str(row["frame_id"])
            ref_patches, query_patches, candidate_valid = build_patch_batch(
                frame, row, rows_by_frame, device)
            roma_channels = build_ro_ma_channels(frame, device)
            geometry = frame_geometry_tensors(frame, row, device)
            gt_visible = torch.as_tensor(frame["gt_visible"], dtype=torch.bool, device=device)
            gt_pixels = torch.as_tensor(frame["gt_pixels"], dtype=torch.float32, device=device)
            peak_pixels = torch.as_tensor(frame["peaks"], dtype=torch.float32, device=device)
            delta_pixels, _ = model(ref_patches, query_patches, roma_channels, candidate_valid)
            corrected_pixels = peak_pixels + delta_pixels
            predicted_pose, solver = solve_actual_backend(
                frame, row, corrected_pixels, geometry, device)
            with np.load(Path(args.lidar_cache) / (row_id + ".npz")) as cache:
                gt_pose = torch.as_tensor(np.asarray(cache["GT"], dtype=np.float64),
                                          dtype=torch.float64, device=device)
            gt_pose_loss = pose_errors(predicted_pose, gt_pose)
            pose_loss = (gt_pose_loss[0] / scales["translation_m"] +
                         gt_pose_loss[1] / scales["rotation_deg"])
            pixel_loss_values = F.smooth_l1_loss(
                corrected_pixels[gt_visible], gt_pixels[gt_visible], beta=1., reduction="none").sum(dim=1)
            pixel_loss = pixel_loss_values.mean() / scales["pixel_endpoint_px"]
            loss = pose_loss.to(torch.float32) + PIXEL_WEIGHT * pixel_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite final-pose training loss: " + row_id)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            backward = dict(_ImplicitBackendPose.last_backward)
            if not backward or not np.isfinite(backward.get("pixel_gradient_norm", np.nan)):
                raise RuntimeError("implicit backend gradient was not produced: " + row_id)
            model_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(model_gradient_norm):
                raise RuntimeError("non-finite refiner gradient: " + row_id)
            optimizer.step()
            rows_for_epoch.append({
                "frame_id": row_id,
                "loss_pose_normalized": float(pose_loss.detach().cpu()),
                "loss_pixel_normalized": float(pixel_loss.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
                "mpe_m": float(gt_pose_loss[0].detach().cpu()),
                "moe_deg": float(gt_pose_loss[1].detach().cpu()),
                "solver_success": solver["success"],
                "solver_iterations": solver["iterations"],
                "solver_projected_kkt_inf": solver["projected_kkt_inf"],
                "solver_objective_parity_abs": solver["objective_parity_abs"],
                "solver_active_dimensions": solver["active_dimensions"],
                "implicit_free_dimensions": backward["free_dimensions"],
                "implicit_hessian_condition": backward["reduced_hessian_condition"],
                "implicit_pixel_gradient_norm": backward["pixel_gradient_norm"],
                "model_gradient_norm_before_clip": float(model_gradient_norm.detach().cpu()),
                "pixel_error_mean_px": float(torch.linalg.vector_norm(
                    corrected_pixels[gt_visible] - gt_pixels[gt_visible], dim=1).mean().detach().cpu()),
            })
            print(json.dumps({"stage": "backend_finetune", "epoch": epoch,
                              "frame": row_id, "loss": rows_for_epoch[-1]["loss_total"],
                              "mpe_m": rows_for_epoch[-1]["mpe_m"],
                              "moe_deg": rows_for_epoch[-1]["moe_deg"],
                              "solver_success": solver["success"],
                              "kkt_inf": solver["projected_kkt_inf"],
                              "implicit_free_dimensions": backward["free_dimensions"]}), flush=True)

        epoch_summary = {
            "epoch": epoch,
            "frames": len(rows_for_epoch),
            "mean_pose_loss_normalized": float(np.mean([row["loss_pose_normalized"] for row in rows_for_epoch])),
            "mean_pixel_loss_normalized": float(np.mean([row["loss_pixel_normalized"] for row in rows_for_epoch])),
            "mean_total_loss": float(np.mean([row["loss_total"] for row in rows_for_epoch])),
            "mean_train_mpe_m": float(np.mean([row["mpe_m"] for row in rows_for_epoch])),
            "mean_train_moe_deg": float(np.mean([row["moe_deg"] for row in rows_for_epoch])),
            "mean_pixel_error_px": float(np.mean([row["pixel_error_mean_px"] for row in rows_for_epoch])),
            "solver_nonconverged_finite_frames": int(sum(not row["solver_success"] for row in rows_for_epoch)),
            "solver_max_objective_parity_abs": float(max(row["solver_objective_parity_abs"] for row in rows_for_epoch)),
            "solver_mean_projected_kkt_inf": float(np.mean([row["solver_projected_kkt_inf"] for row in rows_for_epoch])),
            "solver_max_projected_kkt_inf": float(max(row["solver_projected_kkt_inf"] for row in rows_for_epoch)),
            "active_bound_dimensions": int(sum(row["solver_active_dimensions"] for row in rows_for_epoch)),
            "implicit_mean_free_dimensions": float(np.mean([row["implicit_free_dimensions"] for row in rows_for_epoch])),
            "implicit_max_hessian_condition": float(max(row["implicit_hessian_condition"] for row in rows_for_epoch)),
            "implicit_mean_pixel_gradient_norm": float(np.mean([row["implicit_pixel_gradient_norm"] for row in rows_for_epoch])),
            "per_frame": rows_for_epoch,
        }
        history.append(epoch_summary)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "training_protocol": "continued high-resolution patch-head training; deployed SciPy L-BFGS-B forward with KKT implicit differentiation of the same bounded LiDAR-camera objective; final pose MPE/MOE loss plus weak normalized pixel supervision",
            "config": {
                "method": "actual_backend_implicit_finetune",
                "epochs": POSE_EPOCHS,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": 1e-4,
                "gradient_clip": GRADIENT_CLIP,
                "pixel_weight": PIXEL_WEIGHT,
                "search_radius_px": 24,
                "seed": SEED,
                "train_frames": len(training_frames),
                "training_loss_scales": scales,
                "validation_gt_used_for_training": False,
                "checkpoint_selection": "fixed final epoch; no validation-based selection",
                "solver_forward": "same nre.optimize_pose L-BFGS-B objective, bounds and termination settings as inference",
                "implicit_backward": "KKT active-set implicit derivative of the locally stationary constrained objective",
            },
            "initial_checkpoint_sha256": initial_sha,
            "manifest_sha256": manifest_sha,
            "roma_feature_state_sha256": initial["roma_feature_state_sha256"],
            "training_inputs": [{"frame_id": item["frame_id"],
                                 "cache_sha256": item["cache_sha256"]} for item in training_frames],
            "history": history,
        }
        epoch_checkpoint = output_dir / ("epoch_%02d.pt" % epoch)
        torch.save(checkpoint, epoch_checkpoint)
        write_json(output_dir / "training_progress.json", {
            "protocol": checkpoint["training_protocol"],
            "completed_epochs": epoch,
            "scales": scales,
            "history": history,
            "latest_checkpoint": str(epoch_checkpoint),
            "latest_checkpoint_sha256": sha256_file(epoch_checkpoint),
        })
        print(json.dumps({"stage": "epoch_complete", **{key: value for key, value in epoch_summary.items()
                                                            if key != "per_frame"},
                          "checkpoint_sha256": sha256_file(epoch_checkpoint)}), flush=True)

    final_path = output_dir / "highres_patch_pose_head.pt"
    torch.save(checkpoint, final_path)
    report = {
        "protocol": checkpoint["training_protocol"],
        "training_split": "manifest split=train only; validation frames and GT are never loaded by fine-tuning",
        "fine_tune_from_sha256": initial_sha,
        "checkpoint_sha256": sha256_file(final_path),
        "manifest_sha256": manifest_sha,
        "train_frames": len(training_frames),
        "train_correspondences": int(sum(len(item["frame"]["points"]) for item in training_frames)),
        "visible_pixel_labels": int(sum(np.asarray(item["frame"]["gt_visible"], dtype=bool).sum()
                                         for item in training_frames)),
        "epochs": POSE_EPOCHS,
        "selection": "fixed final epoch; no validation-based stopping or checkpoint selection",
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                      "weight_decay": 1e-4, "gradient_clip": GRADIENT_CLIP},
        "head": {"search_radius_px": 24, "reference_patch_px": 49,
                 "query_patch_px": 65, "confidence_head": False},
        "loss": {
            "pose": "final deployed-backend translation norm / training mean baseline MPE plus SO(3) geodesic degrees / training mean baseline MOE",
            "pixel": "visible-point smooth L1 endpoint loss / training mean baseline pixel endpoint error",
            "pixel_weight": PIXEL_WEIGHT,
            "training_scales": scales,
        },
        "deployed_backend": {
            "implementation": "nre.optimize_pose using SciPy L-BFGS-B",
            "objective": "quadratic precision-weighted visual reprojection plus LiDAR information-matrix prior and existing negative-depth penalty",
            "bounds": {"translation_each_axis_m": 0.1, "rotation_each_axis_deg": 1.0},
            "robust_loss_in_actual_objective": False,
            "note": "The frozen deployed objective has no robust visual loss term; no new robust term was added.",
            "implicit_gradient": "KKT active-set derivative over free dimensions; active bound dimensions receive zero local derivative",
        },
        "validation": {
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "note": "The 32-frame set is a repeatedly used development set from the same sequence, not an independent test sequence.",
        },
        "training_inputs": [{"frame_id": item["frame_id"],
                             "cache_sha256": item["cache_sha256"]} for item in training_frames],
        "history": history,
        "checkpoint_path": str(final_path),
    }
    write_json(output_dir / "training_report.json", report)
    print(json.dumps({"stage": "complete", "report": str(output_dir / "training_report.json"),
                      "checkpoint_sha256": report["checkpoint_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--train-input-dir", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/"
                                                        "highres_patch_pose_20260923/train_inputs"))
    parser.add_argument("--initial-checkpoint", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/"
                                                          "research/results/highres_patch_pose_20260923/highres_patch_pose_head.pt"))
    parser.add_argument("--output-dir", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/"
                                                  "highres_patch_pose_backend_ft_20260923"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-frames", type=int, default=47)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
