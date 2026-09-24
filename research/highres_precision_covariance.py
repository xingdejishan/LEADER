import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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


FEATURE_DIM = 112
PRECISION_FLOOR = 1e-4
EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.
POSE_WEIGHT = 1.
NLL_WEIGHT = 1.
MAX_PROJECTED_KKT_INF = .1
RELATIVE_PRECISION_SCALE_BOUND = math.log(2.)
RELATIVE_CORRELATION_BOUND = .95


class PrecisionCovarianceHead(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


def regularize_precision(precision, floor=PRECISION_FLOOR):
    precision = precision.to(torch.float64)
    precision = .5 * (precision + precision.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(precision)
    values = values.clamp_min(float(floor))
    base = (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    square_root = (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return base, square_root, values


def covariance_features(summary, original_precision):
    base_precision, _, _ = regularize_precision(original_precision)
    base_covariance = torch.linalg.inv(base_precision)
    sigma_x = base_covariance[:, 0, 0].clamp_min(1e-12).sqrt()
    sigma_y = base_covariance[:, 1, 1].clamp_min(1e-12).sqrt()
    correlation = base_covariance[:, 0, 1] / (sigma_x * sigma_y).clamp_min(1e-12)
    correlation = correlation.clamp(-.999, .999)
    base_description = torch.stack((sigma_x.log(), sigma_y.log(),
                                    torch.atanh(correlation)), dim=1).to(torch.float32)
    values = (
        summary["reference_descriptor"],
        summary["query_descriptor_mean"],
        summary["query_descriptor_variance"],
        summary["roma_mean"],
        summary["probability_entropy"],
        summary["correlation_moments"],
        summary["offset_covariance"],
        summary["normalized_delta"],
        base_description,
    )
    combined = torch.cat(values, dim=1)
    if combined.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"covariance input dimension changed: {combined.shape[1]}")
    return combined, base_precision


def precision_from_raw(raw, base_precision, return_covariance=False):
    base_precision, square_root, _ = regularize_precision(base_precision)
    raw = raw.to(torch.float64)
    log_scale = RELATIVE_PRECISION_SCALE_BOUND * torch.tanh(raw[:, :2])
    scale = torch.exp(log_scale)
    correlation = RELATIVE_CORRELATION_BOUND * torch.tanh(raw[:, 2])
    relative = torch.zeros((len(raw), 2, 2), dtype=torch.float64, device=raw.device)
    relative[:, 0, 0] = scale[:, 0].square()
    relative[:, 1, 1] = scale[:, 1].square()
    relative[:, 0, 1] = correlation * scale[:, 0] * scale[:, 1]
    relative[:, 1, 0] = relative[:, 0, 1]
    precision = square_root @ relative @ square_root
    precision = .5 * (precision + precision.transpose(-1, -2))
    if not return_covariance:
        return precision
    covariance = torch.linalg.inv(precision)
    return precision, covariance


def pose_errors(predicted, target):
    translation = torch.linalg.vector_norm(predicted[:3, 3] - target[:3, 3])
    relative = predicted[:3, :3] @ target[:3, :3].transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.) * .5).clamp(-1., 1.)
    sine_vector = torch.stack((relative[2, 1] - relative[1, 2],
                               relative[0, 2] - relative[2, 0],
                               relative[1, 0] - relative[0, 1])) * .5
    rotation = torch.atan2(torch.linalg.vector_norm(sine_vector), cosine) * (180. / math.pi)
    return translation, rotation


def precision_nll(pixels, targets, visible, precision):
    error = pixels.to(torch.float64) - targets.to(torch.float64)
    selected_error = error[visible]
    selected_precision = precision[visible]
    sign, logdet = torch.linalg.slogdet(selected_precision)
    if not bool((sign > 0).all()):
        raise RuntimeError("predicted visual precision is not positive definite")
    mahalanobis = torch.einsum("ni,nij,nj->n", selected_error,
                               selected_precision, selected_error)
    return (.5 * mahalanobis - .5 * logdet).mean()


def pose_numpy_error(predicted, target):
    translation = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]))
    relative = predicted[:3, :3] @ target[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.) * .5, -1., 1.)
    sine = .5 * np.linalg.norm(np.array((relative[2, 1] - relative[1, 2],
                                         relative[0, 2] - relative[2, 0],
                                         relative[1, 0] - relative[0, 1])))
    return translation, float(np.degrees(np.arctan2(sine, cosine)))


def checked_training_frames(args, initial, rows_by_frame, manifest_sha):
    if initial.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("frozen pixel checkpoint and manifest hashes differ")
    if initial.get("config", {}).get("validation_gt_used_for_training") is not False:
        raise RuntimeError("frozen pixel checkpoint does not certify development-label isolation")
    training_inputs = initial.get("training_inputs", [])
    if len(training_inputs) != args.expected_train_frames:
        raise RuntimeError("frozen checkpoint training frame denominator changed")
    result = []
    for item in training_inputs:
        frame_id = str(item["frame_id"])
        row = rows_by_frame.get(frame_id)
        if row is None or row["split"] != "train":
            raise RuntimeError("non-train row appears in frozen checkpoint provenance: " + frame_id)
        path = Path(args.train_input_dir) / (frame_id + ".npz")
        if sha256_file(path) != item["cache_sha256"]:
            raise RuntimeError("materialized training cache hash mismatch: " + frame_id)
        frame = load_match_frame(path)
        if frame.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("training cache manifest hash mismatch: " + frame_id)
        lidar_path = Path(args.lidar_cache) / (frame_id + ".npz")
        if frame.get("training_lidar_cache_sha256") != sha256_file(lidar_path):
            raise RuntimeError("training LiDAR/GT cache hash mismatch: " + frame_id)
        if len(frame["points"]) != len(frame["gt_visible"]):
            raise RuntimeError("training labels do not match correspondence count: " + frame_id)
        if not np.asarray(frame["gt_visible"], dtype=bool).any():
            raise RuntimeError("training frame has no visible pixel targets: " + frame_id)
        if frame.get("train_gt_pose_sha256") is None:
            raise RuntimeError("training cache lacks train-only GT provenance: " + frame_id)
        result.append({"frame_id": frame_id, "frame": frame, "row": row,
                       "cache_path": str(path), "cache_sha256": item["cache_sha256"]})
    return result


def prepare_covariance_inputs(frames, initial, rows_by_frame, device):
    frozen = HighResPatchPoseRefiner().to(device)
    frozen.load_state_dict(initial["model_state_dict"], strict=True)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    prepared = []
    total_visible = 0
    rank_deficient = 0
    for item in frames:
        frame = item["frame"]
        row = item["row"]
        ref_patches, query_patches, candidate_valid = build_patch_batch(
            frame, row, rows_by_frame, device)
        roma_channels = build_ro_ma_channels(frame, device)
        with torch.no_grad():
            delta, _, summary = frozen(ref_patches, query_patches, roma_channels,
                                       candidate_valid, return_features=True)
            original_precision = torch.as_tensor(frame["peak_precisions"],
                                                 dtype=torch.float64, device=device)
            inputs, base_precision = covariance_features(summary, original_precision)
        if inputs.requires_grad or any(value.requires_grad for value in summary.values()):
            raise RuntimeError("frozen pixel head leaked gradients into covariance training")
        peak_pixels = torch.as_tensor(frame["peaks"], dtype=torch.float32, device=device)
        corrected = peak_pixels + delta.detach()
        visible = torch.as_tensor(frame["gt_visible"], dtype=torch.bool, device=device)
        targets = torch.as_tensor(frame["gt_pixels"], dtype=torch.float32, device=device)
        if not torch.isfinite(inputs).all() or not torch.isfinite(corrected).all():
            raise RuntimeError("frozen feature or pixel prediction is non-finite")
        original_values = torch.linalg.eigvalsh(original_precision)
        rank_deficient += int((original_values[:, 0] < PRECISION_FLOOR).sum().item())
        total_visible += int(visible.sum().item())
        geom = frame_geometry_tensors(frame, row, device)
        geom["precision"] = base_precision
        prepared.append({**item, "features": inputs.detach(), "pixels": corrected.detach(),
                         "visible": visible, "targets": targets,
                         "base_precision": base_precision.detach(), "geometry": geom,
                         "original_precision": original_precision.detach(),
                         "pixel_delta": delta.detach()})
        print(json.dumps({"stage": "frozen_pixel_features", "frame": item["frame_id"],
                          "correspondences": int(len(frame["points"])),
                          "visible_train_labels": int(visible.sum().item())}), flush=True)
    if total_visible == 0:
        raise RuntimeError("no visible train pixel labels were materialized")
    return prepared, rank_deficient


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
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    frames = checked_training_frames(args, initial, rows_by_frame, manifest_sha)
    prepared, rank_deficient = prepare_covariance_inputs(frames, initial,
                                                          rows_by_frame, device)
    if len(prepared) != args.expected_train_frames:
        raise RuntimeError("training frame denominator changed before optimization")

    translations, rotations, baseline_nll = [], [], []
    for item in prepared:
        with np.load(Path(args.lidar_cache) / (item["frame_id"] + ".npz")) as data:
            gt_pose = torch.as_tensor(np.asarray(data["GT"], dtype=np.float64),
                                      dtype=torch.float64, device=device)
        baseline = torch.as_tensor(item["frame"]["baseline_pose"],
                                   dtype=torch.float64, device=device)
        translation, rotation = pose_errors(baseline, gt_pose)
        translations.append(float(translation.detach().cpu()))
        rotations.append(float(rotation.detach().cpu()))
        errors = item["pixels"].to(torch.float64) - item["targets"].to(torch.float64)
        p = item["base_precision"]
        visible = item["visible"]
        mahal = torch.einsum("ni,nij,nj->n", errors[visible], p[visible], errors[visible])
        _, logdet = torch.linalg.slogdet(p[visible])
        baseline_nll.extend((.5 * mahal - .5 * logdet).abs().detach().cpu().tolist())
        item["gt_pose"] = gt_pose
    scales = {"translation_m": max(float(np.mean(translations)), 1e-6),
              "rotation_deg": max(float(np.mean(rotations)), 1e-6),
              "pixel_gaussian_nll": max(float(np.mean(baseline_nll)), 1e-6)}

    head = PrecisionCovarianceHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    history = []
    state_path = Path(args.state_path)
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("initial_checkpoint_sha256") != initial_sha or \
                state.get("manifest_sha256") != manifest_sha or \
                state.get("epochs") != args.epochs or state.get("scales") != scales:
            raise RuntimeError("saved covariance training state does not match this run")
        head.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        start_epoch = int(state["completed_epoch"]) + 1
        print(json.dumps({"stage": "resume", "completed_epoch": start_epoch - 1,
                          "next_epoch": start_epoch, "state": str(state_path)}), flush=True)
    else:
        start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(prepared))
        epoch_rows = []
        for position in order:
            item = prepared[int(position)]
            raw = head(item["features"])
            precision, covariance = precision_from_raw(
                raw, item["base_precision"], return_covariance=True)
            nll = precision_nll(item["pixels"], item["targets"],
                                item["visible"], precision)
            predicted_pose, solver = solve_actual_backend(
                item["frame"], item["row"], item["pixels"], item["geometry"],
                device, precision=precision)
            if not np.isfinite(solver["final_objective"]):
                raise RuntimeError("deployed backend returned a non-finite objective for " + item["frame_id"])
            if solver["projected_kkt_inf"] > MAX_PROJECTED_KKT_INF:
                raise RuntimeError("deployed backend KKT residual exceeds the fixed acceptance bound for " +
                                   item["frame_id"] + ": " + str(solver["projected_kkt_inf"]))
            translation, rotation = pose_errors(predicted_pose, item["gt_pose"])
            pose_loss = (translation / scales["translation_m"] +
                         rotation / scales["rotation_deg"])
            nll_scaled = nll / scales["pixel_gaussian_nll"]
            loss = POSE_WEIGHT * pose_loss.to(torch.float32) + \
                NLL_WEIGHT * nll_scaled.to(torch.float32)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite covariance training loss: " + item["frame_id"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            backward = dict(_ImplicitBackendPose.last_backward)
            if not backward or not np.isfinite(backward.get("precision_gradient_norm", np.nan)):
                raise RuntimeError("implicit precision gradient was not produced: " + item["frame_id"])
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite covariance head gradient: " + item["frame_id"])
            optimizer.step()
            translation_value, rotation_value = pose_numpy_error(
                solver["pose"], item["gt_pose"].detach().cpu().numpy())
            eig = torch.linalg.eigvalsh(precision.detach())
            sigma = torch.linalg.eigvalsh(covariance.detach()).clamp_min(1e-20).sqrt()
            epoch_rows.append({
                "frame_id": item["frame_id"],
                "loss_pose_normalized": float(pose_loss.detach().cpu()),
                "loss_gaussian_nll": float(nll.detach().cpu()),
                "loss_nll_normalized": float(nll_scaled.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
                "mpe_m": translation_value,
                "moe_deg": rotation_value,
                "solver_success": solver["success"],
                "solver_status": solver["status"],
                "solver_message": solver["message"],
                "solver_iterations": solver["iterations"],
                "solver_objective_parity_abs": solver["objective_parity_abs"],
                "solver_projected_kkt_inf": solver["projected_kkt_inf"],
                "implicit_precision_gradient_norm": backward["precision_gradient_norm"],
                "head_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "precision_eigenvalue_min": float(eig.min().cpu()),
                "precision_eigenvalue_max": float(eig.max().cpu()),
                "covariance_sigma_median_px": float(sigma.median().cpu()),
            })
        record = {
            "epoch": epoch,
            "frames": len(epoch_rows),
            "mean_pose_loss_normalized": float(np.mean([r["loss_pose_normalized"] for r in epoch_rows])),
            "mean_gaussian_nll": float(np.mean([r["loss_gaussian_nll"] for r in epoch_rows])),
            "mean_nll_normalized": float(np.mean([r["loss_nll_normalized"] for r in epoch_rows])),
            "mean_total_loss": float(np.mean([r["loss_total"] for r in epoch_rows])),
            "mean_train_mpe_m": float(np.mean([r["mpe_m"] for r in epoch_rows])),
            "mean_train_moe_deg": float(np.mean([r["moe_deg"] for r in epoch_rows])),
            "solver_failures": int(sum(not r["solver_success"] for r in epoch_rows)),
            "solver_nonconverged_finite_frames": int(sum(not r["solver_success"] for r in epoch_rows)),
            "solver_max_objective_parity_abs": float(max(r["solver_objective_parity_abs"] for r in epoch_rows)),
            "solver_max_projected_kkt_inf": float(max(r["solver_projected_kkt_inf"] for r in epoch_rows)),
            "solver_max_accepted_projected_kkt_inf": MAX_PROJECTED_KKT_INF,
            "implicit_mean_precision_gradient_norm": float(np.mean(
                [r["implicit_precision_gradient_norm"] for r in epoch_rows])),
            "precision_eigenvalue_min": float(min(r["precision_eigenvalue_min"] for r in epoch_rows)),
            "precision_eigenvalue_max": float(max(r["precision_eigenvalue_max"] for r in epoch_rows)),
            "covariance_sigma_median_px": float(np.median(
                [r["covariance_sigma_median_px"] for r in epoch_rows])),
            "per_frame": epoch_rows,
        }
        history.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "per_frame"}),
              flush=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save({
            "completed_epoch": epoch,
            "epochs": args.epochs,
            "initial_checkpoint_sha256": initial_sha,
            "manifest_sha256": manifest_sha,
            "scales": scales,
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, temporary_state)
        temporary_state.replace(state_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "precision_covariance_head.pt"
    covariance_checkpoint = {
        "model_state_dict": head.state_dict(),
        "method": "frozen_highres_pixel_head_full_covariance_actual_backend_pose_supervision",
        "config": {
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "precision_floor_px_minus2": PRECISION_FLOOR,
            "relative_precision_scale_bound": RELATIVE_PRECISION_SCALE_BOUND,
            "relative_correlation_bound": RELATIVE_CORRELATION_BOUND,
            "pose_weight": POSE_WEIGHT,
            "nll_weight": NLL_WEIGHT,
            "maximum_accepted_projected_kkt_inf": MAX_PROJECTED_KKT_INF,
            "training_loss_scales": scales,
            "train_frames": len(prepared),
            "train_correspondences": int(sum(len(item["frame"]["points"]) for item in prepared)),
            "train_visible_labels": int(sum(int(item["visible"].sum().item()) for item in prepared)),
            "rank_deficient_original_precision_count": rank_deficient,
            "validation_gt_used_for_training": False,
            "checkpoint_selection": "fixed final epoch; no development-set selection",
            "training_state_path": str(state_path),
            "solver_forward": "same deployed bounded SciPy L-BFGS-B LiDAR-camera objective",
            "implicit_backward": "active-set KKT derivative with respect to full precision matrices",
            "frozen_pixel_model": str(initial_path),
            "frozen_pixel_model_sha256": initial_sha,
        },
        "manifest_sha256": manifest_sha,
        "roma_feature_state_sha256": initial["roma_feature_state_sha256"],
        "training_inputs": [
            {"frame_id": item["frame_id"], "cache_sha256": item["cache_sha256"],
             "correspondences": int(len(item["frame"]["points"])),
             "visible_train_labels": int(item["visible"].sum().item())}
            for item in prepared
        ],
        "history": history,
    }
    torch.save(covariance_checkpoint, checkpoint_path)
    report = {
        "protocol": "train-only full 2x2 visual precision learning with frozen high-resolution pixel measurements and deployed pose backend",
        "training_split": "manifest split=train only; development images and labels are not loaded by the trainer",
        "train_frames": len(prepared),
        "train_correspondences": int(sum(len(item["frame"]["points"]) for item in prepared)),
        "visible_train_labels": int(sum(int(item["visible"].sum().item()) for item in prepared)),
        "rank_deficient_original_precision_count": rank_deficient,
        "precision_floor_px_minus2": PRECISION_FLOOR,
        "covariance_parameterization": "full SPD 2x2 covariance represented by a bounded full relative precision matrix around the regularized original precision; no framewise weight normalization",
        "loss": {
            "pose": "final translation norm and SO(3) angle from the deployed bounded LiDAR-camera solver, normalized by train-only mean baseline errors",
            "uncertainty": "visible-point Gaussian NLL 0.5 e^T P e - 0.5 logdet(P), normalized by train-only mean absolute baseline NLL",
            "pose_weight": POSE_WEIGHT,
            "nll_weight": NLL_WEIGHT,
        },
        "frozen_pixel_head": {
            "path": str(initial_path),
            "sha256": initial_sha,
            "all_parameters_frozen": True,
            "pixel_positions_retrained": False,
        },
        "scales": scales,
        "epochs": args.epochs,
        "selection": "fixed final epoch; no development-set checkpoint selection",
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                      "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
        "backend": {
            "forward": "same bounded L-BFGS-B deployed objective, LiDAR information, bounds, and visual precision weighting",
            "backward": "active-set KKT implicit derivative to full precision matrices",
            "finite_nonconverged_policy": "retain and report finite solver results only when projected KKT infinity norm is at most the fixed bound",
        },
        "manifest_sha256": manifest_sha,
        "roma_feature_state_sha256": initial["roma_feature_state_sha256"],
        "training_inputs": covariance_checkpoint["training_inputs"],
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "complete", "checkpoint": str(checkpoint_path),
                      "checkpoint_sha256": report["checkpoint_sha256"],
                      "report": str(report_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--initial-checkpoint", default="research/results/highres_patch_pose_backend_ft_20260923/highres_patch_pose_head.pt")
    parser.add_argument("--train-input-dir", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923/train_inputs")
    parser.add_argument("--output-dir", default="research/results/highres_precision_covariance_20260924")
    parser.add_argument("--state-path", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_precision_covariance_20260924/training_state.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-frames", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
