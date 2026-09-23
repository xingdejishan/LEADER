import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nre_scoremap_pose_runner as nre
from highres_patch_pose_refiner import (
    HighResPatchPoseRefiner,
    POSE_STEPS,
    POSE_WEIGHT,
    QUERY_PATCH_SIZE,
    REFERENCE_PATCH_SIZE,
    SEARCH_RADIUS,
    SEED,
    TRAIN_EPOCHS,
    add_training_targets,
    build_patch_batch,
    build_ro_ma_channels,
    differentiable_pose_solve,
    frame_geometry_tensors,
    load_backend,
    load_match_frame,
    prepare_match_frame,
    save_match_frame,
    sha256_file,
)


def training_cache_valid(path, manifest_sha, feature_sha, match_sha, lidar_sha):
    if not Path(path).is_file():
        return False
    try:
        frame = load_match_frame(path)
    except Exception:
        return False
    return (frame.get("manifest_sha256") == manifest_sha and
            frame.get("roma_feature_state_sha256") == feature_sha and
            frame.get("match_cache_sha256") == match_sha and
            frame.get("training_lidar_cache_sha256") == lidar_sha and
            "gt_pixels" in frame and "gt_visible" in frame)


def train(args):
    torch.set_num_threads(16)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    cache_dir = Path(args.train_match_cache)
    train_rows = [row for row in rows if row["split"] == "train" and
                  (cache_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if args.frames:
        train_rows = train_rows[:args.frames]
    if len(train_rows) != args.expected_train_frames:
        raise RuntimeError(f"expected {args.expected_train_frames} cached train frames, got {len(train_rows)}")
    if any(row["split"] != "train" for row in train_rows):
        raise RuntimeError("non-train row entered training")
    manifest_sha = sha256_file(args.manifest)
    lidar_root = Path(args.lidar_cache)
    train_dir = Path(args.output_dir) / "train_inputs"
    train_dir.mkdir(parents=True, exist_ok=True)

    roma_features = nre.RoMaFeatures(args.device, "precise", 4)
    feature_sha = roma_features.model_sha256()
    matcher, full_pool_refine = load_backend(args.baseline_code_root, args.device)
    input_rows = []
    for row_index, row in enumerate(train_rows):
        frame_id = str(row["frame_id"])
        match_path = cache_dir / (frame_id + ".npz")
        lidar_path = lidar_root / (frame_id + ".npz")
        cache_path = train_dir / (frame_id + ".npz")
        match_sha = sha256_file(match_path)
        lidar_sha = sha256_file(lidar_path)
        if not training_cache_valid(cache_path, manifest_sha, feature_sha, match_sha, lidar_sha):
            frame = prepare_match_frame(row, rows_by_frame, match_path, lidar_root, matcher,
                                        full_pool_refine, roma_features, args.device, SEED + row_index)
            frame = add_training_targets(frame, row, lidar_root)
            frame["training_lidar_cache_sha256"] = lidar_sha
            frame["manifest_sha256"] = manifest_sha
            frame["roma_feature_state_sha256"] = feature_sha
            save_match_frame(frame, cache_path)
        frame = load_match_frame(cache_path)
        if frame.get("manifest_sha256") != manifest_sha or frame.get("roma_feature_state_sha256") != feature_sha:
            raise RuntimeError("training materialization provenance mismatch: " + frame_id)
        if len(frame["points"]) != len(frame["gt_visible"]):
            raise RuntimeError("training label count mismatch: " + frame_id)
        input_rows.append({
            "frame_id": frame_id,
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "match_cache_sha256": match_sha,
            "lidar_cache_sha256": lidar_sha,
            "matched_points": int(frame["match_cache_rows"]),
            "usable_local_peaks": int(frame["usable_local_peaks"]),
            "visible_train_labels": int(np.asarray(frame["gt_visible"], dtype=bool).sum()),
        })
        print(json.dumps({"stage": "materialize", "frame": frame_id,
                          "usable_local_peaks": int(frame["usable_local_peaks"]),
                          "visible_train_labels": int(np.asarray(frame["gt_visible"], dtype=bool).sum())}),
              flush=True)

    model = HighResPatchPoseRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(input_rows))
        epoch_pixel, epoch_pose, epoch_total, used_frames = [], [], [], 0
        for position in order:
            item = input_rows[int(position)]
            frame = load_match_frame(item["cache_path"])
            row = rows_by_frame[item["frame_id"]]
            if row["split"] != "train":
                raise RuntimeError("validation row reached the optimizer")
            ref_patches, query_patches, candidate_valid = build_patch_batch(frame, row, rows_by_frame, device)
            roma_channels = build_ro_ma_channels(frame, device)
            geometry = frame_geometry_tensors(frame, row, device)
            gt_pixels = torch.as_tensor(frame["gt_pixels"], dtype=torch.float32, device=device)
            gt_visible = torch.as_tensor(frame["gt_visible"], dtype=torch.bool, device=device)
            if not gt_visible.any():
                raise RuntimeError("training frame has no visible pixel labels: " + item["frame_id"])

            delta, _ = model(ref_patches, query_patches, roma_channels, candidate_valid)
            peak_pixels = torch.as_tensor(frame["peaks"], dtype=torch.float32, device=device)
            corrected_pixels = peak_pixels + delta
            direct_error = F.smooth_l1_loss(corrected_pixels[gt_visible], gt_pixels[gt_visible],
                                            beta=1., reduction="none").sum(dim=1)
            pixel_loss = direct_error.mean()
            predicted_pose, pose_delta = differentiable_pose_solve(
                geometry["points"], corrected_pixels, geometry["precision"],
                geometry["camera_to_body"], geometry["calibration"],
                geometry["baseline_pose"], geometry["lidar_information"], POSE_STEPS)
            projected_pose_pixels, _ = nre_project_from_delta(geometry, pose_delta)
            pose_error = F.smooth_l1_loss(projected_pose_pixels[gt_visible],
                                          gt_pixels[gt_visible].to(torch.float64),
                                          beta=1., reduction="none").sum(dim=1)
            pose_loss = pose_error.mean().to(torch.float32)
            loss = pixel_loss + POSE_WEIGHT * pose_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss: " + item["frame_id"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_pixel.append(float(pixel_loss.detach().cpu()))
            epoch_pose.append(float(pose_loss.detach().cpu()))
            epoch_total.append(float(loss.detach().cpu()))
            used_frames += 1
        record = {"epoch": epoch, "frames": used_frames,
                  "mean_pixel_loss_px": float(np.mean(epoch_pixel)),
                  "mean_pose_reprojection_loss_px": float(np.mean(epoch_pose)),
                  "mean_total_loss": float(np.mean(epoch_total))}
        history.append(record)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {"epochs": args.epochs, "optimizer": "AdamW", "learning_rate": 1e-3,
                       "weight_decay": 1e-4, "gradient_clip": 1., "seed": SEED,
                       "search_radius_px": SEARCH_RADIUS, "pose_steps": POSE_STEPS,
                       "pose_loss_weight": POSE_WEIGHT, "train_frames": len(train_rows),
                       "train_correspondences": int(sum(row["usable_local_peaks"] for row in input_rows)),
                       "validation_gt_used_for_training": False,
                       "checkpoint_selection": "fixed final epoch; no validation-based selection"},
            "manifest_sha256": manifest_sha,
            "roma_feature_state_sha256": feature_sha,
            "training_inputs": input_rows,
            "history": history,
        }
        checkpoint_path = Path(args.output_dir) / "highres_patch_pose_head.pt"
        torch.save(checkpoint, checkpoint_path)
        print(json.dumps({"stage": "train", **record,
                          "checkpoint_sha256": sha256_file(checkpoint_path)}), flush=True)

    report = {
        "protocol": "train-only raw RGB patch matching head with unrolled shared-pose reprojection supervision",
        "training_split": "manifest split=train only; validation frames and GT are never loaded by this script",
        "train_frames": len(train_rows),
        "train_correspondences": int(sum(row["usable_local_peaks"] for row in input_rows)),
        "visible_pixel_labels": int(sum(row["visible_train_labels"] for row in input_rows)),
        "epochs": args.epochs,
        "selection": "fixed final epoch; no validation-based early stopping or checkpoint selection",
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3,
                      "weight_decay": 1e-4, "gradient_clip": 1.},
        "head": {"shared_rgb_patch_encoder": True, "reference_patch_px": REFERENCE_PATCH_SIZE,
                 "query_search_patch_px": QUERY_PATCH_SIZE, "offset_radius_px": SEARCH_RADIUS,
                 "search_grid": 2 * SEARCH_RADIUS + 1, "confidence_head": False},
        "loss": {"pixel": "mean per-frame smooth L1, beta=1 px",
                 "pose": "mean per-frame smooth L1 of GT reprojection after five differentiable Gauss-Newton updates",
                 "pose_weight": POSE_WEIGHT},
        "baseline": "SC2-PCR plus two-stage full-pool refinement at 1.2 m and 0.6 m",
        "manifest_sha256": manifest_sha,
        "roma_feature_state_sha256": feature_sha,
        "training_inputs": input_rows,
        "history": history,
        "checkpoint_path": str(Path(args.output_dir) / "highres_patch_pose_head.pt"),
        "checkpoint_sha256": sha256_file(Path(args.output_dir) / "highres_patch_pose_head.pt"),
    }
    report_path = Path(args.output_dir) / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "complete", "report": str(report_path),
                      "checkpoint_sha256": report["checkpoint_sha256"]}), flush=True)


def nre_project_from_delta(geometry, delta):
    import highres_patch_pose_refiner as refiner

    projected, jacobian = refiner.project_and_jacobian(
        geometry["points"], delta, geometry["baseline_pose"],
        geometry["camera_to_body"], geometry["calibration"])
    return projected, jacobian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    parser.add_argument("--train-match-cache", default=("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/"
                                                          "research/prevoxel_multiview/results/roma_train_precise_top2_cache_refuv"))
    parser.add_argument("--baseline-code-root", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/glace-local/code")
    parser.add_argument("--output-dir", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-frames", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
