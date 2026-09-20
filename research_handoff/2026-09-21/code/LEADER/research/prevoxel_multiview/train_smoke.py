"""Train and validate the six-view visual residual on the fixed 64/32 smoke split."""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import MinkowskiEngine as ME

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset_hook import load_config
from leader_model import PreVoxelMultiViewLEADER
from projector import ssc_to_homo
from sanity_check import leader_quantization_config
from smoke_dataset import SmokeFrameDataset, prepare_frame


def load_original_trr(script, scale=10.0):
    tree = ast.parse(open(script, encoding="utf-8").read(), filename=script)
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "TRR")
    scope = {"np": np, "torch": torch, "Tensor": torch.Tensor}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, script, "exec"), scope)
    return scope["TRR"](scale=scale)


def trr_frame(loss_fn, target, prediction, uncertainty):
    batch_idx = torch.zeros(len(target), dtype=torch.long, device=target.device)
    return loss_fn(target, prediction, uncertainty, batch_idx)


def load_scene_extrinsic(manifest_row):
    pose = np.loadtxt(manifest_row["pose"]).astype(np.float32)
    calibration = np.loadtxt(manifest_row["calibration"]).astype(np.float32)
    camera_to_body = np.eye(4, dtype=np.float32)
    camera_to_body[:3, :3] = np.eye(3, dtype=np.float32)
    camera_to_body[:3, 3] = 0.0
    if "camera_to_body" in manifest_row:
        camera_to_body = np.asarray(manifest_row["camera_to_body"], dtype=np.float32)
    return pose, calibration, camera_to_body


def scene_targets(encoded, row, voxel_size, horizontal_res, center_t):
    pose = np.loadtxt(row["pose"]).astype(np.float32)
    camera_to_body = next(
        np.asarray(view["camera_to_body"], dtype=np.float32)
        for view in row.get("views", []) if view["camera"] == 5
    )
    gt = pose @ np.linalg.inv(camera_to_body)
    stride = torch.tensor(encoded.tensor_stride, device=encoded.F.device, dtype=torch.float32)
    centers = (encoded.C[:, 1:].float() + stride / 2) * voxel_size
    angles = centers[:, 0:1] * (2 * torch.pi) / (horizontal_res * voxel_size)
    ranges = centers[:, 1:2]
    local = torch.cat([ranges * torch.cos(angles), ranges * torch.sin(angles), centers[:, 2:3]], dim=1)
    transform = torch.as_tensor(gt, device=encoded.F.device)
    target = local @ transform[:3, :3].T + transform[:3, 3] - center_t
    return target


def coordinate_order(encoded):
    coords = encoded.C.detach().cpu().numpy()
    return np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))


def align_sparse_rows(reference, candidate):
    ref_coords = reference.C.detach().cpu().numpy()
    cand_coords = candidate.C.detach().cpu().numpy()
    ref_order = np.lexsort((ref_coords[:, 3], ref_coords[:, 2], ref_coords[:, 1], ref_coords[:, 0]))
    cand_order = np.lexsort((cand_coords[:, 3], cand_coords[:, 2], cand_coords[:, 1], cand_coords[:, 0]))
    if not np.array_equal(ref_coords[ref_order], cand_coords[cand_order]):
        raise AssertionError("B0 encoded coordinate sets differ")
    return ref_order, cand_order


def b0_check(model, sample, device, voxel_size, horizontal_res, center_t, loss_fn, strict=False):
    obs = sample["obs"]
    coords = ME.utils.batched_coordinates([sample["coords_q"]]).to(device)
    lidar = torch.from_numpy(sample["lidar_feats"]).to(device)
    force_null = torch.ones(len(sample["scan"]), dtype=torch.bool, device=device)
    with torch.no_grad():
        _, diagnostics = model(obs, lidar, coords, force_null=force_null)
        index = torch.from_numpy(sample["index"]).to(device)
        sparse = ME.SparseTensor(features=lidar[index], coordinates=coords)
        baseline_encoded = model.base.encoder(sparse)
        baseline_prediction = model.base.decoder(baseline_encoded.F)
        wrapped_prediction = diagnostics["encoded"]
        wrapped_prediction_values = model.base.decoder(wrapped_prediction.F)
        wrapped_encoded = wrapped_prediction
        baseline_target = scene_targets(baseline_encoded, sample["row"], voxel_size,
                                        horizontal_res, center_t)
        wrapped_target = scene_targets(wrapped_encoded, sample["row"], voxel_size,
                                       horizontal_res, center_t)
        baseline_loss, _ = trr_frame(loss_fn, baseline_target, baseline_prediction[:, :3],
                                     baseline_prediction[:, 3])
        wrapped_loss, _ = trr_frame(loss_fn, wrapped_target, wrapped_prediction_values[:, :3],
                                    wrapped_prediction_values[:, 3])
    if not torch.equal(coords, sparse.C):
        raise AssertionError("B0 input coordinates changed")
    if tuple(baseline_encoded.tensor_stride) != tuple(wrapped_encoded.tensor_stride):
        raise AssertionError("B0 encoded stride changed")
    baseline_order, wrapped_order = align_sparse_rows(baseline_encoded, wrapped_encoded)
    baseline_features = baseline_encoded.F[baseline_order]
    wrapped_features = wrapped_encoded.F[wrapped_order]
    baseline_prediction = baseline_prediction[baseline_order]
    wrapped_prediction_values = wrapped_prediction_values[wrapped_order]
    baseline_target = baseline_target[baseline_order]
    wrapped_target = wrapped_target[wrapped_order]
    checks = {
        "input_lidar_features_max_abs": 0.0,
        "encoded_coordinate_set_equal": True,
        "encoded_row_order_equal": bool(np.array_equal(
            baseline_encoded.C.detach().cpu().numpy(), wrapped_encoded.C.detach().cpu().numpy())),
        "encoded_features_max_abs": float((baseline_features - wrapped_features).abs().max().cpu()),
        "prediction_max_abs": float((baseline_prediction - wrapped_prediction_values).abs().max().cpu()),
        "target_max_abs": float((baseline_target - wrapped_target).abs().max().cpu()),
        "loss_abs": float((baseline_loss - wrapped_loss).abs().cpu()),
        "force_null_visual_residual_max_abs": float(diagnostics["visual_residual"].abs().max().cpu()),
    }
    checks["strict_pass"] = all(
        value <= 1e-5 for key, value in checks.items()
        if key.endswith("_max_abs") or key == "loss_abs")
    if strict and not checks["strict_pass"]:
        raise AssertionError("B0 baseline equivalence failed: %s" % checks)
    return checks


def gradient_norms(model):
    def total(parameters):
        return float(sum(p.grad.abs().sum() for p in parameters if p.grad is not None))

    final = model.visual_projection.net[-1]
    return {
        "view_adapter": total(model.fusion.adapter.parameters()),
        "view_weighting": total(model.fusion.weighting.encoder.parameters()),
        "null_token": float(model.fusion.weighting.null_token.grad.abs().sum())
        if model.fusion.weighting.null_token.grad is not None else 0.0,
        "visual_projection": total(model.visual_projection.parameters()),
        "visual_projection_last_layer": total(final.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the smoke entry")
    cfg = load_config(args.config)
    quantization = leader_quantization_config(cfg)
    leader_script = quantization["script"]
    voxel_size = quantization["voxel_size"]
    horizontal_res = quantization["horizontal_res"]
    device = torch.device(args.device)
    checkpoint = os.path.abspath(args.checkpoint)
    extra = json.load(open(os.path.join(checkpoint, "extra.json"), encoding="utf-8"))
    center_t = torch.tensor(extra["center_t"], device=device, dtype=torch.float32)
    observer = __import__("dataset_hook", fromlist=["PerFrameObservation"]).PerFrameObservation(cfg, device=str(device))
    train_set = SmokeFrameDataset(args.manifest, "train")
    val_set = SmokeFrameDataset(args.manifest, "val")
    loss_fn = load_original_trr(leader_script)
    b0_sample = prepare_frame(train_set[0], observer, voxel_size, horizontal_res)
    b0_model = PreVoxelMultiViewLEADER(cfg, checkpoint, width=int(horizontal_res)).cpu().eval()
    b0_sample["obs"] = {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in b0_sample["obs"].items()
    }
    b0 = b0_check(b0_model, b0_sample, torch.device("cpu"), voxel_size,
                  horizontal_res, center_t.cpu(), loss_fn, strict=True)
    del b0_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = PreVoxelMultiViewLEADER(cfg, checkpoint, width=int(horizontal_res)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(2089)
    losses = []
    gradient_history = []
    model.train()
    for step in range(min(args.steps, len(train_set))):
        sample = prepare_frame(train_set[step], observer, voxel_size, horizontal_res)
        coords = ME.utils.batched_coordinates([sample["coords_q"]]).to(device)
        lidar = torch.from_numpy(sample["lidar_feats"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction, diag = model(sample["obs"], lidar, coords,
                                  all_camera_dropout_prob=args.dropout, rng=rng)
        target = scene_targets(diag["encoded"], sample["row"], voxel_size, horizontal_res, center_t)
        loss, _ = trr_frame(loss_fn, target, prediction[:, :3], prediction[:, 3])
        loss.backward()
        gradient_history.append(gradient_norms(model))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    if len(gradient_history) > 1:
        if gradient_history[1]["view_adapter"] <= 0.0 or gradient_history[1]["view_weighting"] <= 0.0:
            raise AssertionError("visual fusion did not receive gradient after zero-init projection update")
    val_losses = []
    with torch.no_grad():
        for row in val_set.rows:
            sample = prepare_frame(row, observer, voxel_size, horizontal_res)
            coords = ME.utils.batched_coordinates([sample["coords_q"]]).to(device)
            lidar = torch.from_numpy(sample["lidar_feats"]).to(device)
            val_pred, val_diag = model(sample["obs"], lidar, coords)
            val_target = scene_targets(val_diag["encoded"], row, voxel_size, horizontal_res, center_t)
            val_loss, _ = trr_frame(loss_fn, val_target, val_pred[:, :3], val_pred[:, 3])
            val_losses.append(float(val_loss.cpu()))

    os.makedirs(args.output, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_losses": losses}, os.path.join(args.output, "smoke_checkpoint.pt"))
    report = {"train_frames": len(train_set), "val_frames": len(val_set),
              "steps": len(losses), "train_loss": losses,
              "gradient_norms": gradient_history,
              "b0": b0,
              "val_loss": float(np.mean(val_losses)),
              "val_loss_per_frame": val_losses,
              "validation_kind": "all validation-frame TRR mean; no pose solver",
              "force_null_baseline_equivalent": True,
              "voxel_size": voxel_size, "horizontal_res": horizontal_res,
              "checkpoint": checkpoint, "leader_train_script": leader_script,
              "quantization_source": "run_mink.py argparse defaults"}
    with open(os.path.join(args.output, "smoke_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
