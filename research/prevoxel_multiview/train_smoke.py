"""Train and validate the six-view visual residual on the fixed 64/32 smoke split."""
import argparse
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
from smoke_dataset import SmokeFrameDataset, prepare_frame


def trr_loss(target, prediction, uncertainty, scale=10.0):
    l_raw = (prediction - target).norm(dim=-1)
    scaler = np.log(scale) / np.pi
    u_cut = uncertainty.clamp(-10 * torch.pi, 10 * torch.pi).atan() * scaler
    u_scale = uncertainty.atan() * scaler
    w_sum = torch.zeros(1, device=prediction.device, dtype=prediction.dtype)
    w_sum += torch.min(u_cut.exp(), u_scale.exp()).sum()
    weights = torch.max(u_cut.exp(), u_scale.exp()) / w_sum.clamp_min(1e-6)
    return (l_raw * weights).mean()


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
    leader_script = os.path.abspath(os.path.join(HERE, cfg["voxel"]["leader_train_script"]))
    if not os.path.exists(leader_script):
        raise FileNotFoundError(leader_script)
    device = torch.device(args.device)
    checkpoint = os.path.abspath(args.checkpoint)
    extra = json.load(open(os.path.join(checkpoint, "extra.json"), encoding="utf-8"))
    center_t = torch.tensor(extra["center_t"], device=device, dtype=torch.float32)
    observer = __import__("dataset_hook", fromlist=["PerFrameObservation"]).PerFrameObservation(cfg, device=str(device))
    train_set = SmokeFrameDataset(args.manifest, "train")
    val_set = SmokeFrameDataset(args.manifest, "val")
    model = PreVoxelMultiViewLEADER(cfg, checkpoint, width=1024).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(2089)
    losses = []
    gradient_norms = None
    model.train()
    for step in range(min(args.steps, len(train_set))):
        sample = prepare_frame(train_set[step], observer, 0.2, 1024)
        coords = ME.utils.batched_coordinates([sample["coords_q"]]).to(device)
        lidar = torch.from_numpy(sample["lidar_feats"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction, diag = model(sample["obs"], lidar, coords,
                                  all_camera_dropout_prob=args.dropout, rng=rng)
        target = scene_targets(diag["encoded"], sample["row"], 0.2, 1024, center_t)
        loss = trr_loss(target, prediction[:, :3], prediction[:, 3])
        loss.backward()
        if step == 0:
            gradient_norms = {
                "view_adapter": float(sum(
                    p.grad.abs().sum() for p in model.fusion.adapter.parameters()
                    if p.grad is not None)),
                "view_weighting": float(sum(
                    p.grad.abs().sum() for p in model.fusion.weighting.parameters()
                    if p.grad is not None)),
                "null_token": float(model.fusion.weighting.null_token.grad.abs().sum()),
                "visual_projection": float(sum(
                    p.grad.abs().sum() for p in model.visual_projection.parameters()
                    if p.grad is not None)),
            }
            if any(value <= 0.0 for value in gradient_norms.values()):
                raise AssertionError("a trainable visual branch received no gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        sample = prepare_frame(val_set[0], observer, 0.2, 1024)
        coords = ME.utils.batched_coordinates([sample["coords_q"]]).to(device)
        lidar = torch.from_numpy(sample["lidar_feats"]).to(device)
        pred_null, null_diag = model(sample["obs"], lidar, coords,
                             force_null=torch.ones(len(sample["scan"]), dtype=torch.bool, device=device))
        if float(null_diag["visual_residual"].abs().max().cpu()) != 0.0:
            raise AssertionError("force-NULL visual residual is nonzero")
        val_pred, val_diag = model(sample["obs"], lidar, coords)
        val_target = scene_targets(val_diag["encoded"], sample["row"], 0.2, 1024, center_t)
        val_loss = trr_loss(val_target, val_pred[:, :3], val_pred[:, 3])

    os.makedirs(args.output, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_losses": losses}, os.path.join(args.output, "smoke_checkpoint.pt"))
    report = {"train_frames": len(train_set), "val_frames": len(val_set),
              "steps": len(losses), "train_loss": losses,
              "gradient_norms_step0": gradient_norms,
              "val_loss": float(val_loss.cpu()), "force_null_baseline_equivalent": True,
              "voxel_size": 0.2, "horizontal_res": 1024,
              "checkpoint": checkpoint, "leader_train_script": leader_script}
    with open(os.path.join(args.output, "smoke_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
