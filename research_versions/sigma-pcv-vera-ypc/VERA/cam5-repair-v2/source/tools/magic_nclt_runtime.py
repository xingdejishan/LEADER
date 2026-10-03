import hashlib
import json
import math
import os
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.leader_nclt_manifest import DEFAULT_HORIZONTAL_RESOLUTION, DEFAULT_VOXEL_SIZE
from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER
from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine


OFFICIAL_L0_SHA256 = "49cf3085b507e2f67a0f250394cadd980b0f5da18eb00dcfe37a6c766d3f4d7a"
OFFICIAL_SOURCE_WEIGHTS_SHA256 = "a78d5114bb908717676f7851399bdbb78943fd9e0170b1dd8eb592e0558fa028"
OFFICIAL_SOURCE_EXTRA_SHA256 = "5c217c84692a4da5640839e4849670e2931afa180659010c29365fe0133eafa6"
BACKEND = {
    "name": "SC2-PCR+full_pool_refine",
    "matcher": {
        "inlier_threshold": 2.0,
        "d_thre": 2,
        "num_iterations": 10,
        "ratio": 0.15,
        "nms_radius": 0.1,
        "max_points": 3000,
        "k1": 30,
    },
    "full_pool_thresholds_m": [1.2, 0.6],
    "selected_correspondences": "max(min(50,N),floor(0.5*N))",
}


class PoseEstimationFailure(RuntimeError):
    def __init__(self, reason, valid_corr, selected_corr):
        super().__init__(reason)
        self.valid_corr = int(valid_corr)
        self.selected_corr = int(selected_corr)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def canonical_json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(raw)


def source_hashes(project_root):
    root = Path(project_root)
    names = (
        "models/model_mink.py",
        "models/magic_paper_fusion.py",
        "models/magic_fusion.py",
        "models/sc2pcr.py",
        "utils/full_pool_robust_v1.py",
        "utils/pose_util.py",
        "data/leader_nclt_manifest.py",
        "run_mink.py",
        "tools/magic_nclt_runtime.py",
        "tools/train_magic_nclt.py",
        "tools/predict_magic_nclt.py",
        "tools/evaluate_magic_nclt_dev.py",
        "tools/sam_cache_frozen.py",
        "tools/prepare_magic_dev_sam.py",
        "tools/resource_guard.py",
        "tools/guarded_run.py",
        "tools/smoke_magic_nclt.py",
    )
    return {name: sha256_file(root / name) for name in names}


def verify_official_l0(checkpoint_path, checkpoint):
    observed = sha256_file(checkpoint_path)
    if observed != OFFICIAL_L0_SHA256:
        raise ValueError("official_l0.pt SHA-256 differs from the frozen checkpoint")
    settings = checkpoint.get("settings", {})
    if settings.get("variant") != "L0_official" or settings.get("magic") is not False:
        raise ValueError("checkpoint is not the frozen official LiDAR-only LEADER L0")
    if settings.get("source_checkpoint_sha256") != OFFICIAL_SOURCE_WEIGHTS_SHA256:
        raise ValueError("official safetensors source hash is not the frozen value")
    if settings.get("source_extra_sha256") != OFFICIAL_SOURCE_EXTRA_SHA256:
        raise ValueError("official extra.json hash is not the frozen value")
    center = np.asarray(checkpoint.get("center_t"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("official checkpoint has an invalid embedded center_t")
    return observed, center


def trr_loss(targets, predictions, reliability, batch_index, scale=10.0):
    if targets.ndim != 2 or targets.shape[1] != 3 or predictions.shape != targets.shape:
        raise ValueError("TRR point targets and predictions must be [N,3]")
    if reliability.reshape(-1).shape[0] != targets.shape[0] or batch_index.reshape(-1).shape[0] != targets.shape[0]:
        raise ValueError("TRR reliability and batch-index rows must align with points")
    if targets.shape[0] == 0:
        raise ValueError("TRR requires at least one sparse output point")
    batch_index = batch_index.reshape(-1).long()
    batch_size = int(batch_index.max().item()) + 1
    raw = torch.linalg.vector_norm(predictions - targets, dim=-1)
    factor = math.log(float(scale)) / math.pi
    clipped = reliability.reshape(-1).clamp(min=-10 * math.pi, max=10 * math.pi).atan() * factor
    full = reliability.reshape(-1).atan() * factor
    low = clipped.exp()
    high = full.exp()
    denominator = raw.new_zeros((batch_size,))
    denominator.scatter_add_(0, batch_index, torch.minimum(low, high))
    weights = torch.maximum(low, high) / denominator[batch_index].clamp_min(1e-6)
    per_batch = raw.new_zeros((batch_size,))
    per_batch.scatter_add_(0, batch_index, raw * weights)
    return per_batch.mean(), raw.mean()


def model_parameter_groups(model, lidar_lr=1e-5, fusion_lr=1e-4, decoder_lr=1e-4):
    modules = (
        ("rpge", model.encoder, float(lidar_lr)),
        ("fusion", model.magic_fusion, float(fusion_lr)),
        ("mmregressor", model.decoder, float(decoder_lr)),
    )
    groups = []
    seen = set()
    for name, module, lr in modules:
        if module is None:
            raise ValueError("MaGiC fine-tuning requires all three optimizer modules")
        parameters = []
        for parameter in module.parameters():
            if id(parameter) in seen:
                raise ValueError("optimizer module parameter overlap detected")
            seen.add(id(parameter))
            parameter.requires_grad_(True)
            parameters.append(parameter)
        groups.append({"params": parameters, "lr": lr, "name": name})
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if expected != seen:
        raise ValueError("optimizer groups do not cover the model trainable parameters exactly")
    return groups


def configure_training_mode(model):
    model.train()
    for component in (model.encoder, model.decoder):
        for module in component.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, ME.MinkowskiBatchNorm)):
                module.eval()
    return model


def load_pretrained_model(checkpoint_path, magic, device):
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint_path.name == "official_l0.pt":
        verify_official_l0(checkpoint_path, checkpoint)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512,
                   width=DEFAULT_HORIZONTAL_RESOLUTION, magic=bool(magic))
    if magic:
        if checkpoint.get("settings", {}).get("magic") is True:
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            if unexpected or any(not key.startswith("magic_fusion.") for key in missing):
                raise ValueError("official L0 initialization mismatch: missing={}, unexpected={}".format(
                    missing, unexpected))
            expected_missing = {
                key for key in model.state_dict()
                if key.startswith("magic_fusion.") and not key.endswith(".num_batches_tracked")
            }
            if set(missing) != expected_missing:
                raise ValueError("MaGiC initialization is missing unexpected fusion parameters")
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    return model, checkpoint


def batch_forward(model, batch, center_t, device, magic):
    features = batch["feats"].to(device=device, dtype=torch.float32, non_blocking=True)
    coordinates = batch["coords"].to(device=device, dtype=torch.int32, non_blocking=True)
    sparse = ME.SparseTensor(features=features, coordinates=coordinates)
    if magic:
        encoded, stages = model.encoder(sparse, return_stages=True)
    else:
        encoded = model.encoder(sparse)
        stages = None
    stride = torch.as_tensor(encoded.tensor_stride, device=device, dtype=torch.float32)
    points = polar_voxel_centers(encoded.C, stride, DEFAULT_VOXEL_SIZE,
                                 DEFAULT_HORIZONTAL_RESOLUTION)
    batch_index = encoded.C[:, 0].long()
    if "T" not in batch:
        raise ValueError("training batch does not contain T_WB labels")
    poses = batch["T"].to(device=device, dtype=torch.float32, non_blocking=True)
    corrections = batch["T_corr"].to(device=device, dtype=torch.float32, non_blocking=True)
    corrected_pose = poses @ torch.linalg.inv(corrections)
    selected_pose = corrected_pose[batch_index]
    targets = torch.bmm(selected_pose[:, :3, :3], points.unsqueeze(-1)).squeeze(-1)
    targets = targets + selected_pose[:, :3, 3] - center_t
    output_features = encoded.F
    if magic:
        sam = batch["sam_features"].to(device=device, dtype=encoded.F.dtype, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device=device, dtype=encoded.F.dtype, non_blocking=True)
        camera_from_lidar = batch["camera_from_lidar"].to(device=device, dtype=encoded.F.dtype,
                                                          non_blocking=True)
        bounds = batch["image_bounds"].to(device=device, dtype=encoded.F.dtype, non_blocking=True)
        valid_mask = batch["image_valid_mask"].to(device=device, non_blocking=True)
        recovery = torch.linalg.inv(corrections.to(device=device, dtype=encoded.F.dtype))
        output_features = model.magic_fusion(
            encoded.F, points, encoded.C, stride, sam, intrinsics,
            camera_from_lidar, recovery, bounds, stages=stages,
            voxel_size=DEFAULT_VOXEL_SIZE,
            horizontal=DEFAULT_HORIZONTAL_RESOLUTION,
            image_valid_mask=valid_mask,
        )
    predictions = model.decoder(output_features)
    loss, raw = trr_loss(targets, predictions[:, :3], predictions[:, 3], batch_index)
    return loss, raw, {
        "points": points,
        "predictions": predictions,
        "batch_index": batch_index,
        "encoded_coordinates": encoded.C,
        "stages": stages,
    }


def make_matcher(device):
    config = BACKEND["matcher"]
    return Matcher(
        inlier_threshold=config["inlier_threshold"],
        d_thre=config["d_thre"],
        num_iterations=config["num_iterations"],
        ratio=config["ratio"],
        nms_radius=config["nms_radius"],
        max_points=config["max_points"],
        k1=config["k1"],
    )


def predict_pose(model, batch, center_t, device, magic, seed):
    sparse = ME.SparseTensor(
        features=batch["feats"].to(device=device, dtype=torch.float32),
        coordinates=batch["coords"].to(device=device, dtype=torch.int32),
    )
    if magic:
        encoded, stages = model.encoder(sparse, return_stages=True)
    else:
        encoded = model.encoder(sparse)
        stages = None
    stride = torch.as_tensor(encoded.tensor_stride, device=device, dtype=torch.float32)
    points = polar_voxel_centers(encoded.C, stride, DEFAULT_VOXEL_SIZE,
                                 DEFAULT_HORIZONTAL_RESOLUTION)
    features = encoded.F
    if magic:
        recovery = torch.linalg.inv(batch["T_corr"].to(device=device, dtype=encoded.F.dtype))
        features = model.magic_fusion(
            encoded.F,
            points,
            encoded.C,
            stride,
            batch["sam_features"].to(device=device, dtype=encoded.F.dtype),
            batch["intrinsics"].to(device=device, dtype=encoded.F.dtype),
            batch["camera_from_lidar"].to(device=device, dtype=encoded.F.dtype),
            recovery,
            batch["image_bounds"].to(device=device, dtype=encoded.F.dtype),
            stages=stages,
            voxel_size=DEFAULT_VOXEL_SIZE,
            horizontal=DEFAULT_HORIZONTAL_RESOLUTION,
            image_valid_mask=batch["image_valid_mask"].to(device=device),
        )
    output = model.decoder(features)
    count = max(min(50, len(output)), int(0.5 * len(output)))
    row = {
        "valid_corr": int(len(output)),
        "selected_corr": int(count),
        "status": "success",
        "failure_reason": None,
    }
    if count < 3:
        raise PoseEstimationFailure("fewer than three LEADER correspondences", len(output), count)
    indices = output[:, 3].topk(count).indices
    source = points[indices].float()
    target = output[indices, :3].float()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    matcher = make_matcher(device)
    initial = matcher.estimator(source[None], target[None])[0]
    refined = full_pool_refine(initial, source, target)
    refined[:3, 3] += center_t
    correction = batch["T_corr"].to(device=device, dtype=refined.dtype)[0]
    refined = refined @ correction
    if not torch.isfinite(refined).all():
        raise PoseEstimationFailure("pose backend returned a nonfinite transform", len(output), count)
    return refined.detach().cpu().double().numpy(), row


def identity_failure(error, valid_corr=0, selected_corr=0):
    return np.eye(4, dtype=np.float64), {
        "valid_corr": int(valid_corr),
        "selected_corr": int(selected_corr),
        "status": "identity_fallback",
        "failure_reason": "{}: {}".format(type(error).__name__, str(error)),
    }


def safe_json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(str(temporary), str(path))
