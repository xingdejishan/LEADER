import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data.leader_nclt_manifest import (
    DEV_DATES,
    EXPECTED_MANIFEST_SHA256,
    TRAIN_DATES,
    SAM_CHECKPOINT_SHA256,
    ManifestLEADERDataset,
    collate_leader_frames,
)
from tools.magic_nclt_runtime import (
    BACKEND,
    OFFICIAL_L0_SHA256,
    OFFICIAL_SOURCE_EXTRA_SHA256,
    OFFICIAL_SOURCE_WEIGHTS_SHA256,
    PoseEstimationFailure,
    canonical_json_sha256,
    identity_failure,
    load_pretrained_model,
    predict_pose,
    safe_json_dump,
    sha256_file,
    source_hashes,
)
from tools.resource_guard import inherited_disk_guard


def _manifest_paths(root):
    return [Path(root) / date / "manifest.json" for date in DEV_DATES]


def _ordered_keys_sha256(keys):
    return canonical_json_sha256(list(keys))


def predict(args):
    guard = inherited_disk_guard()
    if guard is not None:
        guard.require(64 * 1024 * 1024)
    project_root = Path(__file__).resolve().parents[1]
    predictor_path = Path(__file__).resolve()
    evaluator_path = project_root / "tools" / "evaluate_magic_nclt_dev.py"
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    settings = dict(checkpoint.get("settings", {}))
    magic = bool(settings.get("magic", False))
    checkpoint_hash = sha256_file(checkpoint_path)
    runtime_hashes = source_hashes(project_root)
    if checkpoint_path.name == "official_l0.pt" and not magic:
        if settings.get("variant") != "L0_official":
            raise ValueError("LiDAR-only baseline checkpoint is not official L0")
    if not magic and checkpoint_hash != OFFICIAL_L0_SHA256:
        raise ValueError("LiDAR-only prediction requires the frozen official L0 checkpoint")
    if "center_t" not in checkpoint:
        raise ValueError("checkpoint has no frozen center_t")
    center_array = np.asarray(checkpoint["center_t"], dtype=np.float64)
    if center_array.shape != (3,) or not np.isfinite(center_array).all():
        raise ValueError("checkpoint center_t is invalid")
    del checkpoint
    manifest_paths = _manifest_paths(args.manifest_root)
    dataset = ManifestLEADERDataset(
        manifest_paths,
        Path(args.data_root),
        split="dev",
        sam_cache_manifest=Path(args.sam_cache_manifest),
        strict_frozen=True,
        expected_cache_sha256=args.sam_cache_sha256,
        cache_memory_frames=1,
    )
    if len(dataset) != 1200:
        raise ValueError("development prediction requires all 1200 frozen frames")
    for date, expected in EXPECTED_MANIFEST_SHA256.items():
        if date in DEV_DATES and dataset.manifest_hashes.get(date) != expected:
            raise ValueError("development manifest hash differs for {}".format(date))
    if settings.get("magic") is True:
        expected_train = {date: EXPECTED_MANIFEST_SHA256[date] for date in TRAIN_DATES}
        if settings.get("training_manifest_hashes") != expected_train:
            raise ValueError("candidate checkpoint training manifests differ from the frozen dates")
        expected_dev = {date: EXPECTED_MANIFEST_SHA256[date] for date in DEV_DATES}
        if settings.get("dev_manifest_hashes") != expected_dev:
            raise ValueError("candidate checkpoint was trained against different dev manifests")
        if settings.get("dev_sam_cache_sha256") != args.sam_cache_sha256:
            raise ValueError("candidate checkpoint dev SAM cache hash differs")
        if settings.get("source_code_sha256") != runtime_hashes:
            raise ValueError("candidate checkpoint runtime source hashes differ")
        if (settings.get("base_checkpoint_sha256") != OFFICIAL_L0_SHA256
                or settings.get("base_checkpoint_source_weights_sha256") != OFFICIAL_SOURCE_WEIGHTS_SHA256
                or settings.get("base_checkpoint_source_extra_sha256") != OFFICIAL_SOURCE_EXTRA_SHA256
                or not np.array_equal(center_array, np.asarray(
                    settings.get("base_checkpoint_center_t"), dtype=np.float64))):
            raise ValueError("candidate checkpoint is not bound to the frozen official L0 initialization")
        protocol = {key: value for key, value in settings.items()
                    if key not in ("training_protocol_sha256", "magic", "variant", "epoch")}
        protocol_sha = settings.get("training_protocol_sha256")
        if (not isinstance(protocol_sha, str) or len(protocol_sha) != 64
                or canonical_json_sha256(protocol) != protocol_sha
                or int(settings.get("seed", -1)) != int(args.seed)):
            raise ValueError("candidate checkpoint has an invalid protocol hash or prediction seed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("development prediction requires exactly one visible GPU")
    model, loaded_checkpoint = load_pretrained_model(checkpoint_path, magic, device)
    del loaded_checkpoint
    model.eval()
    center = torch.as_tensor(center_array, device=device, dtype=torch.float32)
    output_path = Path(args.predictions).resolve()
    freeze_path = Path(args.freeze).resolve()
    if output_path.exists() or freeze_path.exists():
        raise FileExistsError("prediction outputs already exist; use a new validation attempt directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    started = time.perf_counter()
    rows = []
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        with torch.inference_mode():
            for index in range(len(dataset)):
                if guard is not None:
                    guard.require(64 * 1024 * 1024)
                sample = dataset[index]
                batch = collate_leader_frames([sample])
                torch.manual_seed(int(args.seed) + index)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(args.seed) + index)
                try:
                    pose, metadata = predict_pose(
                        model, batch, center, device, magic, int(args.seed) + index)
                except PoseEstimationFailure as error:
                    pose, metadata = identity_failure(error, error.valid_corr, error.selected_corr)
                row = {
                    "date": sample["date"],
                    "timestamp_us": int(sample["timestamp_us"]),
                    "frame_key": sample["frame_key"],
                    "T_WB_pred": np.asarray(pose, dtype=np.float64).tolist(),
                    "status": metadata["status"],
                    "valid_corr": int(metadata["valid_corr"]),
                    "selected_corr": int(metadata["selected_corr"]),
                    "failure_reason": metadata["failure_reason"],
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
                print("{}/1200 {} {}".format(index + 1, row["status"], row["frame_key"]), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(output_path))
    manifest_hashes = {date: dataset.manifest_hashes[date] for date in DEV_DATES}
    freeze = {
        "schema": "leader_magic_dev_prediction_freeze_v1",
        "role": "development_validation",
        "independent_test": False,
        "ground_truth_was_read_by_predictor": False,
        "frame_count": len(rows),
        "ordered_frame_keys_sha256": _ordered_keys_sha256(dataset.keys),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_variant": settings.get("variant", "trained_candidate"),
        "magic_enabled": magic,
        "fusion_mode": settings.get('fusion_mode', 'official_l0'),
        "image_features_used_by_model": False,
        "predictor_path": str(predictor_path),
        "predictor_sha256": sha256_file(predictor_path),
        "prediction_path": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path) if evaluator_path.exists() else None,
        "runtime_source_hashes": runtime_hashes,
        "training_protocol_sha256": settings.get("training_protocol_sha256"),
        "dev_manifest_hashes": manifest_hashes,
        "sam_cache_manifest_path": str(Path(args.sam_cache_manifest).resolve(strict=True)),
        "sam_cache_manifest_sha256": sha256_file(args.sam_cache_manifest),
        "official_sam_checkpoint_sha256": SAM_CHECKPOINT_SHA256,
        "center_t": center_array.tolist(),
        "backend": BACKEND,
        "seed": int(args.seed),
        "elapsed_seconds": time.perf_counter() - started,
        "frame_status_counts": {
            "success": sum(row["status"] == "success" for row in rows),
            "identity_fallback": sum(row["status"] == "identity_fallback" for row in rows),
        },
    }
    safe_json_dump(freeze_path, freeze)
    if guard is not None:
        guard.require()
    print(json.dumps({
        "prediction_sha256": freeze["prediction_sha256"],
        "freeze_sha256": sha256_file(freeze_path),
        "frame_count": freeze["frame_count"],
        "status_counts": freeze["frame_status_counts"],
        "elapsed_seconds": freeze["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)
    return freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--sam-cache-manifest", type=Path, required=True)
    parser.add_argument("--sam-cache-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
