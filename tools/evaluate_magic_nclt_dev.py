import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import numpy as np

from data.leader_nclt_manifest import DEV_DATES, EXPECTED_MANIFEST_SHA256
from tools.magic_nclt_runtime import (
    BACKEND,
    OFFICIAL_L0_SHA256,
    canonical_json_sha256,
    safe_json_dump,
    sha256_file,
    source_hashes,
)
from tools.resource_guard import inherited_disk_guard


EXPECTED_PER_DATE = 300
EXPECTED_TOTAL = 1200


def _safe_path(root, date, value):
    text = str(value)
    relative = PurePosixPath(text)
    if ("\\" in text or ":" in text or "\x00" in text or relative.is_absolute()
            or not relative.parts or any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError("GT manifest path is not a clean relative path")
    base = Path(root).resolve(strict=True)
    date_root = (base / date).resolve(strict=True)
    path = date_root.joinpath(*relative.parts).resolve(strict=True)
    if base not in date_root.parents or date_root not in path.parents:
        raise ValueError("GT path escapes its frozen date directory")
    return path


def _load_public_frames(manifest_root):
    root = Path(manifest_root).resolve(strict=True)
    frames = []
    hashes = {}
    t_bc_by_date = {}
    for date in DEV_DATES:
        path = root / date / "manifest.json"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != EXPECTED_MANIFEST_SHA256[date]:
            raise ValueError("frozen public dev manifest SHA mismatch for {}".format(date))
        document = json.loads(raw.decode("utf-8"))
        if (document.get("date") != date or document.get("split") != "test"
                or document.get("complete") is not True or document.get("frame_count") != EXPECTED_PER_DATE):
            raise ValueError("public dev manifest metadata mismatch for {}".format(date))
        rows = document.get("frames")
        if not isinstance(rows, list) or len(rows) != EXPECTED_PER_DATE:
            raise ValueError("public dev manifest must contain exactly 300 frames for {}".format(date))
        t_bc = np.asarray(document["roi"]["T_BC"], dtype=np.float64)
        if t_bc.shape != (4, 4) or not np.isfinite(t_bc).all():
            raise ValueError("public dev manifest T_BC is invalid for {}".format(date))
        t_bc_by_date[date] = t_bc
        timestamps = []
        for row in rows:
            timestamp = int(row["timestamp_us"])
            if (int(row.get("camera_id", -1)) != 5
                    or int(row["scan"]["timestamp_us"]) != timestamp):
                raise ValueError("public dev manifest has an unsynchronized frame")
            timestamps.append(timestamp)
            frames.append({"date": date, "timestamp_us": timestamp,
                           "frame_key": "{}/{}".format(date, timestamp)})
        if len(set(timestamps)) != EXPECTED_PER_DATE:
            raise ValueError("public dev manifest contains duplicate timestamps for {}".format(date))
        hashes[date] = digest
    return frames, hashes, t_bc_by_date


def _read_gt_date(gt_root, date, expected_frames, t_bc):
    date_root = Path(gt_root).resolve(strict=True) / date
    manifest_path = date_root / "manifest.json"
    sidecar_path = date_root / "manifest.sha256"
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = sidecar_path.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0] != digest:
        raise ValueError("evaluator-only GT sidecar mismatch for {}".format(date))
    document = json.loads(raw.decode("utf-8"))
    rows = document.get("frames")
    if (document.get("date") != date or document.get("split") != "test_evaluator_only"
            or document.get("frame_count") != EXPECTED_PER_DATE
            or not isinstance(rows, list) or len(rows) != EXPECTED_PER_DATE):
        raise ValueError("evaluator-only GT manifest metadata mismatch for {}".format(date))
    expected_timestamps = [frame["timestamp_us"] for frame in expected_frames]
    poses = []
    for index, (row, timestamp) in enumerate(zip(rows, expected_timestamps)):
        if row.get("index") != index or row.get("timestamp_us") != timestamp:
            raise ValueError("evaluator-only GT ordering/key mismatch for {}".format(date))
        gt = row.get("ground_truth")
        expected_relative = "groundtruth_interpolated/{}_T_WC.txt".format(timestamp)
        if (not isinstance(gt, dict) or gt.get("pose") != "T_WC"
                or gt.get("path") != expected_relative
                or not re.fullmatch(r"[0-9a-f]{64}", str(gt.get("sha256", "")))):
            raise ValueError("evaluator-only GT record is invalid for {}".format(date))
        path = _safe_path(Path(gt_root), date, gt["path"])
        pose_bytes = path.read_bytes()
        if hashlib.sha256(pose_bytes).hexdigest() != gt["sha256"]:
            raise ValueError("evaluator-only GT pose SHA mismatch: {}".format(path))
        T_WC = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
        if (not np.isfinite(T_WC).all() or not np.allclose(T_WC[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(T_WC[:3, :3].T @ T_WC[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(T_WC[:3, :3]), 1.0, rtol=0, atol=1e-5)):
            raise ValueError("evaluator-only GT pose is not a rigid transform")
        poses.append(T_WC @ np.linalg.inv(t_bc))
    return poses, digest, hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def _pose_errors(predicted, truth):
    transform = np.asarray(predicted, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("prediction pose must be finite [4,4]")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-2)):
        raise ValueError("prediction pose is not a rigid transform")
    translation_error = float(np.linalg.norm(transform[:3, 3] - truth[:3, 3]))
    cosine = (np.trace(rotation.T @ truth[:3, :3]) - 1.0) / 2.0
    rotation_error = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return translation_error, rotation_error


def _summary(rows):
    translation = np.asarray([row["mpe_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row["moe_deg"] for row in rows], dtype=np.float64)
    return {
        "frames": int(len(rows)),
        "success_frames": int(sum(row["status"] == "success" for row in rows)),
        "failure_frames": int(sum(row["status"] != "success" for row in rows)),
        "mpe_mean_m": float(translation.mean()),
        "mpe_p90_m": float(np.percentile(translation, 90)),
        "moe_mean_deg": float(rotation.mean()),
        "moe_p90_deg": float(np.percentile(rotation, 90)),
    }


def score_predictions(args):
    guard = inherited_disk_guard()
    if guard is not None:
        guard.require(64 * 1024 * 1024)
    predictions_path = Path(args.predictions).resolve(strict=True)
    freeze_path = Path(args.freeze).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    predictor_path = Path(args.predictor).resolve(strict=True)
    evaluator_path = Path(__file__).resolve()
    project_root = predictor_path.parents[1]
    observed_source_hashes = source_hashes(project_root)
    report_path = Path(args.report).resolve()
    gt_root = Path(args.dev_gt_root).resolve(strict=True)
    if gt_root.name != "test_gt" or gt_root.parent.name != "evaluator_only":
        raise ValueError("dev GT root must remain under evaluator_only/test_gt")
    if gt_root == report_path or gt_root in report_path.parents:
        raise ValueError("development report cannot be written inside the GT-only tree")
    if report_path.exists():
        raise FileExistsError("refusing to overwrite a development evaluation report")
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes.decode("utf-8"))
    prediction_bytes = predictions_path.read_bytes()
    if (freeze.get("schema") != "leader_magic_dev_prediction_freeze_v1"
            or freeze.get("role") != "development_validation"
            or freeze.get("independent_test") is not False
            or freeze.get("ground_truth_was_read_by_predictor") is not False):
        raise ValueError("prediction freeze does not declare GT-free development inference")
    if args.role == "l0_reference_development_validation":
        if (args.l0_report or freeze.get("magic_enabled") is not False
                or freeze.get("checkpoint_sha256") != OFFICIAL_L0_SHA256):
            raise ValueError("L0 reference evaluation requires the unmodified LiDAR-only checkpoint")
    elif args.role == "development_validation":
        if not args.l0_report or freeze.get("magic_enabled") is not True:
            raise ValueError("candidate validation requires MaGiC and a frozen L0 normalizer")
    else:
        raise ValueError("unknown development evaluation role")
    expected_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictor_sha256": sha256_file(predictor_path),
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "evaluator_sha256": sha256_file(evaluator_path),
    }
    expected_paths = {
        "checkpoint_path": str(checkpoint_path),
        "predictor_path": str(predictor_path),
        "prediction_path": str(predictions_path),
        "evaluator_path": str(evaluator_path),
    }
    for field, observed in expected_hashes.items():
        if freeze.get(field) != observed:
            raise ValueError("prediction freeze {} mismatch".format(field))
    for field, observed in expected_paths.items():
        if Path(freeze.get(field, "")).resolve() != Path(observed):
            raise ValueError("prediction freeze {} path mismatch".format(field))
    if freeze.get("runtime_source_hashes") != observed_source_hashes:
        raise ValueError("prediction freeze runtime source hashes differ from the executed sources")
    if freeze.get("predictor_sha256") != observed_source_hashes.get("tools/predict_magic_nclt.py"):
        raise ValueError("prediction freeze predictor hash differs from runtime source manifest")
    if freeze.get("evaluator_sha256") != observed_source_hashes.get("tools/evaluate_magic_nclt_dev.py"):
        raise ValueError("prediction freeze evaluator hash differs from runtime source manifest")
    frames, manifest_hashes, T_BC = _load_public_frames(args.manifest_root)
    if freeze.get("dev_manifest_hashes") != manifest_hashes or freeze.get("frame_count") != EXPECTED_TOTAL:
        raise ValueError("prediction freeze does not bind the exact four dev manifests")
    ordered_hash = canonical_json_sha256([frame["frame_key"] for frame in frames])
    if freeze.get("ordered_frame_keys_sha256") != ordered_hash:
        raise ValueError("prediction freeze frame key ordering hash mismatch")
    if freeze.get("backend") != BACKEND:
        raise ValueError("prediction freeze pose backend differs from the frozen implementation")
    rows = []
    for line_number, line in enumerate(prediction_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError("prediction JSONL contains an empty line")
        row = json.loads(line)
        if not isinstance(row, dict) or "ground_truth" in row or "T_WB_gt" in row:
            raise ValueError("prediction JSONL contains invalid or GT fields")
        rows.append(row)
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError("prediction JSONL must contain exactly 1200 rows")
    gt_poses = {}
    gt_hashes = {}
    gt_sidecar_hashes = {}
    for date in DEV_DATES:
        date_frames = [frame for frame in frames if frame["date"] == date]
        poses, gt_hash, sidecar_hash = _read_gt_date(gt_root, date, date_frames, T_BC[date])
        gt_poses[date] = poses
        gt_hashes[date] = gt_hash
        gt_sidecar_hashes[date] = sidecar_hash
    scored = []
    cursor_by_date = {date: 0 for date in DEV_DATES}
    for expected, row in zip(frames, rows):
        if any(row.get(name) != expected[name] for name in ("date", "timestamp_us", "frame_key")):
            raise ValueError("prediction order/key mismatch at {}".format(expected["frame_key"]))
        if row.get("status") not in ("success", "identity_fallback"):
            raise ValueError("prediction status is outside the frozen schema")
        valid_corr = row.get("valid_corr")
        selected_corr = row.get("selected_corr")
        if (not isinstance(valid_corr, int) or valid_corr < 0
                or not isinstance(selected_corr, int) or selected_corr < 0):
            raise ValueError("prediction correspondence counts must be nonnegative integers")
        date = expected["date"]
        index = cursor_by_date[date]
        truth = gt_poses[date][index]
        cursor_by_date[date] += 1
        mpe, moe = _pose_errors(row.get("T_WB_pred"), truth)
        scored.append({
            "date": date,
            "timestamp_us": expected["timestamp_us"],
            "frame_key": expected["frame_key"],
            "status": row["status"],
            "valid_corr": valid_corr,
            "selected_corr": selected_corr,
            "failure_reason": row.get("failure_reason"),
            "mpe_m": mpe,
            "moe_deg": moe,
        })
    per_date = {}
    for date in DEV_DATES:
        per_date[date] = _summary([row for row in scored if row["date"] == date])
    overall = _summary(scored)
    l0_metrics = None
    selection = None
    if args.l0_report:
        l0_path = Path(args.l0_report).resolve(strict=True)
        l0 = json.loads(l0_path.read_text(encoding="utf-8"))
        if (l0.get("role") != "l0_reference_development_validation"
                or l0.get("dev_manifest_hashes") != manifest_hashes
                or l0.get("sam_cache_manifest_sha256") != freeze.get("sam_cache_manifest_sha256")):
            raise ValueError("L0 normalizer does not use the same dev manifests and SAM cache")
        l0_provenance = l0.get("provenance", {})
        if (l0.get("independent_test") is not False
                or l0_provenance.get("checkpoint_sha256") != OFFICIAL_L0_SHA256
                or l0_provenance.get("runtime_source_hashes") != observed_source_hashes):
            raise ValueError("L0 normalizer does not bind the official checkpoint and same runtime sources")
        l0_metrics = l0.get("overall")
        if (not l0_metrics or l0_metrics.get("frames") != EXPECTED_TOTAL
                or not isinstance(l0_metrics.get("failure_frames"), int)
                or l0_metrics["failure_frames"] < 0 or l0_metrics["failure_frames"] > EXPECTED_TOTAL
                or not isinstance(l0_metrics.get("success_frames"), int)
                or l0_metrics["success_frames"] + l0_metrics["failure_frames"] != EXPECTED_TOTAL):
            raise ValueError("L0 normalizer has an incomplete denominator")
        try:
            l0_mpe = float(l0_metrics["mpe_mean_m"])
            l0_moe = float(l0_metrics["moe_mean_deg"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("L0 normalizer metrics must be positive")
        if not np.isfinite([l0_mpe, l0_moe]).all() or l0_mpe <= 0 or l0_moe <= 0:
            raise ValueError("L0 normalizer metrics must be positive")
        if (set(l0.get("per_date", {})) != set(DEV_DATES)
                or any(value.get("frames") != EXPECTED_PER_DATE
                       for value in l0["per_date"].values())):
            raise ValueError("L0 normalizer per-date denominators are incomplete")
        selection = {
            "failed_frames_first": overall["failure_frames"],
            "normalized_pose_objective": (
                overall["mpe_mean_m"] / l0_mpe
                + overall["moe_mean_deg"] / l0_moe
            ),
            "l0_reference_report_sha256": sha256_file(l0_path),
        }
    report = {
        "schema": "leader_magic_development_evaluation_v1",
        "role": args.role,
        "independent_test": False,
        "ground_truth_was_used_for_checkpoint_selection": args.role == "development_validation",
        "protocol": {
            "dates": list(DEV_DATES),
            "frames_per_date": EXPECTED_PER_DATE,
            "total_frames": EXPECTED_TOTAL,
            "failure_denominator": "all frozen frames; identity fallback is scored as its emitted pose",
            "translation_error": "Euclidean T_WB translation error in meters",
            "rotation_error": "SO(3) geodesic angle in degrees",
            "ground_truth_composition": "T_WB = T_WC @ inverse(T_BC)",
        },
        "provenance": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
            "predictor_path": str(predictor_path),
            "predictor_sha256": expected_hashes["predictor_sha256"],
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": expected_hashes["evaluator_sha256"],
            "prediction_path": str(predictions_path),
            "prediction_sha256": expected_hashes["prediction_sha256"],
            "prediction_freeze_path": str(freeze_path),
            "prediction_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
            "training_protocol_sha256": freeze.get("training_protocol_sha256"),
            "dev_manifest_hashes": manifest_hashes,
            "sam_cache_manifest_sha256": freeze.get("sam_cache_manifest_sha256"),
            "runtime_source_hashes": observed_source_hashes,
            "dev_gt_manifest_hashes": gt_hashes,
            "dev_gt_sidecar_sha256": gt_sidecar_hashes,
        },
        "dev_manifest_hashes": manifest_hashes,
        "sam_cache_manifest_sha256": freeze.get("sam_cache_manifest_sha256"),
        "overall": overall,
        "per_date": per_date,
        "selection": selection,
        "rows": scored,
    }
    if guard is not None:
        guard.require(64 * 1024 * 1024)
    safe_json_dump(report_path, report)
    print(json.dumps({"overall": overall, "per_date": per_date,
                      "selection": selection}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--dev-gt-root", type=Path, required=True)
    parser.add_argument("--l0-report", type=Path)
    parser.add_argument("--role", choices=("l0_reference_development_validation",
                                            "development_validation"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    score_predictions(args)


if __name__ == "__main__":
    main()
