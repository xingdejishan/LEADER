import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import numpy as np

DEV_DATES = ("2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26")
EXPECTED_MANIFEST_SHA256 = {
    "2012-02-12": "7e0369a8cf90be044650daa5f0f0e90923b674d7ca08724c122ab96e0e3a06ad",
    "2012-02-19": "3d91e68296357b64c10128f8156b14374b7b1564bc7289a174367c2e1308b458",
    "2012-03-31": "b5a8d08517269891dfa001cf6812d45cf3c63adda47d3d75b04374973e088064",
    "2012-05-26": "9442142a38347c4ea984367dcb0d9bf11504382ed6cecc414b669dce865401f8",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_child(root, date, relative):
    raw = str(relative)
    path = PurePosixPath(raw)
    if ("\\" in raw or ":" in raw or path.is_absolute() or not path.parts
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ValueError("unsafe evaluator GT path")
    root = Path(root).resolve(strict=True)
    date_root = (root / date).resolve(strict=True)
    child = date_root.joinpath(*path.parts).resolve(strict=True)
    if root not in date_root.parents or date_root not in child.parents:
        raise ValueError("evaluator GT path escapes its date directory")
    return child


def ground_truth(public_root, gt_root):
    expected = []
    transforms = []
    manifest_hashes = {}
    gt_hashes = {}
    for date in DEV_DATES:
        public_path = Path(public_root) / date / "manifest.json"
        if sha256_file(public_path) != EXPECTED_MANIFEST_SHA256[date]:
            raise ValueError("public manifest SHA-256 mismatch")
        document = json.loads(public_path.read_text(encoding="utf-8"))
        rows = document.get("frames")
        if document.get("split") != "test" or not isinstance(rows, list) or len(rows) != 300:
            raise ValueError("public validation manifest must contain 300 rows")
        if any("ground_truth" in row for row in rows):
            raise ValueError("public validation manifest contains GT")
        t_bc = np.asarray(document["roi"]["T_BC"], dtype=np.float64)
        if t_bc.shape != (4, 4):
            raise ValueError("invalid T_BC")
        gt_manifest = Path(gt_root) / date / "manifest.json"
        gt_digest = sha256_file(gt_manifest)
        sidecar = gt_manifest.with_suffix(".sha256")
        if sidecar.read_text(encoding="ascii").split()[0] != gt_digest:
            raise ValueError("evaluator-only manifest sidecar mismatch")
        gt_document = json.loads(gt_manifest.read_text(encoding="utf-8"))
        gt_rows = gt_document.get("frames")
        if (gt_document.get("split") != "test_evaluator_only" or not isinstance(gt_rows, list)
                or len(gt_rows) != 300):
            raise ValueError("evaluator-only manifest must contain 300 rows")
        for index, (row, gt_row) in enumerate(zip(rows, gt_rows)):
            stamp = int(row["timestamp_us"])
            if gt_row.get("index") != index or int(gt_row["timestamp_us"]) != stamp:
                raise ValueError("GT row order differs from public manifest")
            gt = gt_row["ground_truth"]
            if (gt.get("pose") != "T_WC" or not re.fullmatch(r"[0-9a-f]{64}",
                                                              str(gt.get("sha256", "")))):
                raise ValueError("invalid evaluator-only GT record")
            path = safe_child(gt_root, date, gt["path"])
            if sha256_file(path) != gt["sha256"]:
                raise ValueError("GT file SHA-256 mismatch")
            t_wc = np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)
            if not np.isfinite(t_wc).all() or not np.allclose(t_wc[3], [0, 0, 0, 1]):
                raise ValueError("GT pose is invalid")
            expected.append("{}/{}".format(date, stamp))
            transforms.append(t_wc @ np.linalg.inv(t_bc))
        manifest_hashes[date] = sha256_file(public_path)
        gt_hashes[date] = gt_digest
    return expected, transforms, manifest_hashes, gt_hashes


def read_predictions(path, expected):
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(expected) or [row.get("frame_key") for row in rows] != expected:
        raise ValueError("prediction keys, order or denominator differ")
    return rows


def score_rows(rows, truths):
    errors = []
    for row, truth in zip(rows, truths):
        predicted = np.asarray(row["T_WB_pred"], dtype=np.float64)
        if predicted.shape != (4, 4) or not np.isfinite(predicted).all():
            raise ValueError("prediction pose is not finite [4,4]")
        if (not np.allclose(predicted[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(predicted[:3, :3].T @ predicted[:3, :3], np.eye(3), atol=1e-2)
                or abs(np.linalg.det(predicted[:3, :3]) - 1) > 1e-2):
            raise ValueError("prediction pose is not rigid")
        mpe = float(np.linalg.norm(predicted[:3, 3] - truth[:3, 3]))
        relative = predicted[:3, :3] @ truth[:3, :3].T
        moe = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) * 0.5, -1, 1))))
        errors.append((mpe, moe))
    return np.asarray(errors, dtype=np.float64)


def describe(errors, rows):
    return {"frames": len(errors), "success": sum(row["status"] == "success" for row in rows),
            "failures": sum(row["status"] != "success" for row in rows),
            "mpe_mean_m": float(errors[:, 0].mean()), "moe_mean_deg": float(errors[:, 1].mean()),
            "mpe_p90_m": float(np.percentile(errors[:, 0], 90)),
            "moe_p90_deg": float(np.percentile(errors[:, 1], 90)),
            "joint_1m_2deg": int(((errors[:, 0] < 1) & (errors[:, 1] < 2)).sum())}


def paired_intervals(candidate, baseline, rows):
    delta = candidate - baseline
    groups = {}
    first = {}
    for index, row in enumerate(rows):
        date = row["date"]
        stamp = int(row["timestamp_us"])
        first.setdefault(date, stamp)
        block = (stamp - first[date]) // 30000000
        groups.setdefault(date, {}).setdefault(block, []).append(index)
    rng = np.random.RandomState(37)
    samples = np.zeros((10000, 2), dtype=np.float64)
    for iteration in range(len(samples)):
        selected = []
        for date in DEV_DATES:
            blocks = list(groups[date].values())
            for pick in rng.randint(0, len(blocks), size=len(blocks)):
                selected.extend(blocks[pick])
        samples[iteration] = delta[selected].mean(axis=0)
    return {"mean_delta_m_deg": delta.mean(axis=0).tolist(),
            "block30s_ci95_m_deg": np.percentile(samples, [2.5, 97.5], axis=0).T.tolist(),
            "improved_frames_mpe": int((delta[:, 0] < 0).sum()),
            "improved_frames_moe": int((delta[:, 1] < 0).sum())}


def run(args):
    freeze_path = Path(args.freeze).resolve(strict=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    predictions = Path(args.predictions).resolve(strict=True)
    baseline_path = Path(args.baseline_predictions).resolve(strict=True)
    if (freeze.get("schema") != "leader_cmr_cam5_prediction_v1" or freeze.get("frames") != 1200
            or freeze.get("ground_truth_read_online") is not False
            or freeze.get("prediction_sha256") != sha256_file(predictions)
            or freeze.get("baseline_prediction_sha256") != sha256_file(baseline_path)):
        raise ValueError("frozen CMR prediction identity mismatch")
    expected, truths, manifests, gt_hashes = ground_truth(args.manifest_root, args.gt_root)
    if freeze.get("validation_manifest_sha256") != manifests:
        raise ValueError("validation manifest hashes differ between runner and evaluator")
    cmr_rows = read_predictions(predictions, expected)
    baseline_rows = read_predictions(baseline_path, expected)
    if any(row["status"] != "success" for row in baseline_rows):
        raise ValueError("frozen L0 baseline is not 1200/1200 successful")
    cmr_errors = score_rows(cmr_rows, truths)
    baseline_errors = score_rows(baseline_rows, truths)
    baseline_summary = describe(baseline_errors, baseline_rows)
    if (abs(baseline_summary["mpe_mean_m"] - 0.2025234758270557) > 1e-6
            or abs(baseline_summary["moe_mean_deg"] - 1.6661374730795524) > 1e-6):
        raise ValueError("independent evaluator did not reproduce the frozen 1200-frame L0 reference")
    result = {"schema": "leader_cmr_cam5_evaluation_v1",
              "role": "development_validation", "independent_test": False,
              "prediction_sha256": sha256_file(predictions),
              "freeze_sha256": sha256_file(freeze_path),
              "baseline_prediction_sha256": sha256_file(baseline_path),
              "manifest_sha256": manifests, "gt_manifest_sha256": gt_hashes,
              "baseline": baseline_summary, "cmr": describe(cmr_errors, cmr_rows),
              "paired": paired_intervals(cmr_errors, baseline_errors, cmr_rows),
              "per_date": {date: {"baseline": describe(baseline_errors[i * 300:(i + 1) * 300],
                                                       baseline_rows[i * 300:(i + 1) * 300]),
                                  "cmr": describe(cmr_errors[i * 300:(i + 1) * 300],
                                                  cmr_rows[i * 300:(i + 1) * 300])}
                           for i, date in enumerate(DEV_DATES)}}
    output = Path(args.output).resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "cmr": result["cmr"],
                      "paired": result["paired"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
