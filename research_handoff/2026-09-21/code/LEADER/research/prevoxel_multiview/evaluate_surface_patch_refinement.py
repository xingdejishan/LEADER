import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_error(pose, gt):
    delta = pose[:3, :3].T @ gt[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(pose[:3, 3] - gt[:3, 3])), float(np.degrees(np.arccos(cosine)))


def metrics(values, total_count):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1) if len(values) else np.zeros(0, dtype=bool)
    usable = values[valid]
    return {
        "count": int(total_count),
        "valid_count": int(len(usable)),
        "failure_count": int(total_count - len(usable)),
        "mean_translation_m": float(usable[:, 0].mean()) if len(usable) else float("nan"),
        "median_translation_m": float(np.median(usable[:, 0])) if len(usable) else float("nan"),
        "p90_translation_m": float(np.percentile(usable[:, 0], 90)) if len(usable) else float("nan"),
        "mean_rotation_deg": float(usable[:, 1].mean()) if len(usable) else float("nan"),
        "median_rotation_deg": float(np.median(usable[:, 1])) if len(usable) else float("nan"),
        "p90_rotation_deg": float(np.percentile(usable[:, 1], 90)) if len(usable) else float("nan"),
        "success_1m_2deg": int(((usable[:, 0] < 1.) & (usable[:, 1] < 2.)).sum()) if len(usable) else 0,
    }


def bootstrap_ci(values, seed=2089, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        return [[float("nan"), float("nan")], [float("nan"), float("nan")]]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5], axis=0).tolist()


def load_gt(cache_dir, frame_id):
    with np.load(Path(cache_dir) / (frame_id + ".npz")) as data:
        return np.asarray(data["GT"], dtype=np.float64)


def candidate_pose(record):
    final = np.asarray(record["final_pose"], dtype=np.float64)
    if np.isfinite(final).all():
        return final
    delta = np.asarray(record.get("delta", record["rotation_delta_rad"]), dtype=np.float64)
    initial = np.asarray(record["initial_pose"], dtype=np.float64)
    if not np.isfinite(delta).all():
        return np.full((4, 4), np.nan, dtype=np.float64)
    pose = initial.copy()
    if len(delta) == 6:
        pose[:3, 3] += delta[:3]
        delta = delta[3:]
    pose[:3, :3] = Rotation.from_rotvec(delta).as_matrix() @ pose[:3, :3]
    return pose


def evaluate_variant(payload, lidar_cache, variant_name=None):
    records = []
    for source_record in payload["records"]:
        record = source_record["variants"][variant_name] if variant_name is not None else source_record
        gt = load_gt(lidar_cache, source_record["frame_id"])
        initial = np.asarray(source_record["initial_pose"], dtype=np.float64)
        final = np.asarray(record["final_pose"], dtype=np.float64)
        diagnostic_candidate = candidate_pose({**record, "initial_pose": source_record["initial_pose"]})
        before = pose_error(initial, gt)
        after = pose_error(final, gt) if np.isfinite(final).all() else (float("nan"), float("nan"))
        candidate_after = pose_error(diagnostic_candidate, gt) if np.isfinite(diagnostic_candidate).all() else (float("nan"), float("nan"))
        records.append({**source_record, "variant": variant_name, "variant_record": record,
                        "before": list(before), "after": list(after), "candidate_after": list(candidate_after),
                        "delta": [after[0] - before[0], after[1] - before[1]] if np.isfinite(after).all() else [float("nan"), float("nan")],
                        "candidate_delta": [candidate_after[0] - before[0], candidate_after[1] - before[1]] if np.isfinite(candidate_after).all() else [float("nan"), float("nan")],
                        "leader_success": bool(before[0] < 1. and before[1] < 2.)})

    before_values = np.asarray([record["before"] for record in records], dtype=np.float64)
    after_values = np.asarray([record["after"] for record in records], dtype=np.float64)
    candidate_values = np.asarray([record["candidate_after"] for record in records], dtype=np.float64)
    paired = after_values - before_values
    candidate_paired = candidate_values - before_values
    finite_paired = paired[np.isfinite(paired).all(axis=1)]
    finite_candidate_paired = candidate_paired[np.isfinite(candidate_paired).all(axis=1)]
    before_success = np.asarray([record["leader_success"] for record in records], dtype=bool)
    after_success = np.isfinite(after_values).all(axis=1) & (after_values[:, 0] < 1.) & (after_values[:, 1] < 2.)
    return {
        "metrics": {"before": metrics(before_values, len(records)), "after": metrics(after_values, len(records)),
                    "finite_candidate_diagnostic": metrics(candidate_values, len(records))},
        "paired": {
            "mean_delta_translation_m_rotation_deg": np.nanmean(paired, axis=0).tolist() if len(records) else [],
            "bootstrap_95ci": bootstrap_ci(finite_paired),
            "finite_candidate_mean_delta_translation_m_rotation_deg": np.nanmean(candidate_paired, axis=0).tolist() if len(records) else [],
            "finite_candidate_bootstrap_95ci": bootstrap_ci(finite_candidate_paired),
            "translation_improved_frames": int(np.sum(paired[:, 0] < 0)),
            "rotation_improved_frames": int(np.sum(paired[:, 1] < 0)),
            "both_improved_frames": int(np.sum((paired[:, 0] < 0) & (paired[:, 1] < 0))),
            "finite_candidate_translation_improved_frames": int(np.sum(candidate_paired[:, 0] < 0)),
            "finite_candidate_rotation_improved_frames": int(np.sum(candidate_paired[:, 1] < 0)),
            "finite_candidate_both_improved_frames": int(np.sum((candidate_paired[:, 0] < 0) & (candidate_paired[:, 1] < 0))),
            "rescue_frames": int(np.sum(~before_success & after_success)),
            "damage_frames": int(np.sum(before_success & ~after_success)),
        },
        "criteria": {
            "mpe_not_worse": bool(np.isfinite(after_values).all() and np.nanmean(after_values[:, 0]) <= np.mean(before_values[:, 0]) + 1e-12),
            "moe_decreased": bool(np.isfinite(after_values).all() and np.nanmean(after_values[:, 1]) < np.mean(before_values[:, 1])),
            "p90_translation_not_worse": bool(np.isfinite(after_values).all() and np.percentile(after_values[:, 0], 90) <= np.percentile(before_values[:, 0], 90)),
            "p90_rotation_not_worse": bool(np.isfinite(after_values).all() and np.percentile(after_values[:, 1], 90) <= np.percentile(before_values[:, 1], 90)),
            "no_runner_failures": bool(not any(record["variant_record"]["run_failed"] for record in records)),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    variant_names = tuple(payload["records"][0]["variants"]) if "variants" in payload["records"][0] else (None,)
    evaluations = {name or "single": evaluate_variant(payload, args.lidar_cache, name) for name in variant_names}
    result = {"protocol": {"predictions": str(args.predictions), "gt_used_only_here": True,
                            "lidar_cache": str(args.lidar_cache)}, "variants": evaluations,
              "source_protocol": payload.get("protocol", {})}
    if len(evaluations) == 1:
        result.update(next(iter(evaluations.values())))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
