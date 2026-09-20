import argparse
import json
from pathlib import Path

import numpy as np


def pose_error(pose, gt):
    delta = pose[:3, :3].T @ gt[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.) / 2., -1., 1.)
    return np.array([
        np.linalg.norm(pose[:3, 3] - gt[:3, 3]),
        np.degrees(np.arccos(cosine)),
    ], dtype=np.float64)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    usable = values[valid]
    return {
        "count": int(len(values)),
        "valid_count": int(len(usable)),
        "failure_count": int(len(values) - len(usable)),
        "MPE_m": float(usable[:, 0].mean()) if len(usable) else float("nan"),
        "MOE_deg": float(usable[:, 1].mean()) if len(usable) else float("nan"),
        "P90_translation_m": float(np.percentile(usable[:, 0], 90)) if len(usable) else float("nan"),
        "P90_rotation_deg": float(np.percentile(usable[:, 1], 90)) if len(usable) else float("nan"),
    }


def bootstrap(values, seed=2089, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(samples, len(values)))
    return np.percentile(values[sample].mean(axis=1), [2.5, 97.5]).tolist()


def load_gt(cache_dir, frame_id):
    with np.load(Path(cache_dir) / (frame_id + ".npz")) as data:
        return np.asarray(data["GT"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    records = payload["records"]
    names = ("B0_official_irls", "B1_lidar", "C_visual")
    errors = {name: [] for name in names}
    frame_records = []
    for record in records:
        gt = load_gt(args.gt_cache_dir, record["frame_id"])
        b0 = np.asarray(record["b0_pose"], dtype=np.float64)
        poses = {"B0_official_irls": b0}
        for name in ("B1_lidar", "C_visual"):
            poses[name] = np.asarray(record["variants"][name]["final_pose"], dtype=np.float64)
        row = {"frame_id": record["frame_id"]}
        for name, pose in poses.items():
            error = pose_error(pose, gt) if np.isfinite(pose).all() else np.full(2, np.nan)
            errors[name].append(error)
            row[name] = error.tolist()
        frame_records.append(row)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in errors.items()}
    baseline = arrays["B0_official_irls"]
    result = {"protocol": {"predictions": str(args.predictions), "gt_used_only_here": True,
                            "gt_cache_dir": str(args.gt_cache_dir), "frames": len(records)},
              "metrics": {name: stats(values) for name, values in arrays.items()},
              "paired_vs_B0": {}, "paired_vs_B1": {},
              "criteria_vs_B0": {}, "criteria_vs_B1": {}, "records": frame_records}
    for name in ("B1_lidar", "C_visual"):
        paired = arrays[name] - baseline
        valid = np.isfinite(paired).all(axis=1)
        result["paired_vs_B0"][name] = {
            "mean_delta_MPE_m_MOE_deg": np.nanmean(paired, axis=0).tolist(),
            "bootstrap_95ci_mean_delta_MPE": bootstrap(paired[valid, 0]),
            "bootstrap_95ci_mean_delta_MOE": bootstrap(paired[valid, 1]),
            "MPE_improved_frames": int(np.sum(paired[:, 0] < 0)),
            "MOE_improved_frames": int(np.sum(paired[:, 1] < 0)),
            "both_improved_frames": int(np.sum((paired[:, 0] < 0) & (paired[:, 1] < 0))),
        }
        candidate = stats(arrays[name])
        base = stats(baseline)
        result["criteria_vs_B0"][name] = {
            "MPE_decreased": bool(candidate["MPE_m"] < base["MPE_m"]),
            "MOE_decreased": bool(candidate["MOE_deg"] < base["MOE_deg"]),
            "P90_translation_not_worse": bool(candidate["P90_translation_m"] <= base["P90_translation_m"]),
            "P90_rotation_not_worse": bool(candidate["P90_rotation_deg"] <= base["P90_rotation_deg"]),
            "no_failures": bool(candidate["failure_count"] == 0),
        }
    b1 = arrays["B1_lidar"]
    candidate = stats(arrays["C_visual"])
    reference = stats(b1)
    paired = arrays["C_visual"] - b1
    valid = np.isfinite(paired).all(axis=1)
    result["paired_vs_B1"]["C_visual"] = {
        "mean_delta_MPE_m_MOE_deg": np.nanmean(paired, axis=0).tolist(),
        "bootstrap_95ci_mean_delta_MPE": bootstrap(paired[valid, 0]),
        "bootstrap_95ci_mean_delta_MOE": bootstrap(paired[valid, 1]),
        "MPE_improved_frames": int(np.sum(paired[:, 0] < 0)),
        "MOE_improved_frames": int(np.sum(paired[:, 1] < 0)),
        "both_improved_frames": int(np.sum((paired[:, 0] < 0) & (paired[:, 1] < 0))),
    }
    result["criteria_vs_B1"]["C_visual"] = {
        "MPE_decreased": bool(candidate["MPE_m"] < reference["MPE_m"]),
        "MOE_decreased": bool(candidate["MOE_deg"] < reference["MOE_deg"]),
        "P90_translation_not_worse": bool(candidate["P90_translation_m"] <= reference["P90_translation_m"]),
        "P90_rotation_not_worse": bool(candidate["P90_rotation_deg"] <= reference["P90_rotation_deg"]),
        "no_failures": bool(candidate["failure_count"] == 0),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
