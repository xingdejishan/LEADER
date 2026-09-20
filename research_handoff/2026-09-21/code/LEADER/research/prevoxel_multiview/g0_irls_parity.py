import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oracle_pose_refinement import load_module
from surface_patch_refinement import pose_from_baseline_online


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_difference(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    translation_m = float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
    relative_rotation = candidate[:3, :3] @ reference[:3, :3].T
    rotation_deg = float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))
    return translation_m, rotation_deg


def reference_poses(payload):
    records = payload.get("records", [])
    output = {}
    for record in records:
        frame_id = str(record["frame_id"])
        if frame_id in output:
            raise ValueError("duplicate frame_id in baseline predictions: %s" % frame_id)
        pose = record.get("initial_pose", record.get("final_pose"))
        if pose is None:
            raise ValueError("record has neither initial_pose nor final_pose: %s" % frame_id)
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid reference pose for frame: %s" % frame_id)
        output[frame_id] = pose
    return output


def evaluate_parity(rows, baseline, runner, translation_tolerance_m, rotation_tolerance_deg, seed):
    records = []
    for index, row in enumerate(rows):
        frame_id = str(row["frame_id"])
        if frame_id not in baseline:
            raise ValueError("baseline predictions missing selected frame: %s" % frame_id)
        candidate = runner(row, seed + index)
        translation_m, rotation_deg = pose_difference(candidate, baseline[frame_id])
        records.append({
            "frame_id": frame_id,
            "reference_pose": baseline[frame_id].tolist(),
            "replayed_pose": np.asarray(candidate, dtype=np.float64).tolist(),
            "translation_difference_m": translation_m,
            "rotation_difference_deg": rotation_deg,
            "within_tolerance": bool(translation_m <= translation_tolerance_m and rotation_deg <= rotation_tolerance_deg),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Replay LEADER + IRLS and compare it with an existing B0 pose artifact.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-split", default="validation", choices=("validation", "train"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--translation-tolerance-m", type=float, default=5e-5)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2e-5)
    args = parser.parse_args()
    if args.frames < 0 or args.translation_tolerance_m < 0 or args.rotation_tolerance_deg < 0:
        parser.error("frames and tolerances must be non-negative")

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = ([row for row in rows if row["split"] == "train"] if args.evaluate_split == "train"
                else [row for row in rows if row["split"] in ("val", "validation", "test")])
    if args.frames:
        selected = selected[:args.frames]
    if not selected:
        raise ValueError("no selected evaluation frames")
    baseline_path = Path(args.baseline_predictions)
    baseline = reference_poses(json.loads(baseline_path.read_text(encoding="utf-8")))

    matcher_module = load_module("g0_parity_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("g0_parity_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)

    def runner(row, seed):
        pose, _ = pose_from_baseline_online(row, args.lidar_cache, matcher,
                                             full_pool_module.full_pool_refine, args.device, seed)
        return pose

    records = evaluate_parity(selected, baseline, runner, args.translation_tolerance_m,
                              args.rotation_tolerance_deg, args.seed)
    translation = np.asarray([record["translation_difference_m"] for record in records])
    rotation = np.asarray([record["rotation_difference_deg"] for record in records])
    result = {
        "name": "G0 LEADER + IRLS parity replay",
        "reference_role": "B0 existing pose artifact",
        "no_gt_used": True,
        "protocol": {
            "manifest": str(args.manifest),
            "manifest_sha256": file_sha256(args.manifest),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": file_sha256(baseline_path),
            "full_pool": str(args.full_pool),
            "full_pool_sha256": file_sha256(args.full_pool),
            "evaluate_split": args.evaluate_split,
            "seed": args.seed,
            "translation_tolerance_m": args.translation_tolerance_m,
            "rotation_tolerance_deg": args.rotation_tolerance_deg,
        },
        "summary": {
            "frames": len(records),
            "passed_frames": int(sum(record["within_tolerance"] for record in records)),
            "failed_frames": int(sum(not record["within_tolerance"] for record in records)),
            "max_translation_difference_m": float(translation.max()),
            "max_rotation_difference_deg": float(rotation.max()),
            "mean_translation_difference_m": float(translation.mean()),
            "mean_rotation_difference_deg": float(rotation.mean()),
            "passed": bool(all(record["within_tolerance"] for record in records)),
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not result["summary"]["passed"]:
        raise SystemExit("G0 parity failed; inspect the output artifact before enabling any new solver behavior.")


if __name__ == "__main__":
    main()
