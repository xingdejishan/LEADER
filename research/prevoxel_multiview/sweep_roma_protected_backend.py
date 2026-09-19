import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oracle_pose_refinement import load_module, metrics, pose_error, pose_from_baseline
from local_visual_refinement_roma import (
    holdout_mask,
    load_match_cache,
    refine_pose_protected,
    reference_pair_ids,
    block_soft_l1_cost,
    whitened_reprojection,
)


DEFAULT_CONFIGS = [
    {"name": "tight_low_visual", "sigma_t": .04, "sigma_r_deg": .4, "visual_lambda": .25, "floor_px": 1., "accept_ratio": .99},
    {"name": "tight_balanced", "sigma_t": .04, "sigma_r_deg": .4, "visual_lambda": .5, "floor_px": 1., "accept_ratio": .99},
    {"name": "medium_low_visual", "sigma_t": .08, "sigma_r_deg": .8, "visual_lambda": .25, "floor_px": 1., "accept_ratio": .99},
    {"name": "medium_balanced", "sigma_t": .08, "sigma_r_deg": .8, "visual_lambda": .5, "floor_px": 1., "accept_ratio": .99},
    {"name": "medium_soft_precision", "sigma_t": .08, "sigma_r_deg": .8, "visual_lambda": .5, "floor_px": 2., "accept_ratio": .99},
    {"name": "medium_lenient_gate", "sigma_t": .08, "sigma_r_deg": .8, "visual_lambda": .5, "floor_px": 1., "accept_ratio": .98},
    {"name": "loose_low_visual", "sigma_t": .12, "sigma_r_deg": 1.2, "visual_lambda": .25, "floor_px": 1., "accept_ratio": .99},
    {"name": "loose_balanced", "sigma_t": .12, "sigma_r_deg": 1.2, "visual_lambda": .5, "floor_px": 1., "accept_ratio": .99},
]


def evaluate_frame(frame, config, max_translation, max_rotation, max_nfev, robust_scale, holdout_modulus, min_holdout):
    initial, gt, views, points, pixels, cameras, precisions, anchor_ids, reference_frames, reference_cameras = frame
    before = pose_error(initial, gt)
    group_ids = reference_pair_ids(cameras, reference_frames, reference_cameras)
    holdout = holdout_mask(group_ids, holdout_modulus, min_holdout)
    fit = ~holdout
    candidate = initial.copy()
    candidate_after = before
    accepted, reason = False, "insufficient_correspondences"
    holdout_before = holdout_after = float("nan")
    if len(points) >= 6 and int(fit.sum()) >= 6 and len(np.unique(cameras[fit])):
        candidate, optimizer, _, _, _ = refine_pose_protected(
            initial, points[fit], pixels[fit], cameras[fit], precisions[fit], views, max_translation, max_rotation,
            max_nfev, config["sigma_t"], math.radians(config["sigma_r_deg"]), config["visual_lambda"], 1.,
            config["floor_px"], robust_scale, 4)
        delta = np.asarray(optimizer.x, dtype=np.float64)
        if np.isfinite(candidate).all():
            candidate_after = pose_error(candidate, gt)
        if int(holdout.sum()) >= min_holdout:
            holdout_before = block_soft_l1_cost(whitened_reprojection(
                initial, points[holdout], pixels[holdout], cameras[holdout], precisions[holdout], views, 1., config["floor_px"]), robust_scale)
            holdout_after = block_soft_l1_cost(whitened_reprojection(
                candidate, points[holdout], pixels[holdout], cameras[holdout], precisions[holdout], views, 1., config["floor_px"]), robust_scale)
        if not optimizer.success:
            reason = "solver_failed"
        elif not np.isfinite(candidate).all() or not np.isfinite(delta).all():
            reason = "nonfinite_candidate"
        elif np.linalg.norm(candidate[:3, 3] - initial[:3, 3]) > max_translation + 1e-9:
            reason = "translation_bound"
        elif np.linalg.norm(delta[3:]) > max_rotation + 1e-9:
            reason = "rotation_bound"
        elif int(holdout.sum()) < min_holdout:
            reason = "insufficient_holdout"
        elif not np.isfinite(holdout_before + holdout_after) or holdout_after > config["accept_ratio"] * holdout_before:
            reason = "holdout_not_improved"
        else:
            accepted, reason = True, "accepted"
    final = candidate_after if accepted else before
    return {"before": list(before), "candidate_after": list(candidate_after), "after": list(final), "accepted": accepted,
            "reason": reason, "holdout_improvement_ratio": 1. - holdout_after / holdout_before if np.isfinite(holdout_before + holdout_after) and holdout_before > 0 else float("nan"),
            "fit_correspondences": int(fit.sum()), "holdout_correspondences": int(holdout.sum())}


def pareto_frontier(sweeps):
    result = []
    for candidate in sweeps:
        current = candidate["metrics"]["final"]
        dominated = any(
            other is not candidate and other["metrics"]["final"]["mean_translation_m"] <= current["mean_translation_m"] and
            other["metrics"]["final"]["mean_rotation_deg"] <= current["mean_rotation_deg"] and
            (other["metrics"]["final"]["mean_translation_m"] < current["mean_translation_m"] or
             other["metrics"]["final"]["mean_rotation_deg"] < current["mean_rotation_deg"])
            for other in sweeps)
        if not dominated:
            result.append(candidate["config"]["name"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--match-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--configs")
    parser.add_argument("--max-translation", type=float, default=2.)
    parser.add_argument("--max-rotation-deg", type=float, default=10.)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--robust-scale-whitened", type=float, default=1.)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--min-holdout", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = [row for row in json.loads(Path(args.manifest).read_text(encoding="utf-8")) if row["split"] == "train"]
    if args.frames:
        rows = rows[:args.frames]
    configs = json.loads(Path(args.configs).read_text(encoding="utf-8")) if args.configs else DEFAULT_CONFIGS
    matcher_module = load_module("roma_sweep_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("roma_sweep_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    frames = []
    for index, row in enumerate(rows):
        initial, gt, _ = pose_from_baseline(row, args.lidar_cache, matcher, full_pool_module.full_pool_refine, args.device, args.seed + index)
        cached = load_match_cache(Path(args.match_cache_dir) / (row["frame_id"] + ".npz"))
        points, pixels, cameras, _, precisions, anchor_ids, reference_frames, reference_cameras, _ = cached
        frames.append((initial, gt, row["views"], points, pixels, cameras, precisions, anchor_ids, reference_frames, reference_cameras))
        print("baseline %d/%d %s" % (index + 1, len(rows), row["frame_id"]), flush=True)
    baseline_records = [{"before": list(pose_error(frame[0], frame[1]))} for frame in frames]
    baseline = metrics(baseline_records, "before")
    sweeps = []
    for config in configs:
        records = [evaluate_frame(frame, config, args.max_translation, math.radians(args.max_rotation_deg), args.max_nfev,
                                  args.robust_scale_whitened, args.holdout_modulus, args.min_holdout) for frame in frames]
        accepted = int(sum(record["accepted"] for record in records))
        final_metrics = metrics(records, "after")
        sweeps.append({"config": config, "metrics": {"candidate": metrics(records, "candidate_after"), "final": final_metrics},
                       "accepted_frames": accepted, "records": records})
        print("%s accepted=%d/%d MPE=%.5f MOE=%.5f" % (config["name"], accepted, len(records),
              final_metrics["mean_translation_m"], final_metrics["mean_rotation_deg"]), flush=True)
    eligible = [sweep for sweep in sweeps if sweep["metrics"]["final"]["mean_translation_m"] < baseline["mean_translation_m"] and
                sweep["metrics"]["final"]["mean_rotation_deg"] < baseline["mean_rotation_deg"]]
    chosen = min(eligible, key=lambda sweep: sweep["metrics"]["final"]["mean_translation_m"] / baseline["mean_translation_m"] +
                 sweep["metrics"]["final"]["mean_rotation_deg"] / baseline["mean_rotation_deg"]) if eligible else None
    result = {"protocol": "train-only frozen RoMa cache backend selection", "baseline": baseline, "sweeps": sweeps,
              "selection": {"eligible_both_metrics_improved": [sweep["config"]["name"] for sweep in eligible],
                            "pareto_frontier": pareto_frontier(sweeps), "chosen": chosen["config"] if chosen else None,
                            "criterion": "both train MPE and MOE lower than LEADER, then minimum normalized sum"}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
