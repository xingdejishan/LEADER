import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_module
from surface_patch_refinement import (
    apply_pose_delta,
    digest_json,
    patch_visual_residual,
    prepare_patch_context,
    select_surface_patches,
)


HUBER_DELTA_M = 0.5
HUBER_ITERS = 10


def load_cache(cache_dir, frame_id):
    path = Path(cache_dir) / (frame_id + ".npz")
    with np.load(path) as data:
        required = {"source", "prediction", "center", "T_corr", "T0_official"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("official cache is missing %s: %s" % (sorted(missing), path))
        result = {key: np.asarray(data[key]) for key in required}
        if "topk_indices" in data.files:
            result["topk_indices"] = np.asarray(data["topk_indices"])
    if result["source"].ndim != 2 or result["source"].shape[1] != 3:
        raise ValueError("source must be [N,3]: %s" % path)
    if result["prediction"].ndim != 2 or result["prediction"].shape[1] < 4:
        raise ValueError("prediction must be [N,4+]: %s" % path)
    if len(result["source"]) != len(result["prediction"]):
        raise ValueError("source/prediction order mismatch: %s" % path)
    for key in ("center", "T_corr", "T0_official"):
        if not np.isfinite(result[key]).all():
            raise ValueError("non-finite %s: %s" % (key, path))
    if result["T_corr"].shape != (4, 4) or result["T0_official"].shape != (4, 4):
        raise ValueError("T_corr and T0_official must be 4x4: %s" % path)
    if result["T0_official"].shape != (4, 4):
        raise ValueError("T0_official must be 4x4: %s" % path)
    return result


def verify_cache_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "concat_mode": "official",
        "t_corr_applied_in_cache": False,
        "query_gt_stored_in_cache": False,
        "source_order": "official_voxel_centers_l",
        "target_order": "official_decoder_output",
        "prediction_dtype": "float32",
        "topk_rule": "max(min(50,N),int(0.5*N))",
        "baseline_pose_stored": True,
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items()
                  if payload.get(key) != value}
    if mismatches:
        raise ValueError("cache protocol mismatch: %s" % mismatches)
    return payload


def official_pose(cached, irls):
    import torch

    source = torch.as_tensor(cached["source"], dtype=torch.float32)
    prediction = torch.as_tensor(cached["prediction"], dtype=torch.float32)
    keep_count = max(min(50, len(prediction)), int(0.5 * len(prediction)))
    keep = prediction[:, 3].topk(keep_count).indices
    if "topk_indices" in cached and not np.array_equal(keep.cpu().numpy(), cached["topk_indices"]):
        raise ValueError("cached topk_indices differ from official torch.topk")
    local_pose = irls(source[keep][None], prediction[keep, :3][None],
                      delta=HUBER_DELTA_M, iters=HUBER_ITERS)[0]
    local_pose[:3, 3] += torch.as_tensor(cached["center"], dtype=torch.float32)
    recomputed = local_pose @ torch.as_tensor(cached["T_corr"], dtype=torch.float32)
    recomputed = recomputed.detach().cpu().numpy().astype(np.float64)
    stored = np.asarray(cached["T0_official"], dtype=np.float64)
    if not np.allclose(recomputed, stored, rtol=2e-5, atol=2e-5):
        raise ValueError("stored T0_official does not reproduce official IRLS pose")
    return stored, keep.cpu().numpy()


def huber_vector_residual(pose, source, target, center, t_corr):
    local_pose = pose @ np.linalg.inv(t_corr)
    error = source @ local_pose[:3, :3].T + local_pose[:3, 3] - (target + center)
    distance = np.linalg.norm(error, axis=1)
    normalized = distance / HUBER_DELTA_M
    rho = np.where(normalized <= 1., .5 * normalized ** 2, normalized - .5)
    scale = np.zeros_like(distance)
    nonzero = distance > 1e-12
    scale[nonzero] = np.sqrt(2. * rho[nonzero]) / distance[nonzero]
    return (error * scale[:, None] / math.sqrt(max(len(error), 1))).ravel()


def lidar_diagnostic(pose, source, target, center, t_corr):
    local_pose = pose @ np.linalg.inv(t_corr)
    error = source @ local_pose[:3, :3].T + local_pose[:3, 3] - (target + center)
    return np.linalg.norm(error, axis=1)


def finite_difference_jacobian(residual, steps):
    steps = np.asarray(steps, dtype=np.float64)

    def jacobian(x):
        x = np.asarray(x, dtype=np.float64)
        base = len(residual(x))
        output = np.empty((base, len(x)), dtype=np.float64)
        for axis, step in enumerate(steps):
            offset = np.zeros_like(x)
            offset[axis] = step
            output[:, axis] = (residual(x + offset) - residual(x - offset)) / (2. * step)
        return output

    return jacobian


def refine(initial, source, target, center, t_corr, patches, query_images, row,
           visual_weight, visual_scale, max_translation, max_rotation, translation_step,
           rotation_step, max_nfev, enable_visual):
    camera_data = {int(view["camera"]): (
        np.asarray(view["camera_to_body"], dtype=np.float64),
        np.loadtxt(view["calibration"]).astype(np.float64)) for view in row["views"]}
    pixel_count = max(sum(len(patch["reference"]) for patch in patches), 1)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    t_corr = np.asarray(t_corr, dtype=np.float64)

    def residual(delta):
        pose = apply_pose_delta(initial, delta)
        lidar = huber_vector_residual(pose, source, target, center, t_corr)
        if not enable_visual:
            return lidar
        visual = patch_visual_residual(pose, patches, query_images, camera_data)
        visual = visual / (visual_scale * math.sqrt(pixel_count))
        return np.concatenate([lidar, visual_weight * visual])

    x0 = np.zeros(6, dtype=np.float64)
    bounds = (-np.r_[np.full(3, max_translation), np.full(3, max_rotation)],
              np.r_[np.full(3, max_translation), np.full(3, max_rotation)])
    steps = np.r_[np.full(3, translation_step), np.full(3, rotation_step)]
    result = least_squares(
        residual, x0, jac=finite_difference_jacobian(residual, steps), bounds=bounds,
        method="trf", loss="linear", x_scale="jac", max_nfev=max_nfev)
    final = apply_pose_delta(initial, result.x)
    before_lidar = lidar_diagnostic(initial, source, target, center, t_corr)
    after_lidar = lidar_diagnostic(final, source, target, center, t_corr)
    before_visual = patch_visual_residual(initial, patches, query_images, camera_data) if enable_visual else np.empty(0)
    after_visual = patch_visual_residual(final, patches, query_images, camera_data) if enable_visual else np.empty(0)
    return final, result, before_lidar, after_lidar, before_visual, after_visual


def frame_ids(rows, split, limit):
    selected = [row for row in rows if row["split"] in ("val", "validation", "test")]
    return selected[:limit] if limit else selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reference-lidar-cache", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--visual-scale-file", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--irls-module", default=str(REPO.parent / "IRLS-Huber_全量结果产物" / "irls_huber.py"))
    parser.add_argument("--evaluate-split", default="validation", choices=("validation", "train"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--surface-voxel-size", type=float, default=.2)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--min-view-cosine", type=float, default=.7)
    parser.add_argument("--plane-radius", type=float, default=.8)
    parser.add_argument("--min-plane-neighbors", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-patches-per-camera", type=int, default=48)
    parser.add_argument("--grid-cell", type=int, default=32)
    parser.add_argument("--min-contrast", type=float, default=.03)
    parser.add_argument("--visual-weight", type=float, default=1.)
    parser.add_argument("--max-rotation-deg", type=float, default=2.)
    parser.add_argument("--max-translation-m", type=float, default=.5)
    parser.add_argument("--translation-step-m", type=float, default=1e-4)
    parser.add_argument("--rotation-step-rad", type=float, default=1e-5)
    parser.add_argument("--max-nfev", type=int, default=300)
    args = parser.parse_args()
    if args.patch_size < 4 or args.patch_size % 2:
        parser.error("patch-size must be even and >= 4")
    if args.visual_weight <= 0 or args.max_translation_m <= 0 or args.max_rotation_deg <= 0:
        parser.error("optimization scales must be positive")

    cache_protocol = verify_cache_manifest(args.cache_manifest)
    visual_scale_payload = json.loads(Path(args.visual_scale_file).read_text(encoding="utf-8"))
    visual_scale = float(visual_scale_payload["sigma_V"])
    if not np.isfinite(visual_scale) or visual_scale <= 0:
        raise ValueError("sigma_V must be positive and finite")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    eval_rows = frame_ids(rows, args.evaluate_split, args.frames)
    rows_for_map = rows
    references = build_reference_observations(
        rows_for_map, args.reference_lidar_cache, args.projection_cache,
        args.map_voxel_size, Path(args.map_cache), 0)
    rows_by_frame = {row["frame_id"]: row for row in rows_for_map}
    reference_context = prepare_patch_context(
        references, rows_by_frame, args.reference_lidar_cache, args.surface_voxel_size)
    irls = load_module("official_irls_huber", Path(args.irls_module))
    records = []
    started = time.time()

    for index, row in enumerate(eval_rows):
        cached = load_cache(args.cache_dir, row["frame_id"])
        initial, keep = official_pose(cached, irls.huber_rigid_torch)
        patches, query_images, patch_diagnostics = select_surface_patches(
            row, initial, references, rows_by_frame, reference_context, args.crop_radius,
            args.min_view_cosine, args.plane_radius, args.min_plane_neighbors, args.patch_size,
            args.max_patches_per_camera, args.grid_cell, args.min_contrast, visibility_check=False)
        source = np.asarray(cached["source"], dtype=np.float64)[keep]
        target = np.asarray(cached["prediction"], dtype=np.float64)[keep, :3]
        center = np.asarray(cached["center"], dtype=np.float64)
        t_corr = np.asarray(cached["T_corr"], dtype=np.float64)
        variants = {}
        for name, enable_visual in (("B1_lidar", False), ("C_visual", True)):
            final, result, before_lidar, after_lidar, before_visual, after_visual = refine(
                initial, source, target, center, t_corr, patches if enable_visual else [], query_images, row,
                args.visual_weight, visual_scale, args.max_translation_m, math.radians(args.max_rotation_deg),
                args.translation_step_m, args.rotation_step_rad, args.max_nfev, enable_visual)
            failed = (not bool(result.success) or not np.isfinite(final).all() or
                      (enable_visual and not patches))
            variants[name] = {
                "final_pose": final.tolist() if not failed else np.full((4, 4), np.nan).tolist(),
                "delta": result.x.tolist(),
                "patch_count": len(patches) if enable_visual else 0,
                "solver": {"success": bool(result.success), "status": int(result.status),
                           "nfev": int(result.nfev), "cost": float(result.cost),
                           "optimality": float(result.optimality), "message": str(result.message)},
                "lidar_rmse_before_m": float(np.sqrt(np.mean(before_lidar ** 2))),
                "lidar_rmse_after_m": float(np.sqrt(np.mean(after_lidar ** 2))),
                "visual_rmse_before": float(np.sqrt(np.mean(before_visual ** 2))) if len(before_visual) else float("nan"),
                "visual_rmse_after": float(np.sqrt(np.mean(after_visual ** 2))) if len(after_visual) else float("nan"),
                "run_failed": bool(failed),
            }
        records.append({
            "frame_id": row["frame_id"],
            "b0_pose": initial.tolist(),
            "official_topk_indices": keep.tolist(),
            "patch_count": len(patches),
            "patch_diagnostics": patch_diagnostics,
            "reference_frames": sorted({patch["reference_frame"] for patch in patches}),
            "reference_indices": [int(patch["reference_index"]) for patch in patches],
            "variants": variants,
        })
        print("%s %d/%d %s patches=%d B1_failed=%s C_failed=%s" % (
            args.evaluate_split, index + 1, len(eval_rows), row["frame_id"], len(patches),
            variants["B1_lidar"]["run_failed"], variants["C_visual"]["run_failed"]), flush=True)

    protocol = {
        "name": "official IRLS-Huber baseline with LiDAR and surface-patch comparison",
        "baseline": "official LEADER output with IRLS-Huber delta=0.5m iters=10",
        "concat_mode": "official",
        "t_corr_formula": "T0=(IRLS(source[topk],prediction[topk,:3]); translation+=center) @ T_corr",
        "t_corr_applied_in_cache": False,
        "geometry_objective": "mean Huber(||(T@inv(T_corr)) source-(target+center)|| / 0.5m)",
        "visual_objective": "mean normalized grayscale patch residual squared",
        "visual_weight": args.visual_weight,
        "visual_scale_sigma_V": visual_scale,
        "visual_scale_file": str(args.visual_scale_file),
        "optimizer": {"method": "trf", "loss": "linear", "jacobian": "central finite difference",
                       "translation_step_m": args.translation_step_m,
                       "rotation_step_rad": args.rotation_step_rad, "max_nfev": args.max_nfev},
        "query_gt_in_runner": False,
        "query_gt_in_online_cache": False,
        "reference_lidar_cache": str(args.reference_lidar_cache),
        "no_baseline_fallback": True,
        "mask_enforced_during_selection_and_optimization": True,
        "visibility_filter": False,
        "cache_protocol": cache_protocol,
        "input_manifest": str(args.manifest),
        "reference_map_cache": str(args.map_cache),
        "manifest_sha256": digest_json(rows),
        "reference_observation_count": int(len(references["world_xyz"])),
        "evaluation_frames": len(records),
    }
    payload = {"protocol": protocol, "records": records, "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
