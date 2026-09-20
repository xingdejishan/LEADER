import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_lidar, load_module
from surface_patch_refinement import (
    apply_pose_delta,
    digest_json,
    fit_surface_plane,
    patch_geometry_transform,
    patch_query_residual,
    pose_from_baseline_online,
    prepare_patch_context,
    select_surface_patches,
)


def frame_clouds(rows, lidar_cache):
    clouds = {}
    for row in rows:
        frame_id = str(row["frame_id"])
        cached = load_lidar(lidar_cache, frame_id)
        source = np.asarray(cached["source"], dtype=np.float64)
        gt = np.asarray(cached["GT"], dtype=np.float64)
        world = source @ gt[:3, :3].T + gt[:3, 3]
        clouds[frame_id] = (cKDTree(world), world)
    return clouds


def estimate_parameter_covariance(references, reference_context, clouds, radius, min_neighbors,
                                  sample_count, seed):
    surface_points, surface_tree = reference_context[:2]
    anchors = np.asarray(references["world_xyz"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(anchors), size=min(sample_count, len(anchors)), replace=False)
    covariances = []
    frame_counts = []
    for index in sample_indices:
        anchor = anchors[index]
        nominal = fit_surface_plane(anchor, surface_tree, surface_points, radius, min_neighbors)
        if nominal is None:
            continue
        values = []
        for tree, points in clouds.values():
            estimate = fit_surface_plane(anchor, tree, points, radius, min_neighbors)
            if estimate is None:
                continue
            normal, distance = estimate[:2]
            if float(normal @ nominal[0]) < 0.:
                normal = -normal
                distance = -distance
            values.append(np.r_[normal, distance - normal @ anchor])
        if len(values) < 3:
            continue
        covariances.append(np.cov(np.asarray(values), rowvar=False, ddof=1))
        frame_counts.append(len(values))
    if not covariances:
        raise ValueError("no usable train-frame plane variation samples")
    covariance = np.median(np.asarray(covariances), axis=0)
    covariance = .5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-10)
    covariance = (vectors * values[None, :]) @ vectors.T
    return covariance, {"sampled_anchors": int(len(sample_indices)),
                        "usable_anchors": int(len(covariances)),
                        "frame_count_median": float(np.median(frame_counts)),
                        "frame_count_p10": float(np.percentile(frame_counts, 10)),
                        "frame_count_p90": float(np.percentile(frame_counts, 90))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--online-cache", required=True)
    parser.add_argument("--reference-lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=12000)
    parser.add_argument("--uncertainty-radius", type=float, default=1.2)
    parser.add_argument("--min-plane-neighbors", type=int, default=8)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--surface-voxel-size", type=float, default=.2)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--min-view-cosine", type=float, default=.7)
    parser.add_argument("--plane-radius", type=float, default=.8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-patches-per-camera", type=int, default=48)
    parser.add_argument("--grid-cell", type=int, default=32)
    parser.add_argument("--min-contrast", type=float, default=.03)
    args = parser.parse_args()

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
    references = build_reference_observations(
        rows, args.reference_lidar_cache, args.projection_cache, args.map_voxel_size,
        Path(args.map_cache), 0)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    reference_context = prepare_patch_context(
        references, rows_by_frame, args.reference_lidar_cache, args.surface_voxel_size)
    clouds = frame_clouds(train_rows, args.reference_lidar_cache)
    covariance, covariance_diagnostics = estimate_parameter_covariance(
        references, reference_context, clouds, args.uncertainty_radius,
        args.min_plane_neighbors, args.sample_count, args.seed)

    matcher_module = load_module("geometry_uncertainty_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("geometry_uncertainty_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    model_for_ratio = {
        "parameter_covariance": covariance.tolist(),
        "parameter_steps": [1e-4, 1e-4, 1e-4, 1e-3],
    }
    ratios = []
    information_ratios = []
    frame_records = []
    shared_plane_cache = {}
    for index, row in enumerate(train_rows):
        initial, _ = pose_from_baseline_online(
            row, args.online_cache, matcher, full_pool_module.full_pool_refine,
            args.device, args.seed + index, return_lidar_evidence=False)
        patches, query_images, diagnostics = select_surface_patches(
            row, initial, references, rows_by_frame, reference_context, args.crop_radius,
            args.min_view_cosine, args.plane_radius, args.min_plane_neighbors, args.patch_size,
            args.max_patches_per_camera, args.grid_cell, args.min_contrast,
            shared_plane_cache=shared_plane_cache)
        camera_data = {int(view["camera"]): (
            np.asarray(view["camera_to_body"], dtype=np.float64),
            np.loadtxt(view["calibration"]).astype(np.float64)) for view in row["views"]}
        frame_ratios = []
        for patch in patches:
            transform, ratio, _ = patch_geometry_transform(
                initial, patch, query_images, camera_data, covariance,
                np.asarray(model_for_ratio["parameter_steps"], dtype=np.float64))
            frame_ratios.append(ratio)
            base = patch_query_residual(initial, patch, patch["points"], query_images, camera_data)
            if base is None:
                continue
            pose_steps = np.asarray([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5], dtype=np.float64)
            jacobian = np.zeros((len(base), len(pose_steps)), dtype=np.float64)
            for axis, step in enumerate(pose_steps):
                delta = np.zeros(6, dtype=np.float64)
                delta[axis] = step
                plus = patch_query_residual(
                    apply_pose_delta(initial, delta), patch, patch["points"], query_images, camera_data)
                minus = patch_query_residual(
                    apply_pose_delta(initial, -delta), patch, patch["points"], query_images, camera_data)
                if plus is not None and minus is not None:
                    jacobian[:, axis] = (plus - minus) / (2. * step)
            denominator = float(np.sum(jacobian ** 2))
            if denominator > 1e-16:
                information_ratios.append(float(np.sqrt(np.sum((transform @ jacobian) ** 2) / denominator)))
        ratios.extend(frame_ratios)
        frame_records.append({"frame_id": row["frame_id"], "patch_count": len(patches),
                              "ratio_count": len(frame_ratios), "diagnostics": diagnostics})
        print("%d/%d %s patches=%d ratios=%d" % (
            index + 1, len(train_rows), row["frame_id"], len(patches), len(frame_ratios)), flush=True)

    ratios = np.asarray(ratios, dtype=np.float64)
    information_ratios = np.asarray(information_ratios, dtype=np.float64)
    if not len(ratios) or not len(information_ratios):
        raise ValueError("no train patch propagation ratios")
    payload = {
        "parameterization": ["normal_x", "normal_y", "normal_z", "local_plane_offset_m"],
        "parameter_covariance": covariance.tolist(),
        "parameter_steps": model_for_ratio["parameter_steps"],
        "weak_visual_scale": float(np.median(information_ratios)),
        "weak_visual_scale_rule": "median train-only sqrt(trace(J_pose^T W^T W J_pose) / trace(J_pose^T J_pose))",
        "training_only": True,
        "query_gt_used": False,
        "ratio_count": int(len(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "information_ratio_count": int(len(information_ratios)),
        "information_ratio_p10": float(np.percentile(information_ratios, 10)),
        "information_ratio_median": float(np.median(information_ratios)),
        "information_ratio_p90": float(np.percentile(information_ratios, 90)),
        "covariance_diagnostics": covariance_diagnostics,
        "parameters": {key: value for key, value in vars(args).items() if key != "output"},
        "manifest_sha256": digest_json(rows),
        "frames": frame_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
