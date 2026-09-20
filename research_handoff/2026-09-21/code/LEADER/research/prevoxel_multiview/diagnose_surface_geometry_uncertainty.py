import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_lidar
from surface_patch_refinement import (
    digest_json,
    fit_surface_plane,
    load_gray_mask,
    patch_geometry_transform,
    patch_points_for_plane_delta,
    patch_query_residual,
    prepare_patch_context,
    select_surface_patches,
)
from calibrate_surface_geometry_uncertainty import (
    estimate_parameter_covariance,
    frame_clouds,
)


def parameter_delta(nominal, observed, anchor):
    normal, distance = observed[:2]
    if float(normal @ nominal[0]) < 0.:
        normal = -normal
        distance = -distance
    return np.r_[normal - nominal[0],
                 (distance - normal @ anchor) - (nominal[1] - nominal[0] @ anchor)]


def holdout_plane_deltas(surface_points, surface_tree, holdout_clouds, radius, min_neighbors,
                         max_samples, seed, match_distance=.5, independent_voxel=.5):
    candidates = []
    for frame_id, (_, points) in holdout_clouds.items():
        for anchor in np.asarray(points, dtype=np.float64)[::1]:
            candidates.append((frame_id, anchor))
    if not candidates:
        return np.empty((0, 4), dtype=np.float64), []
    selected = {}
    for frame_id, anchor in candidates:
        key = tuple(np.floor(anchor / independent_voxel).astype(np.int64).tolist())
        selected.setdefault(key, (frame_id, anchor))
    candidates = list(selected.values())
    rng = np.random.default_rng(seed)
    if len(candidates) > max_samples:
        candidates = [candidates[i] for i in rng.choice(len(candidates), max_samples, replace=False)]
    deltas = []
    metadata = []
    for frame_id, anchor in candidates:
        distance, nearest = surface_tree.query(anchor)
        if not np.isfinite(distance) or distance > match_distance:
            continue
        nominal = fit_surface_plane(anchor, surface_tree, surface_points, radius, min_neighbors)
        hold_tree, hold_points = holdout_clouds[frame_id]
        observed = fit_surface_plane(anchor, hold_tree, hold_points, radius, min_neighbors)
        if nominal is None or observed is None:
            continue
        deltas.append(parameter_delta(nominal, observed, anchor))
        metadata.append({"frame_id": frame_id, "anchor": anchor.tolist(),
                         "map_distance_m": float(distance), "surface_index": int(nearest)})
    return np.asarray(deltas, dtype=np.float64), metadata


def build_reference_patch(ref_index, references, rows_by_frame, reference_cache,
                          surface_points, surface_tree, radius, min_neighbors,
                          patch_size, min_contrast):
    frame_id = str(references["frame_ids"][ref_index])
    camera = int(references["camera_ids"][ref_index])
    row = rows_by_frame[frame_id]
    view = next(view for view in row["views"] if int(view["camera"]) == camera)
    image, mask = load_gray_mask(view)
    pose = load_lidar(reference_cache, frame_id)["GT"].astype(np.float64)
    anchor = np.asarray(references["world_xyz"][ref_index], dtype=np.float64)
    plane = fit_surface_plane(anchor, surface_tree, surface_points, radius, min_neighbors)
    if plane is None:
        return None
    calibration = np.loadtxt(view["calibration"]).astype(np.float64)
    reference_uv = np.asarray(references["ref_uv"][ref_index], dtype=np.float64)
    reference_view = {"shape": image.shape, "camera_to_body": np.asarray(view["camera_to_body"], dtype=np.float64)}
    bound = bind_reference_patch(anchor, pose, reference_view, reference_uv, plane, patch_size, calibration)
    if bound is None:
        return None
    points, pixels = bound
    rounded = np.rint(pixels).astype(np.int64)
    if not mask[rounded[:, 1], rounded[:, 0]].all():
        return None
    from surface_patch_refinement import normalize_patch, sample_bilinear
    values, valid = sample_bilinear(image, pixels)
    if not valid.all():
        return None
    reference, contrast = normalize_patch(values)
    if contrast < min_contrast:
        return None
    return {
        "points": points,
        "reference": reference,
        "camera": camera,
        "anchor": anchor,
        "plane": plane,
        "reference_pose": pose,
        "reference_camera_to_body": reference_view["camera_to_body"],
        "reference_calibration": calibration,
        "reference_uv": reference_uv,
        "reference_shape": tuple(image.shape),
        "patch_size": patch_size,
        "reference_frame": frame_id,
        "query_image": image,
        "query_mask": mask,
    }


def projection_frame_clouds(rows, lidar_cache, projection_cache):
    clouds = {}
    for row in rows:
        frame_id = str(row["frame_id"])
        cached = load_lidar(lidar_cache, frame_id)
        with np.load(Path(projection_cache) / (frame_id + ".npz")) as data:
            local = np.asarray(data["projection_xyz"], dtype=np.float64)
        world = local @ cached["GT"][:3, :3].T + cached["GT"][:3, 3]
        clouds[frame_id] = (cKDTree(world), world)
    return clouds


def plane_metrics(deltas, covariance):
    if not len(deltas):
        return {"count": 0}
    inverse = np.linalg.pinv(covariance)
    mahalanobis = np.einsum("ni,ij,nj->n", deltas, inverse, deltas)
    observed_bias = deltas.mean(axis=0)
    observed_std = deltas.std(axis=0, ddof=1) if len(deltas) > 1 else np.full(4, np.nan)
    return {
        "count": int(len(deltas)),
        "mean_delta": observed_bias.tolist(),
        "mean_delta_norm": float(np.linalg.norm(observed_bias)),
        "observed_std": observed_std.tolist(),
        "predicted_std": np.sqrt(np.maximum(np.diag(covariance), 0.)).tolist(),
        "mahalanobis_mean": float(mahalanobis.mean()),
        "mahalanobis_median": float(np.median(mahalanobis)),
        "coverage_95": float(np.mean(mahalanobis <= chi2.ppf(.95, 4))),
    }


def direction_diagnostic(fit_rows, fit_references, rows_by_frame, reference_cache,
                         holdout_clouds, reference_context, covariance, radius, min_neighbors, patch_size,
                         min_contrast, max_samples, seed):
    rng = np.random.default_rng(seed)
    steps = np.asarray([1e-4, 1e-4, 1e-4, 1e-3], dtype=np.float64)
    ratios = []
    residual_norms = []
    candidate_rows = list(fit_rows)
    rng.shuffle(candidate_rows)
    patch_count = 0
    holdout_count = 0
    for row in candidate_rows:
        if patch_count >= max_samples:
            break
        initial = load_lidar(reference_cache, str(row["frame_id"]))["GT"].astype(np.float64)
        patches, query_images, _ = select_surface_patches(
            row, initial, fit_references, rows_by_frame,
            reference_context, 80., .7, .8,
            min_neighbors, patch_size, 48, 32, min_contrast, visibility_check=False)
        camera_data = {int(view["camera"]): (
            np.asarray(view["camera_to_body"], dtype=np.float64),
            np.loadtxt(view["calibration"]).astype(np.float64)) for view in row["views"]}
        if len(patches) > max_samples - patch_count:
            patches = patches[:max_samples - patch_count]
        for patch in patches:
            transform, _, _ = patch_geometry_transform(
                initial, patch, query_images, camera_data, covariance, steps)
            patch_count += 1
            anchor = patch["anchor"]
            nominal = patch["plane"]
            for tree, points in holdout_clouds.values():
                observed = fit_surface_plane(anchor, tree, points, radius, min_neighbors)
                if observed is None:
                    continue
                delta = parameter_delta(nominal, observed, anchor)
                actual_points = patch_points_for_plane_delta(patch, delta)
                actual = patch_query_residual(initial, patch, actual_points, query_images, camera_data)
                if actual is None:
                    continue
                norm = np.linalg.norm(actual)
                if norm <= 1e-8:
                    continue
                ratios.append(float(np.linalg.norm(transform @ actual) / norm))
                residual_norms.append(float(norm))
                holdout_count += 1
                break
    return {"patch_count": int(patch_count), "count": int(len(ratios)), "holdout_pair_count": holdout_count,
            "attenuation_median": float(np.median(ratios)) if ratios else float("nan"),
            "attenuation_p10": float(np.percentile(ratios, 10)) if ratios else float("nan"),
            "attenuation_p90": float(np.percentile(ratios, 90)) if ratios else float("nan"),
            "attenuation_below_0_99": float(np.mean(np.asarray(ratios) < .99)) if ratios else float("nan"),
            "actual_residual_norm_median": float(np.median(residual_norms)) if residual_norms else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--direction-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--surface-voxel-size", type=float, default=.2)
    parser.add_argument("--uncertainty-radius", type=float, default=1.2)
    parser.add_argument("--plane-radius", type=float, default=.8)
    parser.add_argument("--min-plane-neighbors", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--min-contrast", type=float, default=.03)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.folds < 2 or args.folds > len(train_rows):
        parser.error("folds must be between 2 and the number of train frames")
    fold_ids = np.arange(len(train_rows), dtype=np.int64) % args.folds
    fold_results = []
    root = Path(args.output).with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        fit_rows = [row for index, row in enumerate(train_rows) if fold_ids[index] != fold]
        holdout_rows = [row for index, row in enumerate(train_rows) if fold_ids[index] == fold]
        fold_ref_path = root / ("fold_%d_reference.npz" % fold)
        references = build_reference_observations(
            fit_rows, args.reference_lidar_cache, args.projection_cache, args.map_voxel_size,
            fold_ref_path, 0)
        rows_by_frame = {row["frame_id"]: row for row in fit_rows + holdout_rows}
        reference_context = prepare_patch_context(
            references, rows_by_frame, args.reference_lidar_cache, args.surface_voxel_size)
        fit_cloud = frame_clouds(fit_rows, args.reference_lidar_cache)
        holdout_cloud = frame_clouds(holdout_rows, args.reference_lidar_cache)
        holdout_projection_cloud = projection_frame_clouds(
            holdout_rows, args.reference_lidar_cache, args.projection_cache)
        covariance, covariance_diagnostics = estimate_parameter_covariance(
            references, reference_context, fit_cloud, args.uncertainty_radius,
            args.min_plane_neighbors, args.sample_count, args.seed + fold)
        source_deltas, delta_metadata = holdout_plane_deltas(
            reference_context[0], reference_context[1], holdout_cloud,
            args.uncertainty_radius, args.min_plane_neighbors, args.sample_count,
            args.seed + fold, independent_voxel=.5)
        projection_deltas, projection_metadata = holdout_plane_deltas(
            reference_context[0], reference_context[1], holdout_projection_cloud,
            args.uncertainty_radius, args.min_plane_neighbors, args.sample_count,
            args.seed + fold + 10000, independent_voxel=.5)
        direction_metrics = direction_diagnostic(
            fit_rows, references, rows_by_frame, args.reference_lidar_cache, holdout_projection_cloud,
            reference_context, covariance, args.uncertainty_radius,
            args.min_plane_neighbors, args.patch_size,
            args.min_contrast, args.direction_samples, args.seed + fold)
        fold_results.append({"fold": fold, "fit_frames": [r["frame_id"] for r in fit_rows],
                             "holdout_frames": [r["frame_id"] for r in holdout_rows],
                             "covariance": covariance.tolist(),
                             "covariance_diagnostics": covariance_diagnostics,
                             "plane_metrics_source": plane_metrics(source_deltas, covariance),
                             "plane_metrics_projection": plane_metrics(projection_deltas, covariance),
                             "direction_metrics": direction_metrics,
                             "holdout_examples_source": delta_metadata[:20],
                             "holdout_examples_projection": projection_metadata[:20]})
        print("fold %d/%d fit=%d holdout=%d plane_samples=%d direction_samples=%d" % (
            fold + 1, args.folds, len(fit_rows), len(holdout_rows),
            projection_deltas.shape[0], direction_metrics["count"]), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": {
        "training_only": True,
        "query_gt_used": False,
        "fold_rule": "train frame index modulo folds",
        "parameterization": ["normal_x", "normal_y", "normal_z", "local_plane_offset_m"],
        "uncertainty_radius_m": args.uncertainty_radius,
        "plane_radius_online_m": args.plane_radius,
        "manifest_sha256": digest_json(rows),
    }, "folds": fold_results}, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
