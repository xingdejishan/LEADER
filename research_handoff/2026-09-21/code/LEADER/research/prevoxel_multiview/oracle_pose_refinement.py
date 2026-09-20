"""Camera-geometry upper bound for local refinement of frozen LEADER poses."""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scan(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 8:
        raise ValueError("invalid NCLT scan byte count: %s" % path)
    raw = raw.reshape(-1, 8)
    xyz = raw[:, :6].reshape(-1, 3, 2)
    values = xyz[:, :, 0].astype(np.uint16) + (xyz[:, :, 1].astype(np.uint16) << 8)
    points = values.astype(np.float32) * 0.005 - 100.0
    distance = np.linalg.norm(points, axis=1)
    return points[(distance > 1.0) & (distance < 100.0)]


def load_lidar(cache_dir, frame_id):
    with np.load(Path(cache_dir) / (frame_id + ".npz")) as data:
        return {key: np.asarray(data[key]) for key in ("source", "prediction", "GT", "center")}


def pose_error(pose, gt):
    delta = pose[:3, :3].T @ gt[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(pose[:3, 3] - gt[:3, 3])), float(np.degrees(np.arccos(cosine)))


def pose_from_baseline(row, cache_dir, matcher, full_pool, device, seed, return_lidar_evidence=False):
    import torch

    cached = load_lidar(cache_dir, row["frame_id"])
    source = torch.as_tensor(cached["source"], dtype=torch.float32, device=device)
    prediction = torch.as_tensor(cached["prediction"], dtype=torch.float32, device=device)
    keep_count = max(min(50, len(prediction)), int(0.5 * len(prediction)))
    torch.manual_seed(seed)
    keep = prediction[:, 3].topk(keep_count).indices
    initial = matcher.estimator(source[keep][None], prediction[keep, :3][None])[0]
    output = full_pool(initial, source, prediction[:, :3], return_evidence=True) if return_lidar_evidence else full_pool(initial, source, prediction[:, :3])
    refined, support = output[:2]
    pose = refined.detach().cpu().numpy().astype(np.float64)
    pose[:3, 3] += np.asarray(cached["center"], dtype=np.float64)
    if not return_lidar_evidence:
        return pose, np.asarray(cached["GT"], dtype=np.float64), int(support)
    evidence = output[2]
    if evidence is None:
        return pose, np.asarray(cached["GT"], dtype=np.float64), int(support), None
    center = np.asarray(cached["center"], dtype=np.float64)
    lidar_evidence = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in evidence.items() if key != "threshold"}
    lidar_evidence["target"] += center
    lidar_evidence["threshold"] = float(evidence["threshold"])
    return pose, np.asarray(cached["GT"], dtype=np.float64), int(support), lidar_evidence


def transform_points(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def build_reference_map(rows, cache_dir, voxel_size, output):
    if output.exists():
        with np.load(output) as data:
            return np.asarray(data["points"], dtype=np.float64)
    chunks = []
    for index, row in enumerate(row for row in rows if row["split"] == "train"):
        scan = load_scan(row["scan"])
        gt = load_lidar(cache_dir, row["frame_id"])["GT"]
        chunks.append(transform_points(scan, gt))
        print("map frame %d/%d" % (index + 1, sum(r["split"] == "train" for r in rows)), flush=True)
    points = np.concatenate(chunks, axis=0)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    points = points[np.sort(first)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=points.astype(np.float32), voxel_size=np.float32(voxel_size))
    return points.astype(np.float64)


def se3_exp(delta):
    from scipy.spatial.transform import Rotation

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()
    transform[:3, 3] = delta[:3]
    return transform


def project_world(points, body_pose, camera_to_body, K):
    camera_from_world = np.linalg.inv(body_pose @ camera_to_body)
    camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    projected = camera @ K.T
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv, z


def load_image_and_mask(view):
    from PIL import Image

    image = np.asarray(Image.open(view["image"]).convert("RGB"))
    mask = np.asarray(np.load(view["mask"]))
    if image.shape[:2] != mask.shape:
        raise ValueError("image/mask size mismatch for camera %s" % view["camera"])
    return image, mask


def build_correspondences(row, initial, gt, reference_map, grid=(16, 12), per_cell=3,
                          min_depth=0.5, max_depth=80.0, zbuffer_cell=4,
                          zbuffer_kernel=3, tau_base=0.5, tau_slope=0.03,
                          crop_radius=80.0):
    views = sorted(row["views"], key=lambda view: view["camera"])
    if [view["camera"] for view in views] != list(range(6)):
        raise ValueError("manifest must contain cameras 0..5")
    center = initial[:3, 3]
    local = reference_map[np.linalg.norm(reference_map - center[None], axis=1) <= crop_radius]
    all_points, all_pixels, all_cameras, diagnostics = [], [], [], []
    width = height = None
    for view in views:
        image, image_mask = load_image_and_mask(view)
        height, width = image.shape[:2]
        K = np.loadtxt(view["calibration"]).astype(np.float64)
        camera_to_body = np.asarray(view["camera_to_body"], dtype=np.float64)
        uv, depth = project_world(local, gt, camera_to_body, K)
        valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        valid &= np.isfinite(uv).all(axis=1)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        integer_uv = np.rint(uv).astype(np.int64)
        integer_uv[:, 0] = np.clip(integer_uv[:, 0], 0, width - 1)
        integer_uv[:, 1] = np.clip(integer_uv[:, 1], 0, height - 1)
        valid &= image_mask[integer_uv[:, 1], integer_uv[:, 0]] > 0
        valid &= image[integer_uv[:, 1], integer_uv[:, 0]].max(axis=1) > 10
        indices = np.where(valid)[0]
        if not len(indices):
            diagnostics.append({"camera": int(view["camera"]), "projected": 0, "visible": 0, "selected": 0})
            continue
        cell_u = np.clip((uv[indices, 0] // zbuffer_cell).astype(np.int64), 0, (width - 1) // zbuffer_cell)
        cell_v = np.clip((uv[indices, 1] // zbuffer_cell).astype(np.int64), 0, (height - 1) // zbuffer_cell)
        gw, gh = (width + zbuffer_cell - 1) // zbuffer_cell, (height + zbuffer_cell - 1) // zbuffer_cell
        zbuffer = np.full((gh, gw), np.inf, dtype=np.float64)
        np.minimum.at(zbuffer, (cell_v, cell_u), depth[indices])
        half = zbuffer_kernel // 2
        znear = np.full(len(indices), np.inf, dtype=np.float64)
        for dv in range(-half, half + 1):
            for du in range(-half, half + 1):
                uu = np.clip(cell_u + du, 0, gw - 1)
                vv = np.clip(cell_v + dv, 0, gh - 1)
                znear = np.minimum(znear, zbuffer[vv, uu])
        visible = depth[indices] - znear <= np.maximum(tau_base, tau_slope * depth[indices])
        indices = indices[visible]
        if len(indices):
            gu = np.minimum((uv[indices, 0] / width * grid[0]).astype(np.int64), grid[0] - 1)
            gv = np.minimum((uv[indices, 1] / height * grid[1]).astype(np.int64), grid[1] - 1)
            order = np.argsort(depth[indices], kind="stable")
            selected = []
            counts = {}
            for position in order:
                key = (int(gu[position]), int(gv[position]))
                if counts.get(key, 0) >= per_cell:
                    continue
                counts[key] = counts.get(key, 0) + 1
                selected.append(int(indices[position]))
            indices = np.asarray(selected, dtype=np.int64)
        all_points.append(local[indices])
        all_pixels.append(uv[indices])
        all_cameras.append(np.full(len(indices), int(view["camera"]), dtype=np.int64))
        diagnostics.append({"camera": int(view["camera"]), "projected": int(valid.sum()),
                           "visible": int(visible.sum()), "selected": int(len(indices))})
    if not all_points:
        return np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int64), diagnostics
    return np.concatenate(all_points), np.concatenate(all_pixels), np.concatenate(all_cameras), diagnostics


def refine_pose(initial, points, pixels, cameras, views, max_translation, max_rotation,
                max_nfev, f_scale):
    from scipy.optimize import least_squares

    camera_data = {}
    for view in views:
        camera_data[int(view["camera"])] = (np.asarray(view["camera_to_body"], dtype=np.float64),
                                             np.loadtxt(view["calibration"]).astype(np.float64))

    def residual(delta):
        pose = se3_exp(delta) @ initial
        values = []
        for camera in range(6):
            keep = cameras == camera
            if not keep.any():
                continue
            uv, depth = project_world(points[keep], pose, *camera_data[camera])
            r = uv - pixels[keep]
            bad = (~np.isfinite(r).all(axis=1)) | (depth <= 1e-4)
            r[bad] = 1000.0
            values.append(r.reshape(-1))
        return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)

    bounds = np.concatenate([np.full(3, max_translation), np.full(3, max_rotation)])
    result = least_squares(residual, np.zeros(6), bounds=(-bounds, bounds), method="trf",
                           loss="soft_l1", f_scale=f_scale, max_nfev=max_nfev)
    return se3_exp(result.x) @ initial, result, residual(result.x)


def bootstrap_ci(values, seed=2089, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [[float("nan"), float("nan")], [float("nan"), float("nan")]]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5], axis=0).tolist()


def geometry_diagnostics(points, cameras, gt, views):
    if len(points) == 0:
        return {"point_rank": 0, "point_singular_values": [], "bearing_rank": 0,
                "mean_depth_m": float("nan"), "point_spread_m": float("nan")}
    centered = points - points.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    scale = max(float(singular[0]), 1e-12)
    camera_centers = {}
    depths = []
    for view in views:
        camera_to_body = np.asarray(view["camera_to_body"], dtype=np.float64)
        camera_pose = gt @ camera_to_body
        camera_centers[int(view["camera"])] = camera_pose[:3, 3]
    bearings = []
    for point, camera in zip(points, cameras):
        delta = point - camera_centers[int(camera)]
        depth = np.linalg.norm(delta)
        if depth > 1e-9:
            bearings.append(delta / depth)
            depths.append(depth)
    bearing_singular = np.linalg.svd(np.asarray(bearings) - np.mean(bearings, axis=0), compute_uv=False) if bearings else np.zeros(3)
    bearing_scale = max(float(bearing_singular[0]), 1e-12)
    return {
        "point_rank": int((singular > scale * 1e-3).sum()),
        "point_singular_values": singular.tolist(),
        "bearing_rank": int((bearing_singular > bearing_scale * 1e-3).sum()),
        "bearing_singular_values": bearing_singular.tolist(),
        "mean_depth_m": float(np.mean(depths)) if depths else float("nan"),
        "point_spread_m": float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
    }


def metrics(records, prefix):
    values = np.asarray([[record[prefix][0], record[prefix][1]] for record in records], dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean_translation_m": float(values[:, 0].mean()) if len(values) else float("nan"),
        "median_translation_m": float(np.median(values[:, 0])) if len(values) else float("nan"),
        "p90_translation_m": float(np.percentile(values[:, 0], 90)) if len(values) else float("nan"),
        "mean_rotation_deg": float(values[:, 1].mean()) if len(values) else float("nan"),
        "median_rotation_deg": float(np.median(values[:, 1])) if len(values) else float("nan"),
        "p90_rotation_deg": float(np.percentile(values[:, 1], 90)) if len(values) else float("nan"),
        "success_1m_2deg": int(((values[:, 0] < 1.0) & (values[:, 1] < 2.0)).sum()) if len(values) else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-cache", default=None)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--crop-radius", type=float, default=80.0)
    parser.add_argument("--grid-width", type=int, default=16)
    parser.add_argument("--grid-height", type=int, default=12)
    parser.add_argument("--points-per-cell", type=int, default=3)
    parser.add_argument("--max-translation", type=float, default=2.0)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--f-scale-px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen LEADER solver")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] in ("val", "validation", "test")]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
    if args.frames:
        val_rows = val_rows[:args.frames]
    if not train_rows or not val_rows:
        raise ValueError("manifest must contain train and validation rows")
    map_path = Path(args.map_cache) if args.map_cache else Path(args.output).with_name("reference_map.npz")
    reference_map = build_reference_map(rows if not args.train_frames else train_rows,
                                        args.lidar_cache, args.map_voxel_size, map_path)
    import torch
    matcher_module = load_module("oracle_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("oracle_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    records = []
    started = time.time()
    for index, row in enumerate(val_rows):
        lidar_pose, gt, support = pose_from_baseline(row, args.lidar_cache, matcher,
                                                      full_pool_module.full_pool_refine,
                                                      args.device, args.seed + index)
        before = pose_error(lidar_pose, gt)
        points, pixels, cameras, visibility = build_correspondences(
            row, lidar_pose, gt, reference_map, grid=(args.grid_width, args.grid_height),
            per_cell=args.points_per_cell, crop_radius=args.crop_radius)
        if len(points) >= 6 and len(np.unique(cameras)) >= 1:
            oracle, optimizer, residual = refine_pose(
                lidar_pose, points, pixels, cameras, row["views"], args.max_translation,
                math.radians(args.max_rotation_deg), args.max_nfev, args.f_scale_px)
            after = pose_error(oracle, gt)
            solver = {"success": bool(optimizer.success), "status": int(optimizer.status),
                      "nfev": int(optimizer.nfev), "cost": float(optimizer.cost),
                      "optimality": float(optimizer.optimality),
                      "correspondence_rmse_px": float(np.sqrt(np.mean(residual ** 2)))}
        else:
            oracle, after = lidar_pose.copy(), before
            solver = {"success": False, "status": -1, "nfev": 0, "cost": float("nan"),
                      "optimality": float("nan"), "correspondence_rmse_px": float("nan")}
        n_cameras = int(len(np.unique(cameras))) if len(cameras) else 0
        record = {
            "frame_id": row["frame_id"], "before": list(before), "after": list(after),
            "delta": [after[0] - before[0], after[1] - before[1]],
            "leader_success": bool(before[0] < 1.0 and before[1] < 2.0),
            "n_correspondences": int(len(points)), "n_cameras": n_cameras,
            "per_camera_correspondences": [int((cameras == camera).sum()) for camera in range(6)],
            "solver": solver, "baseline_support": support, "visibility": visibility,
            "geometry": geometry_diagnostics(points, cameras, gt, row["views"]),
            "leader_pose": lidar_pose.tolist(), "oracle_pose": oracle.tolist(), "gt_pose": gt.tolist(),
        }
        records.append(record)
        print("validation %d/%d %s cams=%d corr=%d before=(%.3f,%.3f) after=(%.3f,%.3f)" %
              (index + 1, len(val_rows), row["frame_id"], n_cameras, len(points),
               before[0], before[1], after[0], after[1]), flush=True)
    success = [record for record in records if record["leader_success"]]
    paired = np.asarray([record["delta"] for record in records], dtype=np.float64)
    paired_success = np.asarray([record["delta"] for record in success], dtype=np.float64)
    result = {
        "protocol": {"map_source": "train split only", "map_voxel_size_m": args.map_voxel_size,
                     "local_crop_radius_m": args.crop_radius, "grid": [args.grid_width, args.grid_height],
                     "points_per_cell": args.points_per_cell, "optimizer": "scipy least_squares soft_l1",
                     "initial_pose": "frozen LEADER SC2 + full_pool_refine", "max_translation_m": args.max_translation,
                     "max_rotation_deg": args.max_rotation_deg, "correspondence": "GT pose projection + z-buffer + image mask",
                     "visual_learning_or_matching": "none"},
        "reference_map": {"path": str(map_path), "points": int(len(reference_map)),
                          "train_frames": len(train_rows)},
        "validation_frames": len(records), "records": records,
        "metrics": {"all_before": metrics(records, "before"), "all_after": metrics(records, "after"),
                    "leader_success_before": metrics(success, "before"), "leader_success_after": metrics(success, "after"),
                    "one_camera_or_less_after": metrics([r for r in records if r["n_cameras"] <= 1], "after"),
                    "multi_camera_after": metrics([r for r in records if r["n_cameras"] >= 2], "after")},
        "paired": {"all_mean_delta_translation_m_rotation_deg": paired.mean(axis=0).tolist() if len(paired) else [],
                    "all_median_delta_translation_m_rotation_deg": np.median(paired, axis=0).tolist() if len(paired) else [],
                    "all_bootstrap_95ci": bootstrap_ci(paired),
                    "leader_success_mean_delta_translation_m_rotation_deg": paired_success.mean(axis=0).tolist() if len(paired_success) else [],
                    "leader_success_median_delta_translation_m_rotation_deg": np.median(paired_success, axis=0).tolist() if len(paired_success) else [],
                    "leader_success_bootstrap_95ci": bootstrap_ci(paired_success)},
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "map_points": len(reference_map),
                      "frames": len(records), "leader_success": len(success),
                      "all_delta": result["paired"]["all_mean_delta_translation_m_rotation_deg"],
                      "success_delta": result["paired"]["leader_success_mean_delta_translation_m_rotation_deg"]}, indent=2))


if __name__ == "__main__":
    main()
