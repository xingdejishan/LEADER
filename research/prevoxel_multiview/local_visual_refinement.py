"""LEADER-guided local visual correspondence generation and pose refinement."""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oracle_pose_refinement import (
    bootstrap_ci,
    geometry_diagnostics,
    load_lidar,
    load_module,
    metrics,
    pose_error,
    pose_from_baseline,
    project_world,
    refine_pose,
)


def normalize(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def descriptor_row(path, point_count):
    with np.load(path) as data:
        descriptors = np.asarray(data["image"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=bool)
    if descriptors.shape[:2] != mask.shape or descriptors.shape[0] != point_count:
        raise ValueError("feature/source shape mismatch: %s" % path)
    return descriptors, mask


def build_visual_map(rows, lidar_cache, feature_cache, projection_cache, voxel_size, output, max_history):
    if output.exists():
        with np.load(output) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    point_ids = {}
    points = []
    observations = defaultdict(list)
    train_rows = [row for row in rows if row["split"] == "train"]
    for index, row in enumerate(train_rows):
        cached = load_lidar(lidar_cache, row["frame_id"])
        descriptors, valid = descriptor_row(Path(feature_cache) / (row["frame_id"] + ".npz"), len(cached["source"]))
        with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as mapping:
            projection_xyz = np.asarray(mapping["projection_xyz"], dtype=np.float32)
            localization_xyz = np.asarray(mapping["localization_xyz"], dtype=np.float32)
        if not np.array_equal(localization_xyz, cached["source"]):
            raise ValueError("projection/localization order mismatch: %s" % row["frame_id"])
        world = projection_xyz @ cached["GT"][:3, :3].T + cached["GT"][:3, 3]
        keys = np.floor(world / voxel_size).astype(np.int64)
        for point_index, key_array in enumerate(keys):
            key = tuple(int(value) for value in key_array)
            map_index = point_ids.get(key)
            if map_index is None:
                map_index = len(points)
                point_ids[key] = map_index
                points.append(world[point_index])
            for camera in range(6):
                if not valid[point_index, camera]:
                    continue
                key_obs = (map_index, camera)
                if len(observations[key_obs]) < max_history:
                    observations[key_obs].append(descriptors[point_index, camera])
        print("visual map frame %d/%d" % (index + 1, len(train_rows)), flush=True)
    map_ids, camera_ids, values, counts = [], [], [], []
    for (map_index, camera), history in observations.items():
        for descriptor in history:
            values.append(normalize(np.asarray(descriptor, dtype=np.float32)[None])[0])
            map_ids.append(map_index)
            camera_ids.append(camera)
            counts.append(len(history))
    result = {
        "points": np.asarray(points, dtype=np.float32),
        "descriptors": np.asarray(values, dtype=np.float32),
        "map_ids": np.asarray(map_ids, dtype=np.int32),
        "camera_ids": np.asarray(camera_ids, dtype=np.int8),
        "history_counts": np.asarray(counts, dtype=np.int16),
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


class DenseDescriptorExtractor:
    def __init__(self, device, weights, pca_weights):
        import torch
        import torch.nn.functional as F
        from kornia.feature.dedode.dedode_models import get_descriptor

        self.torch = torch
        self.F = F
        self.device = torch.device(device)
        self.model = get_descriptor("B").to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        pca = torch.load(pca_weights, map_location="cpu", weights_only=True)
        self.pca_weight = pca["weight"].float().to(self.device)
        self.pca_bias = pca["bias"].float().to(self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @property
    def torch_dtype(self):
        return self.torch.float32

    def image(self, frame_id, camera, path):
        from PIL import Image

        with self.torch.no_grad():
            image = Image.open(path).convert("RGB")
            width, height = image.size
            nh = int(np.ceil(height * 480 / min(height, width) / 8) * 8)
            nw = int(np.ceil(width * 480 / min(height, width) / 8) * 8)
            pixels = self.torch.from_numpy(np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR)).copy())
            pixels = pixels.to(self.device).permute(2, 0, 1)[None].float() / 255.0
            with self.torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                dense = self.model((pixels - self.mean) / self.std)
            dense = self.F.conv2d(dense.float(), self.pca_weight, self.pca_bias)
        return dense.float(), (height, width)


class CachedDenseDescriptorExtractor:
    def __init__(self, device, cache_dir):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        self.cache = {}

    def image(self, frame_id, camera, path):
        if frame_id not in self.cache:
            with np.load(self.cache_dir / (frame_id + ".npz")) as data:
                self.cache[frame_id] = [np.asarray(data["cam%d" % index], dtype=np.float32)
                                        for index in range(6)]
        dense = self.torch.from_numpy(self.cache[frame_id][camera]).to(self.device)
        from PIL import Image
        image = Image.open(path)
        return dense, image.size[::-1]


def sample_dense(feature_map, uv, image_hw):
    import torch

    height, width = image_hw
    grid = (uv + .5) / uv.new_tensor([width, height]) * 2 - 1
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    return torch.nn.functional.grid_sample(feature_map, grid[None, None], align_corners=False,
                                           mode="bilinear", padding_mode="zeros")[0, :, 0].T


def visible_map_points(points, pose, view, image, image_mask, zbuffer_cell=4,
                       zbuffer_kernel=3, min_depth=.5, max_depth=80.,
                       tau_base=.5, tau_slope=.03):
    K = np.loadtxt(view["calibration"]).astype(np.float64)
    extrinsic = np.asarray(view["camera_to_body"], dtype=np.float64)
    uv, depth = project_world(points, pose, extrinsic, K)
    height, width = image.shape[:2]
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    valid &= np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    integer = np.rint(uv).astype(np.int64)
    integer[:, 0] = np.clip(integer[:, 0], 0, width - 1)
    integer[:, 1] = np.clip(integer[:, 1], 0, height - 1)
    valid &= image_mask[integer[:, 1], integer[:, 0]] > 0
    valid &= image[integer[:, 1], integer[:, 0]].max(axis=1) > 10
    indices = np.where(valid)[0]
    if not len(indices):
        return uv, np.empty(0, dtype=np.int64)
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
    return uv, indices[visible]


def query_matches(row, initial, visual_map, extractor, crop_radius, search_radius,
                  search_step, min_cosine, grid_cell, max_per_camera):
    points = visual_map["points"].astype(np.float64)
    local_mask = np.linalg.norm(points - initial[:3, 3][None], axis=1) <= crop_radius
    local_ids = np.where(local_mask)[0]
    local_points = points[local_ids]
    map_ids = visual_map["map_ids"].astype(np.int64)
    camera_ids = visual_map["camera_ids"].astype(np.int64)
    descriptors = visual_map["descriptors"].astype(np.float32)
    all_points, all_pixels, all_cameras, all_scores = [], [], [], []
    diagnostics = []
    offsets = np.asarray([(du, dv) for dv in np.arange(-search_radius, search_radius + 1, search_step)
                          for du in np.arange(-search_radius, search_radius + 1, search_step)], dtype=np.float32)
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        from PIL import Image

        image = np.asarray(Image.open(view["image"]).convert("RGB"))
        image_mask = np.asarray(np.load(view["mask"]))
        dense, image_hw = extractor.image(row["frame_id"], int(view["camera"]), view["image"])
        uv, visible_local = visible_map_points(local_points, initial, view, image, image_mask)
        if not len(visible_local):
            diagnostics.append({"camera": int(view["camera"]), "visible_map_points": 0, "accepted": 0})
            continue
        visible_global = local_ids[visible_local]
        record_rows = np.where((camera_ids == int(view["camera"])) & np.isin(map_ids, visible_global))[0]
        if not len(record_rows):
            diagnostics.append({"camera": int(view["camera"]), "visible_map_points": int(len(visible_global)), "accepted": 0})
            continue
        positions = {int(map_index): pos for pos, map_index in enumerate(visible_global)}
        base_uv = np.asarray([uv[visible_local][positions[int(map_index)]] for map_index in map_ids[record_rows]], dtype=np.float32)
        query_uv = base_uv[:, None, :] + offsets[None]
        flat_uv = query_uv.reshape(-1, 2)
        import torch

        sampled = []
        chunk = 2048
        for start in range(0, len(flat_uv), chunk):
            sampled.append(sample_dense(dense, torch.from_numpy(flat_uv[start:start + chunk]).to(extractor.device), image_hw).detach().cpu().numpy())
        sampled = normalize(np.concatenate(sampled, axis=0)).reshape(len(record_rows), len(offsets), -1)
        map_desc = normalize(descriptors[record_rows])
        scores = np.einsum("nkd,nd->nk", sampled, map_desc)
        best_offset = scores.argmax(axis=1)
        best_score = scores[np.arange(len(record_rows)), best_offset]
        keep = best_score >= min_cosine
        best_uv = query_uv[np.arange(len(record_rows)), best_offset]
        if keep.any():
            record_rows = record_rows[keep]
            best_score = best_score[keep]
            best_uv = best_uv[keep]
            cells = np.floor(best_uv / grid_cell).astype(np.int64)
            order = np.argsort(-best_score, kind="stable")
            occupied = set()
            selected = []
            for position in order:
                key = (int(cells[position, 0]), int(cells[position, 1]))
                if key in occupied:
                    continue
                occupied.add(key)
                selected.append(position)
                if len(selected) >= max_per_camera:
                    break
            record_rows = record_rows[selected]
            best_score = best_score[selected]
            best_uv = best_uv[selected]
            all_points.append(points[map_ids[record_rows]])
            all_pixels.append(best_uv)
            all_cameras.append(np.full(len(record_rows), int(view["camera"]), dtype=np.int64))
            all_scores.append(best_score)
        diagnostics.append({"camera": int(view["camera"]), "visible_map_points": int(len(visible_global)),
                            "descriptor_candidates": int(len(record_rows)), "accepted": int(keep.sum()),
                            "score_mean": float(best_score.mean()) if keep.any() else float("nan")})
    if not all_points:
        return np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int64), np.empty(0), diagnostics
    return np.concatenate(all_points), np.concatenate(all_pixels), np.concatenate(all_cameras), np.concatenate(all_scores), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--dedode-weights", default=None)
    parser.add_argument("--pca-weights", default=None)
    parser.add_argument("--dense-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--crop-radius", type=float, default=80.0)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--search-step", type=int, default=4)
    parser.add_argument("--min-cosine", type=float, default=0.65)
    parser.add_argument("--grid-cell", type=int, default=4)
    parser.add_argument("--max-per-camera", type=int, default=300)
    parser.add_argument("--max-translation", type=float, default=2.0)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--f-scale-px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for descriptor extraction and LEADER solver")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] in ("val", "validation", "test")]
    if args.train_frames:
        train_rows = train_rows[:args.train_frames]
    if args.frames:
        val_rows = val_rows[:args.frames]
    map_cache = Path(args.map_cache)
    visual_map = build_visual_map(rows if not args.train_frames else train_rows, args.lidar_cache,
                                  args.feature_cache, args.projection_cache, args.map_voxel_size,
                                  map_cache, args.max_history)
    import torch
    matcher_module = load_module("local_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("local_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    if args.dense_cache:
        extractor = CachedDenseDescriptorExtractor(args.device, args.dense_cache)
    else:
        if not args.dedode_weights or not args.pca_weights:
            raise ValueError("--dedode-weights and --pca-weights are required without --dense-cache")
        extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    records = []
    started = time.time()
    for index, row in enumerate(val_rows):
        initial, gt, support = pose_from_baseline(row, args.lidar_cache, matcher,
                                                  full_pool_module.full_pool_refine,
                                                  args.device, args.seed + index)
        before = pose_error(initial, gt)
        points, pixels, cameras, scores, matching = query_matches(
            row, initial, visual_map, extractor, args.crop_radius, args.search_radius,
            args.search_step, args.min_cosine, args.grid_cell, args.max_per_camera)
        if len(points) >= 6 and len(np.unique(cameras)):
            refined, optimizer, residual = refine_pose(
                initial, points, pixels, cameras, row["views"], args.max_translation,
                math.radians(args.max_rotation_deg), args.max_nfev, args.f_scale_px)
            after = pose_error(refined, gt)
            solver = {"success": bool(optimizer.success), "status": int(optimizer.status),
                      "nfev": int(optimizer.nfev), "cost": float(optimizer.cost),
                      "correspondence_rmse_px": float(np.sqrt(np.mean(residual ** 2)))}
        else:
            refined = initial.copy()
            after = before
            solver = {"success": False, "status": -1, "nfev": 0, "cost": float("nan"),
                      "correspondence_rmse_px": float("nan")}
        record = {
            "frame_id": row["frame_id"], "before": list(before), "after": list(after),
            "delta": [after[0] - before[0], after[1] - before[1]],
            "leader_success": bool(before[0] < 1.0 and before[1] < 2.0),
            "n_correspondences": int(len(points)), "n_cameras": int(len(np.unique(cameras))) if len(cameras) else 0,
            "score_mean": float(scores.mean()) if len(scores) else float("nan"),
            "score_p10": float(np.percentile(scores, 10)) if len(scores) else float("nan"),
            "per_camera_correspondences": [int((cameras == camera).sum()) for camera in range(6)],
            "matching": matching, "solver": solver, "baseline_support": support,
            "geometry": geometry_diagnostics(points, cameras, initial, row["views"]),
            "leader_pose": initial.tolist(), "refined_pose": refined.tolist(), "gt_pose": gt.tolist(),
        }
        records.append(record)
        print("validation %d/%d %s cams=%d corr=%d score=%.3f before=(%.3f,%.3f) after=(%.3f,%.3f)" %
              (index + 1, len(val_rows), row["frame_id"], record["n_cameras"], len(points),
               record["score_mean"], before[0], before[1], after[0], after[1]), flush=True)
    success = [record for record in records if record["leader_success"]]
    paired = np.asarray([record["delta"] for record in records], dtype=np.float64)
    paired_success = np.asarray([record["delta"] for record in success], dtype=np.float64)
    result = {
        "protocol": {"map_source": "train split only", "map_feature_cache": str(args.feature_cache),
                     "map_voxel_size_m": args.map_voxel_size, "local_crop_radius_m": args.crop_radius,
                     "query": "dense DeDoDe-B + PCA128, local descriptor search around T_L projection",
                     "matching": {"search_radius_px": args.search_radius, "search_step_px": args.search_step,
                                  "min_cosine": args.min_cosine, "grid_cell_px": args.grid_cell,
                                  "max_per_camera": args.max_per_camera},
                     "visibility": "T_L projection + image mask + black border + local z-buffer",
                     "optimizer": "same bounded robust LM as oracle", "gt_in_correspondence_path": False},
        "visual_map": {"path": str(map_cache), "points": int(len(visual_map["points"])),
                       "descriptor_observations": int(len(visual_map["descriptors"])), "train_frames": len(train_rows)},
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
    print(json.dumps({"output": str(output), "map_points": len(visual_map["points"]),
                      "descriptor_observations": len(visual_map["descriptors"]), "frames": len(records),
                      "all_delta": result["paired"]["all_mean_delta_translation_m_rotation_deg"],
                      "success_delta": result["paired"]["leader_success_mean_delta_translation_m_rotation_deg"]}, indent=2))


if __name__ == "__main__":
    main()
