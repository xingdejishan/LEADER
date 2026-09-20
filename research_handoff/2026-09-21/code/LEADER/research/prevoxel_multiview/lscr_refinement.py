"""Deterministic LiDAR-conditioned subpixel correspondence refinement."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from joint_lidar_camera_refinement import frozen_lidar_information, pixel_pose_jacobian
from local_visual_refinement import visible_map_points
from local_visual_refinement_roma import load_match_cache
from oracle_pose_refinement import load_module, pose_from_baseline


def summarize(errors, delta):
    errors, delta = np.asarray(errors, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if not len(errors):
        return {"count": 0, "median_pixel_error": None, "p90_pixel_error": None,
                "mean_du_px": None, "mean_dv_px": None, "lt_2px_fraction": None,
                "lt_5px_fraction": None, "lt_8px_fraction": None}
    return {"count": int(len(errors)), "median_pixel_error": float(np.median(errors)),
            "p90_pixel_error": float(np.quantile(errors, .9)),
            "mean_du_px": float(delta[:, 0].mean()), "mean_dv_px": float(delta[:, 1].mean()),
            "lt_2px_fraction": float((errors < 2.).mean()), "lt_5px_fraction": float((errors < 5.).mean()),
            "lt_8px_fraction": float((errors < 8.).mean())}


def truth_pixels(points, cameras, row, gt):
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    visible = np.zeros(len(points), dtype=bool)
    views = {int(view["camera"]): view for view in row["views"]}
    for camera in np.unique(cameras):
        positions = np.where(cameras == camera)[0]
        view = views[int(camera)]
        from PIL import Image
        image = np.asarray(Image.open(view["image"]).convert("RGB"))
        mask = np.asarray(np.load(view["mask"]))
        projected, indices = visible_map_points(points[positions], gt, view, image, mask)
        pixels[positions] = projected
        visible[positions[indices]] = True
    return pixels, visible


def stable_inverse(matrix, floor=1e-8):
    values, vectors = np.linalg.eigh(.5 * (matrix + matrix.T))
    values = np.maximum(values, floor)
    return (vectors / values[None]) @ vectors.T


def quadratic_subpixel_peak(scores):
    """Fit a score quadratic on a 3x3 integer neighbourhood, in local pixels."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (3, 3) or not np.isfinite(scores).all():
        return np.zeros(2), None
    y, x = np.mgrid[-1:2, -1:2]
    design = np.column_stack((x.ravel() ** 2, y.ravel() ** 2, (x * y).ravel(),
                              x.ravel(), y.ravel(), np.ones(9)))
    coeff, _, _, _ = np.linalg.lstsq(design, scores.ravel(), rcond=None)
    hessian = np.array(((2 * coeff[0], coeff[2]), (coeff[2], 2 * coeff[1])))
    if np.linalg.eigvalsh(hessian).max() >= -1e-8:
        return np.zeros(2), None
    offset = -np.linalg.solve(hessian, coeff[3:5])
    if not np.isfinite(offset).all() or np.max(np.abs(offset)) > 1.:
        return np.zeros(2), None
    return offset, stable_inverse(-hessian)


class RoMaFineFeatures:
    def __init__(self, device, setting, feature_stride):
        import torch
        from romav2 import RoMaV2

        self.torch = torch
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device(device)
        self.model = RoMaV2(RoMaV2.Cfg(setting=setting))
        self.feature_stride = int(feature_stride)
        if self.feature_stride not in (1, 2, 4):
            raise ValueError("feature stride must be one of 1, 2, 4")
        self.cache = {}
        self.multiscale_cache = {}

    def pair(self, reference_path, query_path):
        key = (str(reference_path), str(query_path))
        if key not in self.cache:
            import torch.nn.functional as F

            reference = self.model._load_image(reference_path)
            query = self.model._load_image(query_path)
            height, width = ((self.model.H_hr, self.model.W_hr) if self.model.H_hr is not None else
                             (self.model.H_lr, self.model.W_lr))
            reference_resized = F.interpolate(reference, size=(height, width), mode="bicubic",
                                              align_corners=False, antialias=True)
            query_resized = F.interpolate(query, size=(height, width), mode="bicubic",
                                          align_corners=False, antialias=True)
            with self.torch.inference_mode():
                reference_features = self.model.refiner_features(reference_resized)[self.feature_stride].float().permute(0, 3, 1, 2)
                query_features = self.model.refiner_features(query_resized)[self.feature_stride].float().permute(0, 3, 1, 2)
            self.cache[key] = (reference_features, query_features)
        return self.cache[key]

    def clear(self):
        self.cache.clear()
        self.multiscale_cache.clear()

    def correlation_volume(self, reference_features, query_features, reference_uv, reference_hw, query_hw,
                           center_uv, radius, batch_size=128, step=1, return_valid_mask=False,
                           query_valid_mask=None):
        import torch.nn.functional as F

        reference_features = reference_features.to(self.device)
        query_features = query_features.to(self.device)
        reference_uv = np.asarray(reference_uv, dtype=np.float64)
        center_uv = np.asarray(center_uv, dtype=np.float64)
        if query_valid_mask is not None:
            query_valid_mask = np.asarray(query_valid_mask, dtype=bool)
            if query_valid_mask.shape != tuple(query_hw):
                raise ValueError("query valid mask shape does not match query image")
        offsets_y, offsets_x = np.mgrid[-radius:radius + 1:step, -radius:radius + 1:step]
        offsets = np.column_stack((offsets_x.ravel(), offsets_y.ravel())).astype(np.float64)
        output = np.empty((len(reference_uv), len(offsets)), dtype=np.float32)
        valid = np.empty((len(reference_uv), len(offsets)), dtype=bool)
        for start in range(0, len(reference_uv), batch_size):
            stop = min(start + batch_size, len(reference_uv))
            reference_grid = self._normalized_grid(reference_uv[start:stop], reference_hw, reference_features.dtype)
            candidates = center_uv[start:stop, None, :] + offsets[None]
            valid[start:stop] = ((candidates[..., 0] >= 0) & (candidates[..., 0] < query_hw[1]) &
                                 (candidates[..., 1] >= 0) & (candidates[..., 1] < query_hw[0]))
            if query_valid_mask is not None:
                candidate_x = np.clip(np.floor(candidates[..., 0]).astype(np.int64), 0, query_hw[1] - 1)
                candidate_y = np.clip(np.floor(candidates[..., 1]).astype(np.int64), 0, query_hw[0] - 1)
                valid[start:stop] &= query_valid_mask[candidate_y, candidate_x]
            candidates = candidates.copy()
            candidates[..., 0] = np.clip(candidates[..., 0], 0, query_hw[1] - 1)
            candidates[..., 1] = np.clip(candidates[..., 1], 0, query_hw[0] - 1)
            query_grid = self._normalized_grid(candidates.reshape(-1, 2), query_hw, query_features.dtype)
            reference = F.grid_sample(reference_features, reference_grid[None, :, None], mode="bilinear",
                                      padding_mode="border", align_corners=False)[0, :, :, 0].T
            query = F.grid_sample(query_features, query_grid[None, :, None], mode="bilinear",
                                  padding_mode="border", align_corners=False)[0, :, :, 0].T
            reference = F.normalize(reference, dim=1)
            query = F.normalize(query, dim=1).reshape(stop - start, len(offsets), -1)
            output[start:stop] = (query * reference[:, None]).sum(dim=2, dtype=self.torch.float32).detach().cpu().numpy()
        if return_valid_mask:
            return output, valid
        return output

    def pair_multiscale(self, reference_path, query_path, strides=(1, 2, 4)):
        key = (str(reference_path), str(query_path), tuple(strides))
        if key not in self.multiscale_cache:
            import torch.nn.functional as F

            reference = self.model._load_image(reference_path)
            query = self.model._load_image(query_path)
            height, width = ((self.model.H_hr, self.model.W_hr) if self.model.H_hr is not None else
                             (self.model.H_lr, self.model.W_lr))
            reference_resized = F.interpolate(reference, size=(height, width), mode="bicubic",
                                              align_corners=False, antialias=True)
            query_resized = F.interpolate(query, size=(height, width), mode="bicubic",
                                          align_corners=False, antialias=True)
            with self.torch.inference_mode():
                reference_maps = self.model.refiner_features(reference_resized)
                reference_maps = {stride: reference_maps[stride].permute(0, 3, 1, 2).cpu()
                                  for stride in strides}
                self.torch.cuda.empty_cache()
                query_maps = self.model.refiner_features(query_resized)
                query_maps = {stride: query_maps[stride].permute(0, 3, 1, 2).cpu()
                              for stride in strides}
            self.multiscale_cache[key] = {stride: (reference_maps[stride], query_maps[stride]) for stride in strides}
        return self.multiscale_cache[key]

    def score_surface(self, reference_features, query_features, reference_uv, reference_hw, query_hw,
                      center_uv, radius_xy, lidar_uv, lidar_precision, appearance_weight, geometry_weight):
        import torch.nn.functional as F

        reference_uv = np.asarray(reference_uv, dtype=np.float64)
        center_uv, radius_xy, lidar_uv = (np.asarray(value, dtype=np.float64) for value in
                                          (center_uv, radius_xy, lidar_uv))
        count = len(reference_uv)
        lower = np.ceil(center_uv - radius_xy).astype(np.int64)
        upper = np.floor(center_uv + radius_xy).astype(np.int64)
        lower[:, 0] = np.clip(lower[:, 0], 1, query_hw[1] - 2)
        lower[:, 1] = np.clip(lower[:, 1], 1, query_hw[0] - 2)
        upper[:, 0] = np.clip(upper[:, 0], 1, query_hw[1] - 2)
        upper[:, 1] = np.clip(upper[:, 1], 1, query_hw[0] - 2)
        refined = center_uv.copy()
        covariance = np.full((count, 2, 2), np.nan)
        status = np.full(count, "empty", dtype="U24")
        for index in range(count):
            xs = np.arange(lower[index, 0], upper[index, 0] + 1)
            ys = np.arange(lower[index, 1], upper[index, 1] + 1)
            if not len(xs) or not len(ys):
                continue
            x_grid, y_grid = np.meshgrid(xs, ys)
            candidate = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float64)
            ref_grid = self._normalized_grid(reference_uv[index:index + 1], reference_hw)
            query_grid = self._normalized_grid(candidate, query_hw)
            ref = F.grid_sample(reference_features, ref_grid[None, :, None], mode="bilinear",
                                padding_mode="border", align_corners=False)[0, :, :, 0].T
            query = F.grid_sample(query_features, query_grid[None, :, None], mode="bilinear",
                                  padding_mode="border", align_corners=False)[0, :, :, 0].T
            ref = F.normalize(ref, dim=1)
            query = F.normalize(query, dim=1)
            similarity = (query * ref).sum(dim=1).detach().cpu().numpy()
            delta = candidate - lidar_uv[index]
            d2 = np.einsum("ni,ij,nj->n", delta, lidar_precision[index], delta)
            score = appearance_weight * similarity - geometry_weight * d2
            best = int(np.argmax(score))
            best_uv = candidate[best]
            score_grid = score.reshape(len(ys), len(xs))
            local_x, local_y = best % len(xs), best // len(xs)
            if 0 < local_x < len(xs) - 1 and 0 < local_y < len(ys) - 1:
                local = score_grid[local_y - 1:local_y + 2, local_x - 1:local_x + 2]
                offset, local_covariance = quadratic_subpixel_peak(local)
                refined[index] = best_uv + offset
                if local_covariance is not None:
                    covariance[index] = local_covariance
                    status[index] = "subpixel"
                else:
                    status[index] = "integer_nonconcave"
            else:
                refined[index] = best_uv
                status[index] = "integer_boundary"
        return refined, covariance, status

    def _normalized_grid(self, pixels, image_hw, dtype=None):
        pixels = self.torch.as_tensor(pixels, dtype=self.torch.float32 if dtype is None else dtype, device=self.device)
        height, width = image_hw
        return self.torch.stack((2 * (pixels[:, 0] + .5) / width - 1,
                                 2 * (pixels[:, 1] + .5) / height - 1), dim=-1)


def lidar_pixel_prior(points, cameras, pose, views, covariance, inflation, floor_px, alpha, min_radius, max_radius):
    count = len(points)
    projected = np.full((count, 2), np.nan)
    pixel_covariance = np.full((count, 2, 2), np.nan)
    radii = np.full((count, 2), np.nan)
    for camera in np.unique(cameras):
        positions = np.where(cameras == camera)[0]
        view = next(view for view in views if int(view["camera"]) == int(camera))
        base, jacobian = pixel_pose_jacobian(points[positions], pose, np.asarray(view["camera_to_body"]),
                                             np.loadtxt(view["calibration"]))
        local_covariance = inflation ** 2 * np.einsum("nai,ij,nbj->nab", jacobian, covariance, jacobian)
        local_covariance += floor_px ** 2 * np.eye(2)[None]
        local_covariance = .5 * (local_covariance + np.swapaxes(local_covariance, 1, 2))
        projected[positions], pixel_covariance[positions] = base, local_covariance
        radii[positions] = np.clip(alpha * np.sqrt(np.maximum(np.diagonal(local_covariance, axis1=1, axis2=2), 0.)),
                                    min_radius, max_radius)
    precision = np.asarray([stable_inverse(value) for value in pixel_covariance])
    return projected, pixel_covariance, precision, radii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--match-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--evaluate-split", default="train", choices=("train", "validation", "val"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--feature-stride", type=int, default=4)
    parser.add_argument("--appearance-weight", type=float, default=1.)
    parser.add_argument("--geometry-weight", type=float, default=.05)
    parser.add_argument("--covariance-inflation", type=float, default=4.)
    parser.add_argument("--pixel-floor-px", type=float, default=1.)
    parser.add_argument("--window-sigmas", type=float, default=2.)
    parser.add_argument("--minimum-radius-px", type=float, default=3.)
    parser.add_argument("--maximum-radius-px", type=float, default=12.)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if min(args.appearance_weight, args.covariance_inflation, args.pixel_floor_px, args.window_sigmas,
           args.minimum_radius_px, args.maximum_radius_px) <= 0 or args.geometry_weight < 0:
        parser.error("weights, uncertainty, and window sizes must be positive")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    split = "validation" if args.evaluate_split == "val" else args.evaluate_split
    eval_rows = [row for row in rows if row["split"] in (("val", "validation") if split == "validation" else (split,))]
    eval_rows = [row for row in eval_rows if (Path(args.match_cache_dir) / (row["frame_id"] + ".npz")).exists()]
    if args.frames:
        eval_rows = eval_rows[:args.frames]
    if not eval_rows:
        raise ValueError("no selected rows with match caches")
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("lscr_matcher", REPO / "models" / "sc2pcr.py")
    pool_module = load_module("lscr_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    features = RoMaFineFeatures(args.device, args.roma_setting, args.feature_stride)
    all_before_error, all_before_delta, all_after_error, all_after_delta = [], [], [], []
    records, started = [], time.time()
    for sequence_index, row in enumerate(eval_rows):
        original_index = [candidate["frame_id"] for candidate in rows if candidate["split"] == row["split"]].index(row["frame_id"])
        pose, gt, support, evidence = pose_from_baseline(row, args.lidar_cache, matcher, pool_module.full_pool_refine,
                                                          args.device, args.seed + original_index, return_lidar_evidence=True)
        if evidence is None:
            raise RuntimeError("full-pool implementation did not return final Tukey evidence")
        lidar = frozen_lidar_information(pose, evidence)
        cache = load_match_cache(Path(args.match_cache_dir) / (row["frame_id"] + ".npz"))
        points, before_pixels, reference_pixels, cameras, _, _, _, reference_frames, reference_cameras, _ = cache
        lidar_uv, lidar_covariance, lidar_precision, radii = lidar_pixel_prior(
            points, cameras, pose, row["views"], lidar["covariance"], args.covariance_inflation,
            args.pixel_floor_px, args.window_sigmas, args.minimum_radius_px, args.maximum_radius_px)
        after_pixels = before_pixels.astype(np.float64).copy()
        refined_covariance = np.full((len(points), 2, 2), np.nan)
        status = np.full(len(points), "not_processed", dtype="U24")
        for camera in np.unique(cameras):
            query_view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
            from PIL import Image
            with Image.open(query_view["image"]) as image:
                query_hw = (image.height, image.width)
            for reference_frame, reference_camera in sorted(set(zip(reference_frames[cameras == camera], reference_cameras[cameras == camera]))):
                keep = ((cameras == camera) & (reference_frames == reference_frame) &
                        (reference_cameras == reference_camera))
                reference_view = next(view for view in rows_by_frame[str(reference_frame)]["views"]
                                      if int(view["camera"]) == int(reference_camera))
                with Image.open(reference_view["image"]) as image:
                    reference_hw = (image.height, image.width)
                ref_features, query_features = features.pair(reference_view["image"], query_view["image"])
                refined, covariance, local_status = features.score_surface(
                    ref_features, query_features, reference_pixels[keep], reference_hw, query_hw, before_pixels[keep],
                    radii[keep], lidar_uv[keep], lidar_precision[keep], args.appearance_weight, args.geometry_weight)
                after_pixels[keep], refined_covariance[keep], status[keep] = refined, covariance, local_status
        truth, visible = truth_pixels(points, cameras, row, gt)
        before_delta, after_delta = before_pixels[visible] - truth[visible], after_pixels[visible] - truth[visible]
        before_error, after_error = np.linalg.norm(before_delta, axis=1), np.linalg.norm(after_delta, axis=1)
        all_before_error.append(before_error)
        all_before_delta.append(before_delta)
        all_after_error.append(after_error)
        all_after_delta.append(after_delta)
        records.append({"frame_id": row["frame_id"], "baseline_support": support,
                        "correspondences": int(len(points)), "gt_visible": int(visible.sum()),
                        "before": summarize(before_error, before_delta), "after": summarize(after_error, after_delta),
                        "status": {value: int((status == value).sum()) for value in np.unique(status)},
                        "window_radius_px": {"median_x": float(np.median(radii[:, 0])), "median_y": float(np.median(radii[:, 1])),
                                             "p90_x": float(np.quantile(radii[:, 0], .9)), "p90_y": float(np.quantile(radii[:, 1], .9))},
                        "lidar_residual_scale_m": float(lidar["residual_scale_m"]),
                        "subpixel_covariance_count": int(np.isfinite(refined_covariance).all(axis=(1, 2)).sum())})
        print("LSCR %d/%d %s visible=%d before=%.3f after=%.3f" %
              (sequence_index + 1, len(eval_rows), row["frame_id"], visible.sum(), np.median(before_error), np.median(after_error)), flush=True)
        features.clear()
    before_error, before_delta = np.concatenate(all_before_error), np.concatenate(all_before_delta)
    after_error, after_delta = np.concatenate(all_after_error), np.concatenate(all_after_delta)
    result = {"protocol": {"name": "LSCR V1 deterministic frozen RoMa fine-feature refinement",
                            "input": "cached RoMa correspondences; neither reference retrieval nor pose solver is changed",
                            "appearance": "cosine similarity of frozen RoMa stride-%d refiner features" % args.feature_stride,
                            "geometry": "full-pool final-Tukey Hessian pose covariance projected to pixels",
                            "search": "RoMa pixel-centered covariance-adaptive rectangle; LiDAR Mahalanobis prior",
                            "subpixel": "3x3 quadratic score-surface peak; non-concave or boundary maxima remain integer",
                            "gt_used_for": "evaluation only"},
              "settings": vars(args), "frames": records,
              "overall": {"before": summarize(before_error, before_delta), "after": summarize(after_error, after_delta),
                          "median_error_change_px": float(np.median(after_error) - np.median(before_error)),
                          "p90_error_change_px": float(np.quantile(after_error, .9) - np.quantile(before_error, .9))},
              "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
