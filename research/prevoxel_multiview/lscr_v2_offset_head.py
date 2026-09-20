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

from joint_lidar_camera_refinement import frozen_lidar_information
from local_visual_refinement_roma import load_match_cache
from lscr_refinement import RoMaFineFeatures, lidar_pixel_prior, summarize, truth_pixels
from oracle_pose_refinement import load_module, pose_from_baseline


def pixel_geometry_features(pixels, lidar_pixels, covariance, radius):
    standard_deviation = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-8))
    correlation = covariance[:, 0, 1] / np.maximum(standard_deviation[:, 0] * standard_deviation[:, 1], 1e-8)
    return np.column_stack(((pixels - lidar_pixels) / radius,
                            np.log(standard_deviation / radius), correlation))


class OffsetHead:
    def __init__(self, correlation_size, geometry_size, hidden, device):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(correlation_size + geometry_size, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 4),
        ).to(self.device)

    def fit(self, correlation, geometry, target, epochs, batch_size, learning_rate, seed):
        torch = self.torch
        torch.manual_seed(seed)
        inputs = torch.as_tensor(np.concatenate((correlation, geometry), axis=1), dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        for _ in range(epochs):
            order = torch.randperm(len(inputs), device=self.device)
            for start in range(0, len(order), batch_size):
                index = order[start:start + batch_size]
                output = self.model(inputs[index])
                log_sigma = output[:, 2:].clamp(-2.5, 3.)
                squared = ((output[:, :2] - targets[index]) / log_sigma.exp()).square()
                loss = .5 * (squared + 2 * log_sigma).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()

    def predict(self, correlation, geometry, batch_size):
        torch = self.torch
        inputs = torch.as_tensor(np.concatenate((correlation, geometry), axis=1), dtype=torch.float32, device=self.device)
        output = []
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                output.append(self.model(inputs[start:start + batch_size]).cpu().numpy())
        output = np.concatenate(output)
        covariance = np.zeros((len(output), 2, 2), dtype=np.float64)
        covariance[:, 0, 0] = np.exp(2 * np.clip(output[:, 2], -2.5, 3.))
        covariance[:, 1, 1] = np.exp(2 * np.clip(output[:, 3], -2.5, 3.))
        return output[:, :2], covariance


def collect_dataset(rows, rows_by_frame, split, lidar_cache, match_cache_dir, matcher, pool, device, seed, features,
                    feature_radius, covariance_inflation, pixel_floor, window_sigmas, minimum_radius, maximum_radius):
    selected = [row for row in rows if row["split"] in (("val", "validation") if split == "validation" else (split,))]
    selected = [row for row in selected if (Path(match_cache_dir) / (row["frame_id"] + ".npz")).exists()]
    volumes, geometry, baselines, truths, frame_ids = [], [], [], [], []
    for position, row in enumerate(selected):
        original_index = [candidate["frame_id"] for candidate in rows if candidate["split"] == row["split"]].index(row["frame_id"])
        pose, gt, _, evidence = pose_from_baseline(row, lidar_cache, matcher, pool, device, seed + original_index,
                                                    return_lidar_evidence=True)
        if evidence is None:
            raise RuntimeError("full-pool implementation did not return final Tukey evidence")
        lidar = frozen_lidar_information(pose, evidence)
        points, pixels, cameras, _, _, _, reference_frames, reference_cameras, _ = load_match_cache(
            Path(match_cache_dir) / (row["frame_id"] + ".npz"))
        lidar_pixels, covariance, _, radii = lidar_pixel_prior(
            points, cameras, pose, row["views"], lidar["covariance"], covariance_inflation, pixel_floor,
            window_sigmas, minimum_radius, maximum_radius)
        truth, visible = truth_pixels(points, cameras, row, gt)
        local_volume = np.empty((len(points), (2 * feature_radius + 1) ** 2), dtype=np.float32)
        for camera in np.unique(cameras):
            query_view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
            from PIL import Image
            with Image.open(query_view["image"]) as image:
                query_hw = (image.height, image.width)
            pairs = sorted(set(zip(reference_frames[cameras == camera], reference_cameras[cameras == camera])))
            for reference_frame, reference_camera in pairs:
                keep = ((cameras == camera) & (reference_frames == reference_frame) &
                        (reference_cameras == reference_camera))
                reference_view = next(view for view in rows_by_frame[str(reference_frame)]["views"]
                                      if int(view["camera"]) == int(reference_camera))
                with Image.open(reference_view["image"]) as image:
                    reference_hw = (image.height, image.width)
                reference_features, query_features = features.pair(reference_view["image"], query_view["image"])
                local_volume[keep] = features.correlation_volume(reference_features, query_features, pixels[keep],
                                                                 reference_hw, query_hw, pixels[keep], feature_radius)
        keep = visible & np.isfinite(local_volume).all(axis=1) & np.isfinite(lidar_pixels).all(axis=1)
        volumes.append(local_volume[keep])
        geometry.append(pixel_geometry_features(pixels[keep], lidar_pixels[keep], covariance[keep], radii[keep]))
        baselines.append(pixels[keep])
        truths.append(truth[keep])
        frame_ids.append(np.full(int(keep.sum()), row["frame_id"], dtype="U32"))
        print("V2 features %s %d/%d %s visible=%d" % (split, position + 1, len(selected), row["frame_id"], keep.sum()), flush=True)
        features.clear()
    return {"volume": np.concatenate(volumes), "geometry": np.concatenate(geometry), "baseline": np.concatenate(baselines),
            "truth": np.concatenate(truths), "frame_ids": np.concatenate(frame_ids)}


def per_frame_metrics(frame_ids, baseline, prediction, truth):
    records = []
    for frame_id in np.unique(frame_ids):
        keep = frame_ids == frame_id
        before_delta, after_delta = baseline[keep] - truth[keep], prediction[keep] - truth[keep]
        records.append({"frame_id": str(frame_id), "count": int(keep.sum()),
                        "before": summarize(np.linalg.norm(before_delta, axis=1), before_delta),
                        "after": summarize(np.linalg.norm(after_delta, axis=1), after_delta)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--train-match-cache-dir", required=True)
    parser.add_argument("--validation-match-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v1-validation-result", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--feature-stride", type=int, default=4)
    parser.add_argument("--feature-radius", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--covariance-inflation", type=float, default=4.)
    parser.add_argument("--pixel-floor-px", type=float, default=1.)
    parser.add_argument("--window-sigmas", type=float, default=2.)
    parser.add_argument("--minimum-radius-px", type=float, default=3.)
    parser.add_argument("--maximum-radius-px", type=float, default=12.)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("lscr_v2_matcher", REPO / "models" / "sc2pcr.py")
    pool_module = load_module("lscr_v2_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    features = RoMaFineFeatures(args.device, args.roma_setting, args.feature_stride)
    started = time.time()
    train = collect_dataset(rows, rows_by_frame, "train", args.lidar_cache, args.train_match_cache_dir, matcher,
                            pool_module.full_pool_refine, args.device, args.seed, features, args.feature_radius,
                            args.covariance_inflation, args.pixel_floor_px, args.window_sigmas,
                            args.minimum_radius_px, args.maximum_radius_px)
    validation = collect_dataset(rows, rows_by_frame, "validation", args.lidar_cache, args.validation_match_cache_dir,
                                 matcher, pool_module.full_pool_refine, args.device, args.seed, features,
                                 args.feature_radius, args.covariance_inflation, args.pixel_floor_px,
                                 args.window_sigmas, args.minimum_radius_px, args.maximum_radius_px)
    head = OffsetHead(train["volume"].shape[1], train["geometry"].shape[1], args.hidden, args.device)
    head.fit(train["volume"], train["geometry"], train["truth"] - train["baseline"], args.epochs,
             args.batch_size, args.learning_rate, args.seed)
    prediction_offset, predicted_covariance = head.predict(validation["volume"], validation["geometry"], args.batch_size)
    prediction = validation["baseline"] + prediction_offset
    baseline_delta, v2_delta = validation["baseline"] - validation["truth"], prediction - validation["truth"]
    baseline_error, v2_error = np.linalg.norm(baseline_delta, axis=1), np.linalg.norm(v2_delta, axis=1)
    v1 = json.loads(Path(args.v1_validation_result).read_text(encoding="utf-8"))["overall"]["after"]
    import torch
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.model.state_dict(), "settings": vars(args), "input_dimensions":
                {"correlation": int(train["volume"].shape[1]), "geometry": int(train["geometry"].shape[1])}}, checkpoint)
    result = {"protocol": {"name": "LSCR V2 frozen-RoMa offset and uncertainty head",
                            "train": "47 cached train frames; GT supervises only the offset head",
                            "validation": "separate cached validation frames; no parameter selection on validation",
                            "input": "RoMa local correlation volume, RoMa-to-LiDAR pixel delta, and projected LiDAR covariance",
                            "output": "subpixel offset and diagonal pixel covariance"}, "settings": vars(args),
              "train_samples": int(len(train["truth"])), "validation_samples": int(len(validation["truth"])),
              "comparison": {"baseline": summarize(baseline_error, baseline_delta), "v1": v1,
                             "v2": summarize(v2_error, v2_delta),
                             "v2_minus_baseline_median_px": float(np.median(v2_error) - np.median(baseline_error)),
                             "v2_minus_v1_median_px": float(np.median(v2_error) - float(v1["median_pixel_error"]))},
              "validation_frames": per_frame_metrics(validation["frame_ids"], validation["baseline"], prediction, validation["truth"]),
              "predicted_uncertainty": {"median_sigma_x_px": float(np.median(np.sqrt(predicted_covariance[:, 0, 0]))),
                                         "median_sigma_y_px": float(np.median(np.sqrt(predicted_covariance[:, 1, 1])))},
              "checkpoint": str(checkpoint), "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
