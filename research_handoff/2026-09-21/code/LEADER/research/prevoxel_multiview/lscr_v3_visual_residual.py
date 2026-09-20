"""LSCR V3: frozen geometry-only V2 centres plus visual residual classification."""
import argparse
import hashlib
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
from lscr_v2_offset_head import OffsetHead, collect_dataset
from oracle_pose_refinement import load_module, pose_from_baseline


RADIUS = 6
GRID = 2 * RADIUS + 1
NONE_CLASS = GRID * GRID


def ordered_unique(values):
    return list(dict.fromkeys(str(value) for value in values))


def by_frame(frame_ids, values):
    return {frame_id: values[frame_ids == frame_id] for frame_id in ordered_unique(frame_ids)}


def digest_array(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def digest_state_dict(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(digest_array(value).encode())
    return digest.hexdigest()


def match_cache_digest(directory, frame_ids):
    digest = hashlib.sha256()
    for frame_id in ordered_unique(frame_ids):
        path = Path(directory) / (frame_id + ".npz")
        stat = path.stat()
        digest.update((frame_id + ":%d:%d" % (stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def visual_cache_metadata(split, args, v2_head, centres, v2_data, match_cache_dir):
    return {"version": "lscr_v3_visual_cache_v2", "split": split,
            "v2_state_sha256": digest_state_dict(v2_head.model.state_dict()),
            "centre_sha256": digest_array(centres),
            "sample_frame_ids_sha256": digest_array(v2_data["frame_ids"]),
            "sample_baseline_sha256": digest_array(v2_data["baseline"]),
            "match_cache_sha256": match_cache_digest(match_cache_dir, v2_data["frame_ids"]),
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "feature": {"roma_setting": args.roma_setting, "strides": [1, 2, 4], "radius_px": RADIUS,
                        "candidate_validity": "image_bounds_and_query_camera_mask_floor_pixel_v1"},
            "v2": {"oof_folds": args.v2_oof_folds, "hidden": args.v2_hidden, "epochs": args.v2_epochs,
                   "learning_rate": args.v2_learning_rate, "seed": args.seed}}


def geometry_v2_centres(train, validation, args):
    target = train["truth"] - train["baseline"]
    zero_train = np.zeros_like(train["volume"])
    zero_validation = np.zeros_like(validation["volume"])
    frame_order = ordered_unique(train["frame_ids"])
    folds = np.array_split(np.asarray(frame_order), min(args.v2_oof_folds, len(frame_order)))
    oof = np.full_like(train["baseline"], np.nan, dtype=np.float64)
    for fold_index, held_frames in enumerate(folds):
        held = np.isin(train["frame_ids"], held_frames)
        head = OffsetHead(train["volume"].shape[1], train["geometry"].shape[1], args.v2_hidden,
                          args.device, seed=args.seed + fold_index)
        head.fit(zero_train[~held], train["geometry"][~held], target[~held], args.v2_epochs,
                 args.batch_size, args.v2_learning_rate, args.seed + fold_index)
        offset, _ = head.predict(zero_train[held], train["geometry"][held], args.batch_size)
        oof[held] = train["baseline"][held] + offset
        print("V3 V2 OOF fold %d/%d frames=%d" % (fold_index + 1, len(folds), len(held_frames)), flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete V2 out-of-fold predictions")
    head = OffsetHead(train["volume"].shape[1], train["geometry"].shape[1], args.v2_hidden,
                      args.device, seed=args.seed + 100)
    head.fit(zero_train, train["geometry"], target, args.v2_epochs, args.batch_size,
             args.v2_learning_rate, args.seed + 100)
    validation_offset, _ = head.predict(zero_validation, validation["geometry"], args.batch_size)
    return oof, validation["baseline"] + validation_offset, head


def collect_visual_dataset(rows, rows_by_frame, split, lidar_cache, match_cache_dir, matcher, pool, device, seed,
                           features, centres, v2_baselines, covariance_inflation, pixel_floor, window_sigmas,
                           minimum_radius, maximum_radius):
    selected = [row for row in rows if row["split"] in (("val", "validation") if split == "validation" else (split,))]
    selected = [row for row in selected if (Path(match_cache_dir) / (row["frame_id"] + ".npz")).exists()]
    maps, masks, baselines, truths, frame_ids = [], [], [], [], []
    for position, row in enumerate(selected):
        original_index = [candidate["frame_id"] for candidate in rows if candidate["split"] == row["split"]].index(row["frame_id"])
        pose, gt, _, evidence = pose_from_baseline(row, lidar_cache, matcher, pool, device, seed + original_index,
                                                    return_lidar_evidence=True)
        if evidence is None:
            raise RuntimeError("full-pool implementation did not return final Tukey evidence")
        lidar = frozen_lidar_information(pose, evidence)
        points, pixels, reference_pixels, cameras, _, _, _, reference_frames, reference_cameras, _ = load_match_cache(
            Path(match_cache_dir) / (row["frame_id"] + ".npz"))
        lidar_pixels, _, _, _ = lidar_pixel_prior(points, cameras, pose, row["views"], lidar["covariance"],
                                                   covariance_inflation, pixel_floor, window_sigmas,
                                                   minimum_radius, maximum_radius)
        truth, visible = truth_pixels(points, cameras, row, gt)
        expected_baseline = v2_baselines[row["frame_id"]]
        if len(expected_baseline) != int(visible.sum()) or not np.allclose(expected_baseline, pixels[visible]):
            raise RuntimeError("V2/V3 cache alignment failed for frame %s" % row["frame_id"])
        centre = centres[row["frame_id"]]
        local_maps = np.zeros((int(visible.sum()), 3, GRID, GRID), dtype=np.float32)
        local_mask = np.zeros((int(visible.sum()), GRID, GRID), dtype=bool)
        local_index = np.full(len(points), -1, dtype=np.int64)
        local_index[visible] = np.arange(int(visible.sum()))
        for camera in np.unique(cameras):
            query_view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
            from PIL import Image
            with Image.open(query_view["image"]) as image:
                query_hw = (image.height, image.width)
            query_valid_mask = np.asarray(np.load(query_view["mask"]), dtype=bool)
            if query_valid_mask.shape != query_hw:
                raise ValueError("query mask/image shape mismatch: %s" % query_view["image"])
            pairs = sorted(set(zip(reference_frames[cameras == camera], reference_cameras[cameras == camera])))
            for reference_frame, reference_camera in pairs:
                keep = ((cameras == camera) & (reference_frames == reference_frame) &
                        (reference_cameras == reference_camera) & visible)
                if not keep.any():
                    continue
                reference_view = next(view for view in rows_by_frame[str(reference_frame)]["views"]
                                      if int(view["camera"]) == int(reference_camera))
                with Image.open(reference_view["image"]) as image:
                    reference_hw = (image.height, image.width)
                destination = local_index[keep]
                multiscale = features.pair_multiscale(reference_view["image"], query_view["image"])
                for channel, stride in enumerate((1, 2, 4)):
                    reference_features, query_features = multiscale[stride]
                    volume, valid = features.correlation_volume(reference_features, query_features,
                                                                 reference_pixels[keep], reference_hw, query_hw,
                                                                 centre[destination], RADIUS,
                                                                 return_valid_mask=True,
                                                                 query_valid_mask=query_valid_mask)
                    local_maps[destination, channel] = volume.reshape(-1, GRID, GRID)
                    if channel == 0:
                        local_mask[destination] = valid.reshape(-1, GRID, GRID)
        if not np.isfinite(local_maps).all() or not np.isfinite(lidar_pixels[visible]).all():
            raise RuntimeError("non-finite V3 feature input in frame %s" % row["frame_id"])
        maps.append(local_maps)
        masks.append(local_mask)
        baselines.append(centre)
        truths.append(truth[visible])
        frame_ids.append(np.full(int(visible.sum()), row["frame_id"], dtype="U32"))
        print("V3 features %s %d/%d %s visible=%d" %
              (split, position + 1, len(selected), row["frame_id"], visible.sum()), flush=True)
        features.clear()
    return {"maps": np.concatenate(maps), "mask": np.concatenate(masks), "baseline": np.concatenate(baselines),
            "truth": np.concatenate(truths), "frame_ids": np.concatenate(frame_ids)}


def load_or_collect_visual_dataset(cache_path, metadata, collect):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if "metadata" not in cached:
                raise ValueError("V3 visual cache has no metadata; refuse reuse: %s" % cache_path)
            cached_metadata = json.loads(str(cached["metadata"].item()))
            if cached_metadata != metadata:
                raise ValueError("V3 visual cache metadata differs; refuse reuse: %s" % cache_path)
            return {key: cached[key] for key in ("maps", "mask", "baseline", "truth", "frame_ids")}
    dataset = collect()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **dataset, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(cache_path)
    return dataset


def targets(dataset):
    residual = dataset["truth"] - dataset["baseline"]
    integer = np.rint(residual).astype(np.int64)
    inside = ((np.abs(integer) <= RADIUS).all(axis=1) &
              (np.max(np.abs(residual - integer), axis=1) <= .5 + 1e-6))
    indices = (integer[:, 1] + RADIUS) * GRID + integer[:, 0] + RADIUS
    valid_flat = dataset["mask"].reshape(len(indices), -1)
    inside &= valid_flat[np.arange(len(indices)), np.clip(indices, 0, NONE_CLASS - 1)]
    label = np.where(inside, indices, NONE_CLASS).astype(np.int64)
    return label, residual - integer


def target_local_offsets(offsets, cells):
    local_maps = offsets.permute(0, 2, 3, 1).reshape(-1, NONE_CLASS, 2)
    return local_maps[cells.new_tensor(range(len(cells))), cells]


class VisualResidualHead:
    def __init__(self, device, seed):
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.SiLU(),
        ).to(self.device)
        self.location = torch.nn.Conv2d(32, 1, 1).to(self.device)
        self.offset = torch.nn.Conv2d(32, 2, 1).to(self.device)
        self.none = torch.nn.Conv2d(32, 1, 1).to(self.device)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.location.parameters()) + list(self.offset.parameters()) + list(self.none.parameters())

    def forward(self, maps, valid):
        torch = self.torch
        feature = self.backbone(torch.cat((maps, valid[:, None].float()), dim=1))
        location = self.location(feature).flatten(1).masked_fill(~valid.flatten(1), -1e4)
        none = self.none(feature).mean(dim=(2, 3))
        return torch.cat((location, none), dim=1), self.offset(feature), feature

    def fit(self, dataset, train_mask, epochs, batch_size, learning_rate, seed, channel_mask):
        torch = self.torch
        labels, fractions = targets(dataset)
        maps = np.array(dataset["maps"], copy=True)
        maps[:, ~np.asarray(channel_mask, dtype=bool)] = 0.
        input_maps = torch.as_tensor(maps, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(dataset["mask"], dtype=torch.bool, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        fractions = torch.as_tensor(fractions, dtype=torch.float32, device=self.device)
        frame_ids = dataset["frame_ids"]
        available = {frame: np.flatnonzero(train_mask & (frame_ids == frame)) for frame in ordered_unique(frame_ids)}
        available = {frame: value for frame, value in available.items() if len(value)}
        if not available:
            raise ValueError("no V3 coordinate-training samples")
        rng = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        inside_fraction = float((labels != NONE_CLASS).float().mean())
        class_weight = torch.ones(NONE_CLASS + 1, dtype=torch.float32, device=self.device)
        class_weight[NONE_CLASS] = inside_fraction / (GRID * GRID * max(1. - inside_fraction, 1e-4))
        for _ in range(epochs):
            sampled = np.concatenate([rng.choice(value, size=max(len(value) for value in available.values()), replace=True)
                                      for value in available.values()])
            sampled = rng.permutation(sampled)
            for start in range(0, len(sampled), batch_size):
                index = torch.as_tensor(sampled[start:start + batch_size], dtype=torch.long, device=self.device)
                logits, offsets, _ = self.forward(input_maps[index], valid[index])
                position_loss = torch.nn.functional.cross_entropy(logits, labels[index], weight=class_weight)
                inside = labels[index] != NONE_CLASS
                if inside.any():
                    cells = labels[index][inside]
                    local = target_local_offsets(offsets[inside], cells)
                    offset_loss = torch.nn.functional.smooth_l1_loss(.5 * torch.tanh(local), fractions[index][inside])
                else:
                    offset_loss = torch.zeros((), device=self.device)
                loss = position_loss + 2. * offset_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
                optimizer.step()

    def predict(self, dataset, batch_size, channel_mask):
        torch = self.torch
        maps = np.array(dataset["maps"], copy=True)
        maps[:, ~np.asarray(channel_mask, dtype=bool)] = 0.
        input_maps = torch.as_tensor(maps, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(dataset["mask"], dtype=torch.bool, device=self.device)
        prediction, selected = [], []
        with torch.inference_mode():
            for start in range(0, len(maps), batch_size):
                logits, offsets, _ = self.forward(input_maps[start:start + batch_size], valid[start:start + batch_size])
                cells = logits.argmax(dim=1).cpu().numpy()
                offset = offsets.cpu().numpy()
                current = dataset["baseline"][start:start + len(cells)].copy()
                match = cells != NONE_CLASS
                selected_offset = np.zeros((len(cells), 2), dtype=np.float64)
                if match.any():
                    y, x = cells[match] // GRID, cells[match] % GRID
                    selected_offset[match, 0] = x - RADIUS + .5 * np.tanh(offset[match, 0, y, x])
                    selected_offset[match, 1] = y - RADIUS + .5 * np.tanh(offset[match, 1, y, x])
                    current[match] += selected_offset[match]
                prediction.append(current)
                selected.append(cells)
        return np.concatenate(prediction), np.concatenate(selected)


def metrics(baseline, prediction, truth):
    delta = prediction - truth
    error = np.linalg.norm(delta, axis=1)
    result = summarize(error, delta)
    result["lt_1px_fraction"] = float((error < 1.).mean())
    before = np.linalg.norm(baseline - truth, axis=1)
    result["previous_lt_2px_ruined_fraction"] = float(((before < 2.) & (error >= 2.)).mean())
    result["previous_2_to_5px_to_lt_2px_fraction"] = float(((before >= 2.) & (before < 5.) & (error < 2.)).mean())
    return result


def per_frame(dataset, prediction):
    return [{"frame_id": frame, "count": int((dataset["frame_ids"] == frame).sum()),
             "geometry_v2": metrics(dataset["baseline"][dataset["frame_ids"] == frame],
                                    dataset["baseline"][dataset["frame_ids"] == frame],
                                    dataset["truth"][dataset["frame_ids"] == frame]),
             "v3": metrics(dataset["baseline"][dataset["frame_ids"] == frame],
                           prediction[dataset["frame_ids"] == frame], dataset["truth"][dataset["frame_ids"] == frame])}
            for frame in ordered_unique(dataset["frame_ids"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--train-match-cache-dir", required=True)
    parser.add_argument("--validation-match-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-visual-cache", default="")
    parser.add_argument("--validation-visual-cache", default="")
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--v2-oof-folds", type=int, default=5)
    parser.add_argument("--v2-hidden", type=int, default=128)
    parser.add_argument("--v2-epochs", type=int, default=40)
    parser.add_argument("--v2-learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-frame-fraction", type=float, default=.2)
    parser.add_argument("--covariance-inflation", type=float, default=4.)
    parser.add_argument("--pixel-floor-px", type=float, default=1.)
    parser.add_argument("--window-sigmas", type=float, default=2.)
    parser.add_argument("--minimum-radius-px", type=float, default=3.)
    parser.add_argument("--maximum-radius-px", type=float, default=12.)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    if not 0 < args.calibration_frame_fraction < 1:
        parser.error("calibration frame fraction must lie in (0, 1)")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("lscr_v3_matcher", REPO / "models" / "sc2pcr.py")
    pool_module = load_module("lscr_v3_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                                     nms_radius=.1, max_points=3000, k1=30)
    started = time.time()
    v2_features = RoMaFineFeatures(args.device, args.roma_setting, 4)
    train_v2 = collect_dataset(rows, rows_by_frame, "train", args.lidar_cache, args.train_match_cache_dir, matcher,
                               pool_module.full_pool_refine, args.device, args.seed, v2_features, 4,
                               args.covariance_inflation, args.pixel_floor_px, args.window_sigmas,
                               args.minimum_radius_px, args.maximum_radius_px)
    validation_v2 = collect_dataset(rows, rows_by_frame, "validation", args.lidar_cache,
                                    args.validation_match_cache_dir, matcher, pool_module.full_pool_refine,
                                    args.device, args.seed, v2_features, 4, args.covariance_inflation,
                                    args.pixel_floor_px, args.window_sigmas, args.minimum_radius_px,
                                    args.maximum_radius_px)
    train_centres, validation_centres, v2_head = geometry_v2_centres(train_v2, validation_v2, args)
    del v2_features
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    feature_extractor = RoMaFineFeatures(args.device, args.roma_setting, 1)
    train_collect = lambda: collect_visual_dataset(
        rows, rows_by_frame, "train", args.lidar_cache, args.train_match_cache_dir, matcher,
        pool_module.full_pool_refine, args.device, args.seed, feature_extractor,
        by_frame(train_v2["frame_ids"], train_centres), by_frame(train_v2["frame_ids"], train_v2["baseline"]),
        args.covariance_inflation, args.pixel_floor_px, args.window_sigmas, args.minimum_radius_px,
        args.maximum_radius_px)
    validation_collect = lambda: collect_visual_dataset(
        rows, rows_by_frame, "validation", args.lidar_cache, args.validation_match_cache_dir, matcher,
        pool_module.full_pool_refine, args.device, args.seed, feature_extractor,
        by_frame(validation_v2["frame_ids"], validation_centres),
        by_frame(validation_v2["frame_ids"], validation_v2["baseline"]), args.covariance_inflation,
        args.pixel_floor_px, args.window_sigmas, args.minimum_radius_px, args.maximum_radius_px)
    train_metadata = visual_cache_metadata("train", args, v2_head, train_centres, train_v2, args.train_match_cache_dir)
    validation_metadata = visual_cache_metadata("validation", args, v2_head, validation_centres, validation_v2,
                                                args.validation_match_cache_dir)
    train = (load_or_collect_visual_dataset(args.train_visual_cache, train_metadata, train_collect)
             if args.train_visual_cache else train_collect())
    validation = (load_or_collect_visual_dataset(args.validation_visual_cache, validation_metadata, validation_collect)
                  if args.validation_visual_cache else validation_collect())
    train_frames = ordered_unique(train["frame_ids"])
    calibration_count = max(1, int(np.ceil(len(train_frames) * args.calibration_frame_fraction)))
    calibration_frames = train_frames[-calibration_count:]
    coordinate_train = ~np.isin(train["frame_ids"], calibration_frames)
    controls = {"v3_stride1_2_4": (True, True, True), "v3_stride4_only": (False, False, True),
                "v3_retrained_zero_correlation": (False, False, False)}
    predictions, selected = {}, {}
    heads = {}
    for index, (name, channels) in enumerate(controls.items()):
        head = VisualResidualHead(args.device, args.seed + 200 + index)
        head.fit(train, coordinate_train, args.epochs, args.batch_size, args.learning_rate, args.seed + 200 + index, channels)
        predictions[name], selected[name] = head.predict(validation, args.batch_size, channels)
        heads[name] = head
        print("V3 trained %s" % name, flush=True)
    predictions["v3_same_weight_zero_correlation"], selected["v3_same_weight_zero_correlation"] = heads["v3_stride1_2_4"].predict(
        validation, args.batch_size, (False, False, False))
    result = {"protocol": {"name": "LSCR V3 frozen geometry V2 centre plus multi-scale visual residual",
                            "centre": "V2 geometry-only; train centres are contiguous-frame grouped out-of-fold predictions",
                            "visual_input": "observation-specific reference pixels; stride-1/2/4 correlation maps at u2 +/-6 original pixels; invalid candidates masked",
                            "position_training": "13x13 categorical location plus bounded local offset; coordinate loss is CE + 2*SmoothL1 and does not use predicted uncertainty",
                            "calibration": "last contiguous training-frame block is excluded from position training for later uncertainty calibration; uncertainty head is not yet used for gating",
                            "validation": "development validation only; it was used for earlier model exploration"},
              "settings": vars(args), "samples": {"train": int(len(train["truth"]),), "validation": int(len(validation["truth"]))},
              "window_coverage": {"train": float((targets(train)[0] != NONE_CLASS).mean()),
                                  "validation": float((targets(validation)[0] != NONE_CLASS).mean())},
              "comparison": {"geometry_only_v2": metrics(validation["baseline"], validation["baseline"], validation["truth"]),
                             **{name: metrics(validation["baseline"], prediction, validation["truth"])
                                for name, prediction in predictions.items()}},
              "selected_no_match_fraction": {name: float((cells == NONE_CLASS).mean()) for name, cells in selected.items()},
              "validation_frames": per_frame(validation, predictions["v3_stride1_2_4"]),
              "calibration_frames": calibration_frames, "visual_cache": {"train": train_metadata,
                                                                            "validation": validation_metadata},
              "elapsed_s": time.time() - started}
    import torch
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v2_geometry_state_dict": v2_head.model.state_dict(),
                "v3_state_dict": {name: {"backbone": head.backbone.state_dict(), "location": head.location.state_dict(),
                                            "offset": head.offset.state_dict(), "none": head.none.state_dict()}
                                  for name, head in heads.items()}, "settings": vars(args)}, checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
