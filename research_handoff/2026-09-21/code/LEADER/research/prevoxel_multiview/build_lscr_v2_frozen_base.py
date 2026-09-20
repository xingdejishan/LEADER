import argparse
import hashlib
import json
import sys
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
from lscr_refinement import RoMaFineFeatures, lidar_pixel_prior, truth_pixels
from lscr_v2_offset_head import OffsetHead, collect_dataset, pixel_geometry_features
from oracle_pose_refinement import load_module, pose_from_baseline


BASE_VERSION = "lscr_frozen_v2_base_v1"


def ordered_unique(values):
    return list(dict.fromkeys(str(value) for value in values))


def digest_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def digest_state_dict(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(digest_array(value).encode())
    return digest.hexdigest()


def fit_v2_heads(train, args):
    target = train["truth"] - train["baseline"]
    zero = np.zeros_like(train["volume"])
    frame_order = ordered_unique(train["frame_ids"])
    groups = np.array_split(np.asarray(frame_order), min(args.oof_folds, len(frame_order)))
    per_frame = {}
    for fold_index, frames in enumerate(groups):
        held = np.isin(train["frame_ids"], frames)
        head = OffsetHead(train["volume"].shape[1], train["geometry"].shape[1], args.hidden, args.device,
                          seed=args.seed + fold_index)
        head.fit(zero[~held], train["geometry"][~held], target[~held], args.epochs, args.batch_size,
                 args.learning_rate, args.seed + fold_index)
        for frame in frames:
            per_frame[str(frame)] = head
    full = OffsetHead(train["volume"].shape[1], train["geometry"].shape[1], args.hidden, args.device,
                      seed=args.seed + 100)
    full.fit(zero, train["geometry"], target, args.epochs, args.batch_size, args.learning_rate, args.seed + 100)
    return per_frame, full, int(train["volume"].shape[1])


def collect_base(rows, split, rows_by_frame, lidar_cache, cache_dir, matcher, pool, heads, full_head, correlation_size,
                 args):
    selected = [row for row in rows if row["split"] in (("val", "validation") if split == "validation" else (split,))]
    selected = [row for row in selected if (Path(cache_dir) / (row["frame_id"] + ".npz")).exists()]
    inputs = {key: [] for key in ("world_xyz", "roma_pixels", "reference_pixels", "camera_ids", "anchor_ids",
                                  "reference_frames", "reference_cameras", "lidar_pixels", "lidar_covariance",
                                  "v2_pixels", "frame_ids")}
    labels = {"gt_pixels": [], "gt_visible": []}
    for position, row in enumerate(selected):
        same_split = [candidate["frame_id"] for candidate in rows if candidate["split"] == row["split"]]
        pose, gt, _, evidence = pose_from_baseline(row, lidar_cache, matcher, pool, args.device,
                                                    args.seed + same_split.index(row["frame_id"]),
                                                    return_lidar_evidence=True)
        if evidence is None:
            raise RuntimeError("full-pool implementation did not return final Tukey evidence")
        lidar = frozen_lidar_information(pose, evidence)
        points, roma, reference, cameras, _, _, anchor_ids, reference_frames, reference_cameras, _ = load_match_cache(
            Path(cache_dir) / (row["frame_id"] + ".npz"))
        lidar_pixels, covariance, _, radii = lidar_pixel_prior(
            points, cameras, pose, row["views"], lidar["covariance"], args.covariance_inflation, args.pixel_floor_px,
            args.window_sigmas, args.minimum_radius_px, args.maximum_radius_px)
        valid = (np.isfinite(points).all(axis=1) & np.isfinite(roma).all(axis=1) & np.isfinite(reference).all(axis=1) &
                 np.isfinite(lidar_pixels).all(axis=1) & np.isfinite(covariance).all(axis=(1, 2)))
        geometry = pixel_geometry_features(roma[valid], lidar_pixels[valid], covariance[valid], radii[valid])
        head = full_head if split == "validation" else heads[row["frame_id"]]
        offset, _ = head.predict(np.zeros((len(geometry), correlation_size), dtype=np.float32), geometry, args.batch_size)
        truth, visible = truth_pixels(points, cameras, row, gt)
        inputs["world_xyz"].append(points[valid])
        inputs["roma_pixels"].append(roma[valid])
        inputs["reference_pixels"].append(reference[valid])
        inputs["camera_ids"].append(cameras[valid])
        inputs["anchor_ids"].append(anchor_ids[valid])
        inputs["reference_frames"].append(reference_frames[valid])
        inputs["reference_cameras"].append(reference_cameras[valid])
        inputs["lidar_pixels"].append(lidar_pixels[valid])
        inputs["lidar_covariance"].append(covariance[valid])
        inputs["v2_pixels"].append(roma[valid] + offset)
        inputs["frame_ids"].append(np.full(int(valid.sum()), row["frame_id"], dtype="U32"))
        labels["gt_pixels"].append(truth[valid])
        labels["gt_visible"].append(visible[valid])
        print("frozen V2 base %s %d/%d %s samples=%d labels=%d" %
              (split, position + 1, len(selected), row["frame_id"], valid.sum(), visible[valid].sum()), flush=True)
    return ({key: np.concatenate(value) for key, value in inputs.items()},
            {key: np.concatenate(value) for key, value in labels.items()})


def save_verified(path, payload, metadata):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if "metadata" not in cached or json.loads(str(cached["metadata"].item())) != metadata:
                raise ValueError("existing frozen base cache metadata differs: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--train-match-cache-dir", required=True)
    parser.add_argument("--validation-match-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--oof-folds", type=int, default=5)
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
    matcher_module = load_module("frozen_v2_base_matcher", REPO / "models" / "sc2pcr.py")
    pool_module = load_module("frozen_v2_base_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                                     nms_radius=.1, max_points=3000, k1=30)
    features = RoMaFineFeatures(args.device, args.roma_setting, 4)
    train = collect_dataset(rows, rows_by_frame, "train", args.lidar_cache, args.train_match_cache_dir, matcher,
                            pool_module.full_pool_refine, args.device, args.seed, features, 4,
                            args.covariance_inflation, args.pixel_floor_px, args.window_sigmas,
                            args.minimum_radius_px, args.maximum_radius_px)
    heads, full_head, correlation_size = fit_v2_heads(train, args)
    metadata = {"version": BASE_VERSION, "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "full_v2_sha256": digest_state_dict(full_head.model.state_dict()),
                "oof_v2_sha256": {frame: digest_state_dict(head.model.state_dict()) for frame, head in heads.items()},
                "settings": vars(args), "input_fields": ["world_xyz", "roma_pixels", "reference_pixels", "camera_ids",
                "anchor_ids", "reference_frames", "reference_cameras", "lidar_pixels", "lidar_covariance", "v2_pixels", "frame_ids"],
                "label_fields": ["gt_pixels", "gt_visible"]}
    for split, directory in (("train", args.train_match_cache_dir), ("validation", args.validation_match_cache_dir)):
        inputs, labels = collect_base(rows, split, rows_by_frame, args.lidar_cache, directory, matcher,
                                      pool_module.full_pool_refine, heads, full_head, correlation_size, args)
        split_metadata = {**metadata, "split": split, "input_sha256": {key: digest_array(value) for key, value in inputs.items()}}
        output = Path(args.output_dir)
        save_verified(output / ("%s_inputs.npz" % split), inputs, split_metadata)
        save_verified(output / ("%s_labels.npz" % split), labels, {**split_metadata, "role": "labels"})


if __name__ == "__main__":
    main()
