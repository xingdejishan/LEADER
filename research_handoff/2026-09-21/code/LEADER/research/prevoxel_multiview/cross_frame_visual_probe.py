"""Measure cross-frame DeDoDe descriptor transfer without LEADER or pose refinement."""
import argparse
import json
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

from local_visual_refinement import DenseDescriptorExtractor, normalize, sample_dense, visible_map_points
from oracle_pose_refinement import load_lidar, project_world


def project_body(points, camera_to_body, intrinsics):
    camera = (points - camera_to_body[:3, 3]) @ camera_to_body[:3, :3]
    pixels = camera @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:3]
    return uv


def frame_descriptors(row, cached, feature_cache, projection_cache, extractor, descriptor_space):
    with np.load(Path(feature_cache) / (row["frame_id"] + ".npz")) as data:
        valid = np.asarray(data["mask"], dtype=bool)
        cached_descriptors = np.asarray(data["image"], dtype=np.float32) if descriptor_space == "pca128" else None
    with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as mapping:
        projection_xyz = np.asarray(mapping["projection_xyz"], dtype=np.float32)
        localization_xyz = np.asarray(mapping["localization_xyz"], dtype=np.float32)
    if not np.array_equal(localization_xyz, cached["source"]):
        raise ValueError("projection/localization mismatch: %s" % row["frame_id"])
    if descriptor_space == "pca128":
        if cached_descriptors.ndim != 3 or cached_descriptors.shape[:2] != valid.shape:
            raise ValueError("feature/source shape mismatch: %s" % row["frame_id"])
        return cached_descriptors, valid, projection_xyz
    descriptor_values = []
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        camera = int(view["camera"])
        dense, image_hw = extractor.image(row["frame_id"], camera, view["image"])
        K = np.loadtxt(view["calibration"]).astype(np.float32)
        extrinsic = np.asarray(view["camera_to_body"], dtype=np.float32)
        uv = project_body(projection_xyz, extrinsic, K)
        indices = np.where(valid[:, camera] & np.isfinite(uv).all(axis=1))[0]
        values = np.zeros((len(projection_xyz), dense.shape[1]), dtype=np.float32)
        if len(indices):
            sampled = sample_dense(
                dense, extractor.torch.from_numpy(uv[indices].astype(np.float32)).to(extractor.device), image_hw
            ).detach().cpu().numpy()
            values[indices] = sampled
        descriptor_values.append(values)
    return np.stack(descriptor_values, axis=1), valid, projection_xyz


def build_history_map(rows, lidar_cache, feature_cache, projection_cache, voxel_size,
                      extractor, descriptor_space):
    point_ids = {}
    points = []
    observations = defaultdict(list)
    for index, row in enumerate(rows):
        cached = load_lidar(lidar_cache, row["frame_id"])
        descriptors, valid, projection_xyz = frame_descriptors(
            row, cached, feature_cache, projection_cache, extractor, descriptor_space
        )
        projection_xyz = projection_xyz.astype(np.float64)
        world = projection_xyz @ cached["GT"][:3, :3].T + cached["GT"][:3, 3]
        keys = np.floor(world / voxel_size).astype(np.int64)
        for point_index, key_array in enumerate(keys):
            key = tuple(int(value) for value in key_array)
            map_index = point_ids.get(key)
            if map_index is None:
                map_index = len(points)
                point_ids[key] = map_index
                points.append(world[point_index])
            for camera in range(descriptors.shape[1]):
                if valid[point_index, camera]:
                    observations[(map_index, camera)].append(
                        (row["frame_id"], descriptors[point_index, camera].copy(), world[point_index].copy()))
        print("map frame %d/%d %s" % (index + 1, len(rows), row["frame_id"]), flush=True)
    deduplicated = defaultdict(list)
    for key, values in observations.items():
        seen_frames = set()
        for source_frame, descriptor, world_point in values:
            if source_frame not in seen_frames:
                deduplicated[key].append((source_frame, descriptor, world_point))
                seen_frames.add(source_frame)
    return np.asarray(points, dtype=np.float64), deduplicated


def offsets_grid(radius, step):
    return np.asarray([(du, dv) for dv in np.arange(-radius, radius + 1, step)
                       for du in np.arange(-radius, radius + 1, step)], dtype=np.float32)


def summarize_scores(scores, offsets, samples):
    center = int(np.where((offsets == 0).all(axis=1))[0][0])
    correct = scores[:, center]
    wrong = np.delete(scores, center, axis=1)
    best_wrong = wrong.max(axis=1)
    best_index = scores.argmax(axis=1)
    rank = 1 + (scores > correct[:, None] + 1e-7).sum(axis=1)
    distance = np.linalg.norm(offsets[best_index], axis=1)
    margin = correct - best_wrong
    return {
        "samples": int(samples),
        "rank1_fraction": float((rank == 1).mean()),
        "distance_lt5_fraction": float((distance < 5.0).mean()),
        "correct_cosine_mean": float(correct.mean()),
        "correct_cosine_p10": float(np.percentile(correct, 10)),
        "best_wrong_cosine_mean": float(best_wrong.mean()),
        "best_wrong_cosine_p90": float(np.percentile(best_wrong, 90)),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "correct_beats_wrong_fraction": float((correct > best_wrong).mean()),
    }


def weighted_summary(records, key):
    total = sum(record[key]["samples"] for record in records)
    if not total:
        return {"samples": 0}
    names = ["rank1_fraction", "distance_lt5_fraction", "correct_cosine_mean",
             "correct_cosine_p10", "best_wrong_cosine_mean", "best_wrong_cosine_p90",
             "margin_mean", "margin_median", "correct_beats_wrong_fraction"]
    return {"samples": int(total), **{
        name: float(sum(record[key][name] * record[key]["samples"] for record in records) / total)
        for name in names}}


def shuffled_descriptors(pool, target_map_ids, rng):
    if not len(pool):
        return None
    pool_map_ids = np.asarray([item[0] for item in pool], dtype=np.int64)
    pool_values = normalize(np.asarray([item[2] for item in pool], dtype=np.float32))
    selected = rng.integers(0, len(pool), size=len(target_map_ids))
    bad = pool_map_ids[selected] == target_map_ids
    for _ in range(8):
        if not bad.any():
            break
        selected[bad] = rng.integers(0, len(pool), size=int(bad.sum()))
        bad = pool_map_ids[selected] == target_map_ids
    if bad.any():
        for target in np.flatnonzero(bad):
            candidates = np.flatnonzero(pool_map_ids != target_map_ids[target])
            if len(candidates):
                selected[target] = candidates[0]
    return pool_values[selected]


def probe_row(row, observations, lidar_cache, extractor, radius, step, seed):
    cached = load_lidar(lidar_cache, row["frame_id"])
    offsets = offsets_grid(radius, step)
    rng = np.random.default_rng(seed)
    records = []
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        camera = int(view["camera"])
        from PIL import Image

        image = np.asarray(Image.open(view["image"]).convert("RGB"))
        image_mask = np.asarray(np.load(view["mask"]))
        dense, image_hw = extractor.image(row["frame_id"], camera, view["image"])
        target_map_ids = []
        target_points = []
        target_descriptors = []
        target_source_frames = []
        for (map_index, observation_camera), values in observations.items():
            if observation_camera != camera:
                continue
            for source_frame, descriptor, world_point in values:
                if source_frame != row["frame_id"]:
                    target_map_ids.append(map_index)
                    target_points.append(world_point)
                    target_descriptors.append(descriptor)
                    target_source_frames.append(source_frame)
        if not target_points:
            continue
        target_points = np.asarray(target_points, dtype=np.float64)
        target_map_ids = np.asarray(target_map_ids, dtype=np.int64)
        target_descriptors = np.asarray(target_descriptors, dtype=np.float32)
        target_source_frames = np.asarray(target_source_frames, dtype="U32")
        target_uv, visible = visible_map_points(target_points, cached["GT"], view, image, image_mask)
        if not len(visible):
            continue
        height, width = image_hw
        visible = visible[(target_uv[visible, 0] >= radius) & (target_uv[visible, 0] < width - radius) &
                          (target_uv[visible, 1] >= radius) & (target_uv[visible, 1] < height - radius)]
        if not len(visible):
            continue
        target_map_ids = target_map_ids[visible]
        target_descriptors = normalize(target_descriptors[visible])
        target_source_frames = target_source_frames[visible]
        base_uv = target_uv[visible].astype(np.float32)
        query_uv = base_uv[:, None] + offsets[None]
        flat_uv = query_uv.reshape(-1, 2)
        sampled = []
        for start in range(0, len(flat_uv), 4096):
            sampled.append(sample_dense(
                dense, extractor.torch.from_numpy(flat_uv[start:start + 4096]).to(extractor.device), image_hw
            ).detach().cpu().numpy())
        sampled = normalize(np.concatenate(sampled, axis=0)).reshape(len(target_map_ids), len(offsets), -1)
        scores = np.einsum("nkd,nd->nk", sampled, target_descriptors)
        pool = []
        for (map_index, pool_camera), values in observations.items():
            if pool_camera != camera:
                continue
            for source_frame, descriptor, world_point in values:
                if source_frame != row["frame_id"]:
                    pool.append((map_index, source_frame, descriptor))
        shuffled = shuffled_descriptors(pool, target_map_ids, rng)
        shuffled_scores = np.einsum("nkd,nd->nk", sampled, shuffled) if shuffled is not None else np.empty((0, len(offsets)))
        records.append({
            "frame_id": row["frame_id"],
            "camera": camera,
            "query_map_points": int(len(np.unique(target_map_ids))),
            "history_observations": int(len(target_descriptors)),
            "source_frames": int(len(set(target_source_frames.tolist()))),
            "actual": summarize_scores(scores, offsets, len(target_descriptors)),
            "shuffled": summarize_scores(shuffled_scores, offsets, len(shuffled)) if shuffled is not None else {"samples": 0},
            "offsets": offsets.tolist(),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--dedode-weights", required=True)
    parser.add_argument("--pca-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--descriptor-space", choices=("pca128", "raw"), default="pca128")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        rows = rows[:args.frames]
    if len(rows) < 2:
        raise ValueError("cross-frame probe requires at least two train frames")
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights,
                                         apply_pca=args.descriptor_space == "pca128")
    points, observations = build_history_map(rows, args.lidar_cache, args.feature_cache,
                                              args.projection_cache, args.map_voxel_size,
                                              extractor, args.descriptor_space)
    records = []
    started = time.time()
    for index, row in enumerate(rows):
        records.extend(probe_row(row, observations, args.lidar_cache, extractor,
                                 args.radius, args.step, args.seed + index))
        print("probe frame %d/%d %s groups=%d" % (index + 1, len(rows), row["frame_id"], len(records)), flush=True)
    by_camera = {str(camera): {
        "actual": weighted_summary([record for record in records if record["camera"] == camera], "actual"),
        "shuffled": weighted_summary([record for record in records if record["camera"] == camera], "shuffled"),
    } for camera in range(6)}
    result = {
        "protocol": {
            "split": "train-only",
            "leave_one_frame_out": True,
            "map_or_lm": "none",
            "map_point_definition": "0.2 m world voxel association; each descriptor keeps its source observation world coordinate",
            "query_center": "GT projection of the map point into the different query frame",
            "visibility": "FOV + image mask + black border + z-buffer occlusion at GT pose",
            "search_window": "square +/- %d px at step %d" % (args.radius, args.step),
            "actual_descriptor": "historical descriptor from the same map point and camera, source frame != query frame",
            "shuffled_control": "random descriptor from a different map point, same camera, source frame != query frame",
            "descriptor": "DeDoDe-B + PCA128" if args.descriptor_space == "pca128" else "raw DeDoDe-B descriptor",
            "descriptor_space": args.descriptor_space,
            "validation_used": False,
        },
        "map": {"points": int(len(points)), "observations": int(sum(len(v) for v in observations.values())),
                "train_frames": len(rows), "voxel_size_m": args.map_voxel_size},
        "overall": {
            "actual": weighted_summary(records, "actual"),
            "shuffled": weighted_summary(records, "shuffled"),
        },
        "by_camera": by_camera,
        "records": records,
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_camera": by_camera}, indent=2))


if __name__ == "__main__":
    main()
