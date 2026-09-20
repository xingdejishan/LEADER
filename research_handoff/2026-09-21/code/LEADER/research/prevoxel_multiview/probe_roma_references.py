"""Train-only leave-one-out probe for RoMa reference-image retrieval."""
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

from local_visual_refinement import visible_map_points
from local_visual_refinement_roma import RoMaField, build_reference_observations
from oracle_pose_refinement import load_lidar, load_module, pose_from_baseline, project_world


def image_valid(uv, depth, image, mask):
    height, width = image.shape[:2]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > .5)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    integer = np.rint(uv).astype(np.int64)
    integer[:, 0] = np.clip(integer[:, 0], 0, width - 1)
    integer[:, 1] = np.clip(integer[:, 1], 0, height - 1)
    valid &= mask[integer[:, 1], integer[:, 0]] > 0
    valid &= image[integer[:, 1], integer[:, 0]].max(axis=1) > 10
    return valid


def pair_summary(frame_id, camera, pair, selection_score, query_uv, overlap, truth_uv, truth_valid,
                 query_height, query_width, min_overlap):
    predicted = np.isfinite(query_uv).all(axis=1)
    predicted &= (query_uv[:, 0] >= 0) & (query_uv[:, 0] < query_width)
    predicted &= (query_uv[:, 1] >= 0) & (query_uv[:, 1] < query_height)
    error = np.linalg.norm(query_uv - truth_uv, axis=1)
    usable = truth_valid & predicted & (overlap >= min_overlap)
    usable_error = error[usable]
    overlap_pass = int(usable.sum())
    correct_5 = int((usable & (error < 5.)).sum())
    correct_8 = int((usable & (error < 8.)).sum())
    return {
        "query_frame_id": frame_id,
        "query_camera": int(camera),
        "reference_frame_id": pair[0],
        "reference_camera": int(pair[1]),
        "selection_covisible_anchor_count": int(selection_score),
        "anchors": int(len(query_uv)),
        "gt_visible_anchors": int(truth_valid.sum()),
        "prediction_in_image": int((truth_valid & predicted).sum()),
        "overlap_pass": overlap_pass,
        "correct_lt_5px": correct_5,
        "correct_lt_8px": correct_8,
        "precision_lt_5px": float(correct_5 / overlap_pass) if overlap_pass else float("nan"),
        "precision_lt_8px": float(correct_8 / overlap_pass) if overlap_pass else float("nan"),
        "median_pixel_error_overlap": float(np.median(usable_error)) if len(usable_error) else float("nan"),
        "p90_pixel_error_overlap": float(np.percentile(usable_error, 90)) if len(usable_error) else float("nan"),
    }


def aggregate(records):
    if not records:
        return {"pairs": 0, "anchors": 0, "overlap_pass": 0, "correct_lt_5px": 0,
                "correct_lt_8px": 0, "precision_lt_5px": float("nan"), "precision_lt_8px": float("nan")}
    overlap = sum(record["overlap_pass"] for record in records)
    return {
        "pairs": int(len(records)),
        "anchors": int(sum(record["anchors"] for record in records)),
        "gt_visible_anchors": int(sum(record["gt_visible_anchors"] for record in records)),
        "overlap_pass": int(overlap),
        "correct_lt_5px": int(sum(record["correct_lt_5px"] for record in records)),
        "correct_lt_8px": int(sum(record["correct_lt_8px"] for record in records)),
        "precision_lt_5px": float(sum(record["correct_lt_5px"] for record in records) / overlap) if overlap else float("nan"),
        "precision_lt_8px": float(sum(record["correct_lt_8px"] for record in records) / overlap) if overlap else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-setting", default="precise")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--cameras", default="")
    parser.add_argument("--max-pairs-per-camera", type=int, default=0)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--min-overlap", type=float, default=.2)
    parser.add_argument("--min-reference-view-cosine", type=float, default=.7)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2089)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        train_rows = train_rows[:args.frames]
    if not train_rows:
        raise ValueError("manifest has no train rows")
    cameras = set(int(value) for value in args.cameras.split(",") if value) if args.cameras else set(range(6))
    references = build_reference_observations(rows, args.lidar_cache, args.projection_cache,
                                              args.map_voxel_size, Path(args.map_cache), args.max_history)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    matcher_module = load_module("roma_probe_matcher", REPO / "models" / "sc2pcr.py")
    pool_module = load_module("roma_probe_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    roma = RoMaField(args.device, args.roma_setting)
    observation_world = references["world_xyz"].astype(np.float64)
    records, started = [], time.time()
    for frame_index, row in enumerate(train_rows):
        initial, gt, _ = pose_from_baseline(row, args.lidar_cache, matcher, pool_module.full_pool_refine,
                                            args.device, args.seed + frame_index)
        for query_view in sorted(row["views"], key=lambda value: value["camera"]):
            camera = int(query_view["camera"])
            if camera not in cameras:
                continue
            from PIL import Image
            query_image = np.asarray(Image.open(query_view["image"]).convert("RGB"))
            query_mask = np.asarray(np.load(query_view["mask"]))
            query_height, query_width = query_image.shape[:2]
            local_observations = np.where(np.linalg.norm(observation_world - initial[:3, 3], axis=1) <= args.crop_radius)[0]
            _, visible_local = visible_map_points(observation_world[local_observations], initial, query_view, query_image, query_mask)
            eligible = local_observations[visible_local]
            eligible = eligible[references["frame_ids"][eligible] != row["frame_id"]]
            groups = defaultdict(list)
            for record_index in eligible:
                groups[(str(references["frame_ids"][record_index]), int(references["camera_ids"][record_index]))].append(record_index)
            query_center = (initial @ np.asarray(query_view["camera_to_body"], dtype=np.float64))[:3, 3]
            scores = {}
            for pair, indices in groups.items():
                reference_row = rows_by_frame[pair[0]]
                reference_view = next(view for view in reference_row["views"] if int(view["camera"]) == pair[1])
                reference_pose = load_lidar(args.lidar_cache, pair[0])["GT"]
                reference_center = (reference_pose @ np.asarray(reference_view["camera_to_body"], dtype=np.float64))[:3, 3]
                anchor = references["world_xyz"][np.asarray(indices, dtype=np.int64)].astype(np.float64)
                left, right = anchor - query_center, anchor - reference_center
                left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-9)
                right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
                scores[pair] = int(((left * right).sum(axis=1) >= args.min_reference_view_cosine).sum())
            pairs = [pair for pair in groups if scores[pair] > 0]
            pairs.sort(key=lambda pair: (-scores[pair], -len(groups[pair]), pair))
            if args.max_pairs_per_camera:
                pairs = pairs[:args.max_pairs_per_camera]
            top_pairs = set(pairs[:args.top_k])
            pair_records = []
            for pair in pairs:
                reference_row = rows_by_frame[pair[0]]
                reference_view = next(view for view in reference_row["views"] if int(view["camera"]) == pair[1])
                indices = np.asarray(groups[pair], dtype=np.int64)
                with Image.open(reference_view["image"]) as ref_image:
                    ref_height, ref_width = ref_image.height, ref_image.width
                prediction = roma.match(reference_view["image"], query_view["image"])
                query_uv, overlap, _ = roma.sample(prediction, references["ref_uv"][indices],
                                                    (ref_height, ref_width), (query_height, query_width))
                world = references["world_xyz"][indices].astype(np.float64)
                truth_uv, depth = project_world(world, gt, np.asarray(query_view["camera_to_body"], dtype=np.float64),
                                                np.loadtxt(query_view["calibration"]).astype(np.float64))
                truth_valid = image_valid(truth_uv, depth, query_image, query_mask)
                record = pair_summary(row["frame_id"], camera, pair, scores[pair], query_uv, overlap, truth_uv,
                                      truth_valid, query_height, query_width, args.min_overlap)
                record["automatic_top_k"] = pair in top_pairs
                pair_records.append(record)
                roma.clear_cache()
            top_records = [record for record in pair_records if record["automatic_top_k"]]
            best = sorted(pair_records, key=lambda record: (-record["correct_lt_8px"], -record["correct_lt_5px"],
                                                            record["median_pixel_error_overlap"]))[0] if pair_records else None
            records.append({"frame_id": row["frame_id"], "camera": camera, "all_pairs": pair_records,
                            "automatic_top_k": aggregate(top_records), "oracle_best_correspondence_count_pair": best,
                            "all_geometrically_compatible": aggregate(pair_records)})
            print("probe %d/%d %s cam=%d pairs=%d top8=%d best8=%d" %
                  (frame_index + 1, len(train_rows), row["frame_id"], camera, len(pair_records),
                   sum(record["correct_lt_8px"] for record in top_records), best["correct_lt_8px"] if best else 0), flush=True)
        roma.clear_cache()
    result = {"protocol": {"split": "train only, leave-one-frame-out", "gt_use": "labels only; never retrieval or matching",
                             "front_end": "RoMa v2 dense reference-to-query field", "local_gate": "disabled",
                             "reference_pool": "all historical image pairs with at least one ray-compatible visible observation",
                             "automatic_retrieval": "top-k by current ray-compatible visible-observation count",
                             "oracle": "pair with the largest correct correspondence count; pair precision is reported separately"},
              "settings": vars(args), "records": records, "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
