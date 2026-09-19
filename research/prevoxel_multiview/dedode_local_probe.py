"""Measure DeDoDe local pixel discriminability without matching or pose refinement."""
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

from local_visual_refinement import DenseDescriptorExtractor, normalize, sample_dense
from oracle_pose_refinement import load_lidar, project_world


def frame_probe(row, lidar_cache, feature_cache, projection_cache, extractor, radius, step):
    cached = load_lidar(lidar_cache, row["frame_id"])
    with np.load(Path(feature_cache) / (row["frame_id"] + ".npz")) as data:
        reference = np.asarray(data["image"], dtype=np.float32)
        valid = np.asarray(data["mask"], dtype=bool)
    with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as data:
        projection_xyz = np.asarray(data["projection_xyz"], dtype=np.float64)
        localization_xyz = np.asarray(data["localization_xyz"], dtype=np.float64)
    if not np.array_equal(localization_xyz, cached["source"]):
        raise ValueError("projection/localization mismatch: %s" % row["frame_id"])
    world = projection_xyz @ cached["GT"][:3, :3].T + cached["GT"][:3, 3]
    offsets = np.asarray([(du, dv) for dv in np.arange(-radius, radius + 1, step)
                          for du in np.arange(-radius, radius + 1, step)], dtype=np.float32)
    center = int(np.where((offsets == 0).all(axis=1))[0][0])
    records = []
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        camera = int(view["camera"])
        dense, image_hw = extractor.image(row["frame_id"], camera, view["image"])
        K = np.loadtxt(view["calibration"]).astype(np.float64)
        extrinsic = np.asarray(view["camera_to_body"], dtype=np.float64)
        uv, depth = project_world(world, cached["GT"], extrinsic, K)
        height, width = image_hw
        keep = valid[:, camera].copy()
        keep &= np.isfinite(depth) & (depth > 0)
        keep &= np.isfinite(uv).all(axis=1)
        keep &= (uv[:, 0] >= radius) & (uv[:, 0] < width - radius)
        keep &= (uv[:, 1] >= radius) & (uv[:, 1] < height - radius)
        indices = np.where(keep)[0]
        if not len(indices):
            continue
        query_uv = uv[indices, None, :].astype(np.float32) + offsets[None]
        flat = query_uv.reshape(-1, 2)
        sampled = []
        for start in range(0, len(flat), 4096):
            sampled.append(sample_dense(dense, extractor.torch.from_numpy(flat[start:start + 4096]).to(extractor.device), image_hw)
                           .detach().cpu().numpy())
        sampled = normalize(np.concatenate(sampled, axis=0)).reshape(len(indices), len(offsets), -1)
        reference_values = normalize(reference[indices, camera])
        scores = np.einsum("nkd,nd->nk", sampled, reference_values)
        correct = scores[:, center]
        wrong = np.delete(scores, center, axis=1)
        best_wrong = wrong.max(axis=1)
        rank = 1 + (scores > correct[:, None] + 1e-7).sum(axis=1)
        records.append({
            "frame_id": row["frame_id"], "camera": camera, "samples": int(len(indices)),
            "rank1_fraction": float((rank == 1).mean()),
            "rank_le_3_fraction": float((rank <= 3).mean()),
            "median_rank": float(np.median(rank)),
            "correct_cosine_mean": float(correct.mean()),
            "correct_cosine_p10": float(np.percentile(correct, 10)),
            "best_wrong_cosine_mean": float(best_wrong.mean()),
            "best_wrong_cosine_p90": float(np.percentile(best_wrong, 90)),
            "margin_mean": float((correct - best_wrong).mean()),
            "margin_median": float(np.median(correct - best_wrong)),
            "correct_beats_wrong_fraction": float((correct > best_wrong).mean()),
            "offsets": offsets.tolist(),
        })
    return records


def summarize(records):
    def weighted(name):
        total = sum(record["samples"] for record in records)
        return float(sum(record[name] * record["samples"] for record in records) / max(total, 1))

    return {
        "groups": len(records), "samples": int(sum(record["samples"] for record in records)),
        "rank1_fraction": weighted("rank1_fraction"),
        "rank_le_3_fraction": weighted("rank_le_3_fraction"),
        "correct_cosine_mean": weighted("correct_cosine_mean"),
        "best_wrong_cosine_mean": weighted("best_wrong_cosine_mean"),
        "best_wrong_cosine_p90_mean": weighted("best_wrong_cosine_p90"),
        "margin_mean": weighted("margin_mean"),
        "margin_median_mean": weighted("margin_median"),
        "correct_beats_wrong_fraction": weighted("correct_beats_wrong_fraction"),
    }


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
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        rows = rows[:args.frames]
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    records = []
    started = time.time()
    for index, row in enumerate(rows):
        records.extend(frame_probe(row, args.lidar_cache, args.feature_cache,
                                   args.projection_cache, extractor, args.radius, args.step))
        print("train %d/%d %s groups=%d" % (index + 1, len(rows), row["frame_id"], len(records)), flush=True)
    by_camera = {str(camera): summarize([record for record in records if record["camera"] == camera])
                 for camera in range(6)}
    result = {
        "protocol": {"split": "train-only", "map_or_lm": "none",
                     "correct_pixel": "GT reprojection of the frame's own projection_xyz",
                     "wrong_pixels": "square offsets within +/- %d px at step %d" % (args.radius, args.step),
                     "descriptor": "DeDoDe-B + PCA128", "correctness": "correct cosine > every wrong-pixel cosine"},
        "overall": summarize(records), "by_camera": by_camera,
        "records": records, "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_camera": by_camera}, indent=2))


if __name__ == "__main__":
    main()
