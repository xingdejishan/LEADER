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


def project_body(points, camera_to_body, intrinsics):
    camera = (points - camera_to_body[:3, 3]) @ camera_to_body[:3, :3]
    pixels = camera @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:3]
    return uv


def compare_frame(row, feature_cache, projection_cache, extractor):
    with np.load(Path(feature_cache) / (row["frame_id"] + ".npz")) as data:
        cached = np.asarray(data["image"], dtype=np.float32)
        valid = np.asarray(data["mask"], dtype=bool)
    with np.load(Path(projection_cache) / (row["frame_id"] + ".npz")) as data:
        points = np.asarray(data["projection_xyz"], dtype=np.float32)
    records = []
    for view in sorted(row["views"], key=lambda item: item["camera"]):
        camera = int(view["camera"])
        dense, image_hw = extractor.image(row["frame_id"], camera, view["image"])
        height, width = image_hw
        K = np.loadtxt(view["calibration"]).astype(np.float32)
        extrinsic = np.asarray(view["camera_to_body"], dtype=np.float32)
        uv = project_body(points, extrinsic, K)
        keep = valid[:, camera] & np.isfinite(uv).all(axis=1)
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        indices = np.where(keep)[0]
        if not len(indices):
            continue
        sampled = sample_dense(
            dense, extractor.torch.from_numpy(uv[indices].astype(np.float32)).to(extractor.device), image_hw
        ).detach().cpu().numpy()
        cache_values = cached[indices, camera]
        new_values = normalize(sampled)
        cache_values = normalize(cache_values)
        cosine = np.sum(cache_values * new_values, axis=1)
        records.append({
            "frame_id": row["frame_id"], "camera": camera, "samples": int(len(indices)),
            "cosine_mean": float(cosine.mean()), "cosine_p01": float(np.percentile(cosine, 1)),
            "cosine_p10": float(np.percentile(cosine, 10)), "cosine_median": float(np.median(cosine)),
            "cosine_p90": float(np.percentile(cosine, 90)),
            "fraction_ge_099": float((cosine >= .99).mean()),
            "fraction_ge_0999": float((cosine >= .999).mean()),
            "fraction_ge_09999": float((cosine >= .9999).mean()),
            "max_abs_normalized_delta": float(np.max(np.abs(cache_values - new_values))),
        })
    return records


def summarize(records):
    total = sum(record["samples"] for record in records)
    if not total:
        return {"samples": 0}
    names = ["cosine_mean", "cosine_p01", "cosine_p10", "cosine_median", "cosine_p90",
             "fraction_ge_099", "fraction_ge_0999", "fraction_ge_09999", "max_abs_normalized_delta"]
    return {"samples": int(total), **{
        name: float(sum(record[name] * record["samples"] for record in records) / total)
        for name in names}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--dedode-weights", required=True)
    parser.add_argument("--pca-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    rows = [row for row in json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if row["split"] == "train"]
    if args.frames:
        rows = rows[:args.frames]
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    records = []
    started = time.time()
    for index, row in enumerate(rows):
        records.extend(compare_frame(row, args.feature_cache, args.projection_cache, extractor))
        print("sanity frame %d/%d %s groups=%d" % (index + 1, len(rows), row["frame_id"], len(records)), flush=True)
    result = {
        "protocol": {
            "split": "train-only",
            "same_image_same_pixel": True,
            "cached_descriptor": "feature_cache image at the cached valid projection point",
            "regenerated_descriptor": "DenseDescriptorExtractor at the same original-resolution projection pixel",
            "preprocessing": "resize short side 480, ImageNet normalization, DeDoDe-B, PCA128",
            "validation_used": False,
        },
        "overall": summarize(records),
        "by_camera": {str(camera): summarize([r for r in records if r["camera"] == camera])
                      for camera in range(6)},
        "records": records,
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_camera": result["by_camera"]}, indent=2))


if __name__ == "__main__":
    main()
