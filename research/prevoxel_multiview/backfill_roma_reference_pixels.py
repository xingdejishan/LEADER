import argparse
import json
from pathlib import Path

import numpy as np


def recover_reference_pixels(cache, reference):
    points = np.asarray(cache["points"], dtype=np.float32)
    frames = np.asarray(cache["reference_frames"]).astype(str)
    cameras = np.asarray(cache["reference_cameras"], dtype=np.int8)
    map_points = np.asarray(reference["world_xyz"], dtype=np.float32)
    map_frames = np.asarray(reference["frame_ids"]).astype(str)
    map_cameras = np.asarray(reference["camera_ids"], dtype=np.int8)
    map_pixels = np.asarray(reference["ref_uv"], dtype=np.float32)
    result = np.empty((len(points), 2), dtype=np.float32)
    for index, (point, frame, camera) in enumerate(zip(points, frames, cameras)):
        candidates = np.where((map_frames == frame) & (map_cameras == camera) &
                              np.equal(map_points, point).all(axis=1))[0]
        if len(candidates) != 1:
            raise ValueError("expected one reference observation for row %d (%s, camera %d); found %d" %
                             (index, frame, camera, len(candidates)))
        result[index] = map_pixels[candidates[0]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-dir", required=True)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--output-cache-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_cache_dir)
    output = Path(args.output_cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference_map) as reference:
        required = {"world_xyz", "frame_ids", "camera_ids", "ref_uv"}
        if not required.issubset(reference.files):
            raise ValueError("reference map lacks observation-specific reference pixels")
        for path in sorted(source.glob("*.npz")):
            destination = output / path.name
            with np.load(path) as cache:
                if "reference_pixels" in cache.files:
                    raise ValueError("source cache already has reference_pixels: %s" % path)
                pixels = recover_reference_pixels(cache, reference)
                values = {key: np.asarray(cache[key]) for key in cache.files}
            values["reference_pixels"] = pixels
            values["cache_schema"] = np.asarray("reference_pixels_v1")
            values["reference_pixels_provenance"] = np.asarray(
                "exact world_xyz + reference_frame + reference_camera lookup in reference-observation map")
            np.savez_compressed(destination, **values)
            print("backfilled %s rows=%d median_query_to_reference_px=%.3f" %
                  (path.name, len(pixels), np.median(np.linalg.norm(values["pixels"] - pixels, axis=1))), flush=True)


if __name__ == "__main__":
    main()
