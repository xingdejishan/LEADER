import argparse
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

from local_visual_refinement_roma import build_reference_observations
from oracle_pose_refinement import load_module
from surface_patch_refinement import (
    digest_json,
    patch_visual_residual_blocks,
    pose_from_baseline_online,
    prepare_patch_context,
    select_surface_patches,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--online-cache", required=True)
    parser.add_argument("--reference-lidar-cache", required=True)
    parser.add_argument("--projection-cache", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--map-voxel-size", type=float, default=.2)
    parser.add_argument("--surface-voxel-size", type=float, default=.2)
    parser.add_argument("--crop-radius", type=float, default=80.)
    parser.add_argument("--min-view-cosine", type=float, default=.7)
    parser.add_argument("--plane-radius", type=float, default=.8)
    parser.add_argument("--min-plane-neighbors", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-patches-per-camera", type=int, default=48)
    parser.add_argument("--grid-cell", type=int, default=32)
    parser.add_argument("--min-contrast", type=float, default=.03)
    args = parser.parse_args()

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.frames:
        train_rows = train_rows[:args.frames]
    references = build_reference_observations(
        rows, args.reference_lidar_cache, args.projection_cache, args.map_voxel_size,
        Path(args.map_cache), 0)
    rows_by_frame = {row["frame_id"]: row for row in rows}
    reference_context = prepare_patch_context(
        references, rows_by_frame, args.reference_lidar_cache, args.surface_voxel_size)
    matcher_module = load_module("patch_scale_matcher", REPO / "models" / "sc2pcr.py")
    full_pool_module = load_module("patch_scale_full_pool", Path(args.full_pool))
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    rms_values = []
    frame_records = []
    for index, row in enumerate(train_rows):
        initial, _ = pose_from_baseline_online(
            row, args.online_cache, matcher, full_pool_module.full_pool_refine,
            args.device, args.seed + index, return_lidar_evidence=False)
        patches, query_images, diagnostics = select_surface_patches(
            row, initial, references, rows_by_frame, reference_context, args.crop_radius,
            args.min_view_cosine, args.plane_radius, args.min_plane_neighbors, args.patch_size,
            args.max_patches_per_camera, args.grid_cell, args.min_contrast, visibility_check=False)
        camera_data = {int(view["camera"]): (
            np.asarray(view["camera_to_body"], dtype=np.float64),
            np.loadtxt(view["calibration"]).astype(np.float64)) for view in row["views"]}
        blocks = patch_visual_residual_blocks(initial, patches, query_images, camera_data)
        rms = np.asarray([np.sqrt(np.mean(block ** 2)) for block in blocks], dtype=np.float64)
        rms_values.extend(rms.tolist())
        frame_records.append({"frame_id": row["frame_id"], "patch_count": len(patches),
                             "rms_count": len(rms), "diagnostics": diagnostics})
        print("%d/%d %s patches=%d rms=%d" % (
            index + 1, len(train_rows), row["frame_id"], len(patches), len(rms)), flush=True)

    values = np.asarray(rms_values, dtype=np.float64)
    if not len(values):
        raise ValueError("no training patch RMS values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-3)
    payload = {
        "patch_robust_scale": scale,
        "estimator": "1.4826 * MAD of train-only normalized patch RMS",
        "training_only": True,
        "query_gt_used": False,
        "count": int(len(values)),
        "median": median,
        "mad": mad,
        "p90": float(np.percentile(values, 90)),
        "manifest_sha256": digest_json(rows),
        "parameters": {key: value for key, value in vars(args).items() if key != "output"},
        "frames": frame_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
