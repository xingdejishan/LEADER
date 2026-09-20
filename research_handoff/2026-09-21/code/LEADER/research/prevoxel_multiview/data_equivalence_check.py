"""Compare original NCLT_mink quantization with prepare_frame on real frames."""
import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

for optional_dependency in ("pypatchworkpp", "open3d", "cv2"):
    if optional_dependency not in sys.modules:
        try:
            __import__(optional_dependency)
        except ModuleNotFoundError:
            sys.modules[optional_dependency] = types.ModuleType(optional_dependency)

from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
from dataset_hook import load_config
from sanity_check import leader_quantization_config
from smoke_dataset import SmokeFrameDataset, prepare_frame
from train_smoke import scene_targets
from utils.pose_util import polar_expansion_to_cartesian


class GeometryOnlyObserver:
    def observe(self, frame_id, scan):
        n = len(scan)
        return {
            "img_feat": np.zeros((n, 6, 128), dtype=np.float32),
            "quality": np.zeros((n, 6, 9), dtype=np.float32),
            "valid": np.zeros((n, 6), dtype=bool),
        }


def original_nclt_item(row, voxel_size, horizontal_res, gt):
    dataset = NCLT_mink.__new__(NCLT_mink)
    dataset.pcs = [row["scan"]]
    dataset.poses = np.zeros((1, 6), dtype=np.float32)
    dataset.poses[0, :3] = gt[:3, 3]
    dataset.rots = gt[None, :3, :3].astype(np.float32)
    dataset.voxel_size = voxel_size
    dataset.min_range = 1.0
    dataset.max_range = 100.0
    dataset.horizontal_res = horizontal_res
    dataset.level_correction = False
    return dataset[0]


def independent_target(coords, voxel_size, horizontal_res, gt, center_t):
    centers = torch.from_numpy(coords.astype(np.float32) + 0.5).float() * voxel_size
    local = polar_expansion_to_cartesian(centers, horizontal_res * voxel_size)
    transform = torch.from_numpy(gt.astype(np.float32))
    return (local @ transform[:3, :3].T + transform[:3, 3]
            - torch.from_numpy(center_t.astype(np.float32))).numpy()


def compare_frame(row, observer, quantization, center_t, lidar_cache):
    frame_id = row["frame_id"]
    with np.load(Path(lidar_cache) / (frame_id + ".npz")) as cached:
        cached_gt = np.asarray(cached["GT"], dtype=np.float64)
    pose = np.loadtxt(row["pose"]).astype(np.float64)
    camera_to_body = next(
        np.asarray(view["camera_to_body"], dtype=np.float64)
        for view in row.get("views", []) if view["camera"] == 5
    )
    manifest_gt = pose @ np.linalg.inv(camera_to_body)
    pose_float32 = pose.astype(np.float32)
    camera_float32 = camera_to_body.astype(np.float32)
    prepare_gt = pose_float32 @ np.linalg.inv(camera_float32)
    original_coords, original_features, original_scan, _, _ = original_nclt_item(
        row, quantization["voxel_size"], quantization["horizontal_res"], prepare_gt)
    prepared = prepare_frame(row, observer, quantization["voxel_size"], quantization["horizontal_res"])
    prepared_features = prepared["lidar_feats"][prepared["index"]]
    coords_equal = np.array_equal(np.asarray(original_coords), prepared["coords_q"])
    features_equal = np.array_equal(np.asarray(original_features), prepared_features)
    scan_equal = np.array_equal(np.asarray(original_scan), prepared["scan"])

    encoded = SimpleNamespace(
        C=torch.from_numpy(np.concatenate([
            np.zeros((len(original_coords), 1), dtype=np.int32),
            np.asarray(original_coords, dtype=np.int32),
        ], axis=1)),
        F=torch.empty((len(original_coords), 1), dtype=torch.float32),
        tensor_stride=(1, 1, 1),
    )
    prepared_target = scene_targets(
        encoded, row, quantization["voxel_size"], quantization["horizontal_res"],
        torch.from_numpy(center_t.astype(np.float32)),
    ).numpy()
    original_target = independent_target(
        np.asarray(original_coords), quantization["voxel_size"],
        quantization["horizontal_res"], prepare_gt, center_t,
    )
    target_max_abs = float(np.max(np.abs(prepared_target - original_target)))
    return {
        "frame_id": frame_id,
        "raw_points": int(len(original_scan)),
        "quantized_points": int(len(original_coords)),
        "scan_equal": bool(scan_equal),
        "coords_equal": bool(coords_equal),
        "lidar_features_equal": bool(features_equal),
        "target_max_abs": target_max_abs,
        "manifest_gt_vs_cached_gt_max_abs": float(np.max(np.abs(manifest_gt - cached_gt))),
        "float32_prepare_gt_vs_cached_gt_max_abs": float(np.max(np.abs(prepare_gt - cached_gt))),
        "camera_to_body_det": float(np.linalg.det(camera_to_body[:3, :3])),
        "manifest_gt_translation": manifest_gt[:3, 3].tolist(),
        "cached_gt_translation": cached_gt[:3, 3].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    quantization = leader_quantization_config(cfg)
    rows = json.load(open(args.manifest, encoding="utf-8"))[:args.frames]
    if not rows:
        raise ValueError("manifest is empty")
    extra = json.load(open(Path(args.checkpoint) / "extra.json", encoding="utf-8"))
    center_t = np.asarray(extra["center_t"], dtype=np.float64)
    observer = GeometryOnlyObserver()
    records = [compare_frame(row, observer, quantization, center_t, args.lidar_cache) for row in rows]
    max_target = max(record["target_max_abs"] for record in records)
    max_gt = max(record["manifest_gt_vs_cached_gt_max_abs"] for record in records)
    checks = {
        "frames": len(records),
        "all_scan_equal": all(record["scan_equal"] for record in records),
        "all_coords_equal": all(record["coords_equal"] for record in records),
        "all_lidar_features_equal": all(record["lidar_features_equal"] for record in records),
        "max_target_abs": max_target,
        "max_manifest_gt_vs_cached_gt_abs": max_gt,
        "target_equal": max_target == 0.0,
        "manifest_gt_equal_cached_gt": max_gt <= 1e-6,
        "pass": all(record["scan_equal"] and record["coords_equal"]
                     and record["lidar_features_equal"] for record in records)
                and max_target == 0.0 and max_gt <= 1e-6,
        "quantization": quantization,
        "records": records,
    }
    print(json.dumps(checks, indent=2))
    output = HERE / "results" / "data_equivalence_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
