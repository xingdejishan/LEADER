"""96-frame smoke dataset and fixed raw-point observation preparation."""
import json
import os

import numpy as np
import torch

from projector import load_vel_lb3
from dataset_hook import PerFrameObservation, voxel_mapping


def load_native_scan(path):
    dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                      ("i", "u1"), ("l", "u1")])
    raw = np.fromfile(path, dtype=dtype)
    xyz = np.column_stack([raw[k] for k in ("x", "y", "z")]).astype(np.float32)
    xyz = xyz * 0.005 - 100.0
    intensity = raw["i"].astype(np.float32)
    keep = (np.linalg.norm(xyz, axis=1) > 1.0) & (np.linalg.norm(xyz, axis=1) < 100.0)
    return xyz[keep], intensity[keep]


class SmokeFrameDataset:
    def __init__(self, manifest, split):
        with open(manifest, encoding="utf-8") as handle:
            rows = json.load(handle)
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError("manifest has no %s rows" % split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def prepare_frame(row, observer, voxel_size, horizontal_res):
    scan, intensity = load_native_scan(row["scan"])
    observations = observer.observe(row["frame_id"], scan)
    angles = np.arctan2(scan[:, 1], scan[:, 0]).clip(-np.pi, np.pi - 1e-6)
    ranges = np.linalg.norm(scan[:, :2], axis=1, keepdims=True)
    polar = np.concatenate([
        angles[:, None] * horizontal_res / (2 * np.pi),
        ranges,
        scan[:, 2:3],
    ], axis=1).astype(np.float32)
    coords_q, index, inverse = voxel_mapping(polar, voxel_size)
    lidar_feats = np.column_stack([polar[:, 2], polar[:, 1], intensity]).astype(np.float32)
    observations["index"] = index.astype(np.int64)
    observations["inverse"] = inverse.astype(np.int64)
    observations["coords_q"] = coords_q
    tensor_obs = {
        key: torch.from_numpy(value) for key, value in observations.items()
        if isinstance(value, np.ndarray) and key in ("img_feat", "quality", "valid", "index")
    }
    return {
        "row": row,
        "scan": scan,
        "polar": polar,
        "lidar_feats": lidar_feats,
        "coords_q": observations["coords_q"],
        "obs": tensor_obs,
        "index": index,
        "inverse": inverse,
    }
