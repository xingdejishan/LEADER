import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CalibratedImageDataset(Dataset):
    def __init__(self, lidar_dataset, dataset_root, manifest_path):
        self.lidar_dataset = lidar_dataset
        self.dataset_root = os.path.abspath(dataset_root)
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("sam_model") != "vit_l" or not manifest.get("sam_checkpoint_sha256"):
            raise ValueError('Manifest requires cached official SAM ViT-L features')
        self.frames = manifest["frames"]
        self.sam_checkpoint_sha256 = manifest["sam_checkpoint_sha256"]
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.keys = [os.path.relpath(path, self.dataset_root).replace("\\", "/") for path in lidar_dataset.pcs]
        missing = [key for key in self.keys if key not in self.frames]
        if missing:
            raise ValueError(f"Missing calibrated images for {len(missing)} scans; first: {missing[0]}")
        missing_features = [key for key in self.keys if "sam_features" not in self.frames[key]]
        if missing_features:
            raise ValueError(f"Missing SAM ViT-L features for {len(missing_features)} scans; first: {missing_features[0]}")

    def __len__(self):
        return len(self.lidar_dataset)

    def get_center_t(self):
        return self.lidar_dataset.get_center_t()

    def __getitem__(self, index):
        lidar = self.lidar_dataset[index]
        record = self.frames[self.keys[index]]
        image_path = record["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.manifest_dir, image_path)
        with Image.open(image_path) as source:
            width, height = source.size
        scale = 1024.0 / max(width, height)
        resized_width = int(width * scale + 0.5)
        resized_height = int(height * scale + 0.5)
        if record.get("resized_size") != [resized_width, resized_height]:
            raise ValueError(f"SAM resize mismatch: {self.keys[index]}")
        feature_path = record["sam_features"]
        if not os.path.isabs(feature_path):
            feature_path = os.path.join(self.manifest_dir, feature_path)
        features = np.load(feature_path, allow_pickle=False)
        if features.shape != (256, 64, 64) or not np.isfinite(features).all():
            raise ValueError(f"Invalid SAM feature map: {self.keys[index]}")
        if record.get("sam_checkpoint_sha256") != self.sam_checkpoint_sha256:
            raise ValueError(f"SAM checkpoint mismatch: {self.keys[index]}")
        intrinsic = np.asarray(record["K"], dtype=np.float32)
        extrinsic = np.asarray(record["T_camera_lidar"], dtype=np.float32)
        if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
            raise ValueError(f"Invalid calibration shape: {self.keys[index]}")
        if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
            raise ValueError(f"Nonfinite calibration: {self.keys[index]}")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError(f"Invalid focal length: {self.keys[index]}")
        if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f"Invalid extrinsic last row: {self.keys[index]}")
        rotation = extrinsic[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or np.linalg.det(rotation) < 0:
            raise ValueError(f"Invalid extrinsic rotation: {self.keys[index]}")
        intrinsic = intrinsic.copy()
        intrinsic[0, :] *= resized_width / width
        intrinsic[1, :] *= resized_height / height
        return lidar + (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(intrinsic),
            torch.from_numpy(extrinsic),
            torch.tensor([resized_width, resized_height], dtype=torch.float32),
        )
