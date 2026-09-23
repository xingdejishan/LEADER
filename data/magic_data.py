import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CalibratedImageDataset(Dataset):
    def __init__(self, lidar_dataset, dataset_root, manifest_path, image_size):
        if image_size < 64 or image_size % 32:
            raise ValueError('image_size must be at least 64 and divisible by 32')
        self.lidar_dataset = lidar_dataset
        self.dataset_root = os.path.abspath(dataset_root)
        self.image_size = image_size
        with open(manifest_path, encoding="utf-8") as stream:
            self.frames = json.load(stream)["frames"]
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.keys = [os.path.relpath(path, self.dataset_root).replace("\\", "/") for path in lidar_dataset.pcs]
        missing = [key for key in self.keys if key not in self.frames]
        if missing:
            raise ValueError(f"Missing calibrated images for {len(missing)} scans; first: {missing[0]}")

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
            image = source.convert("RGB")
            width, height = image.size
            scale = self.image_size / max(width, height)
            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize((resized_width, resized_height), resampling.BILINEAR)
            canvas = Image.new("RGB", (self.image_size, self.image_size))
            canvas.paste(image, (0, 0))
            pixels = np.asarray(canvas, dtype=np.float32).copy() / 255.0
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
            torch.from_numpy(pixels).permute(2, 0, 1),
            torch.from_numpy(intrinsic),
            torch.from_numpy(extrinsic),
            torch.tensor([resized_width, resized_height], dtype=torch.float32),
        )
