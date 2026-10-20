import base64
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.leader_nclt_manifest import (
    CAM5_INTRINSICS,
    EXPECTED_MANIFEST_SHA256,
    SAM_CHECKPOINT_SHA256,
    ManifestLEADERDataset,
    _safe_child,
    collate_leader_frames,
)
from data.robotcar_sdk.python.velodyne import get_velo
from utils.pose_util import cartesian_to_polar_expansion


RAW_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("intensity", "u1"), ("ring", "u1")])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transform(angle, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]]
    matrix[:3, 3] = translation
    return matrix


class ManifestFixture:
    def __init__(self, root, split, date, include_gt):
        self.root = Path(root)
        self.split = split
        self.date = date
        self.timestamp = 1327251800599458
        date_root = self.root / date
        scan_dir = date_root / "velodyne_sync"
        image_dir = date_root / "cam5"
        gt_dir = date_root / "groundtruth_interpolated"
        scan_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        if include_gt:
            gt_dir.mkdir(parents=True)
        self.scan_path = scan_dir / (str(self.timestamp) + ".bin")
        self.image_path = image_dir / (str(self.timestamp) + ".jpg")
        raw = np.zeros(32, dtype=RAW_DTYPE)
        raw["x"] = np.asarray([22000 + 37 * index for index in range(32)], dtype=np.uint16)
        raw["y"] = np.asarray([19900 + 19 * (index % 7) for index in range(32)], dtype=np.uint16)
        raw["z"] = np.asarray([19700 + 31 * (index % 9) for index in range(32)], dtype=np.uint16)
        raw["intensity"] = np.arange(32, dtype=np.uint8) * 7
        raw["ring"] = np.arange(32, dtype=np.uint8)
        raw["x"][0] = 20001
        raw["x"][1] = 40000
        raw.tofile(self.scan_path)
        self.image_path.write_bytes(b"synthetic-cam5-image")
        self.T_BC = _transform(0.23, [0.2, -0.1, 1.4])
        self.T_WB = _transform(-0.41, [6.0, -3.0, 0.75])
        row = {
            "timestamp_us": self.timestamp,
            "camera_id": 5,
            "image": {"path": "cam5/{}.jpg".format(self.timestamp), "sha256": _sha256(self.image_path)},
            "scan": {
                "path": "velodyne_sync/{}.bin".format(self.timestamp),
                "timestamp_us": self.timestamp,
                "sha256": _sha256(self.scan_path),
            },
        }
        if include_gt:
            self.gt_path = gt_dir / (str(self.timestamp) + "_T_WC.txt")
            T_WC = self.T_WB @ self.T_BC
            np.savetxt(self.gt_path, T_WC, fmt="%.17g")
            row["ground_truth"] = {
                "path": "groundtruth_interpolated/{}_T_WC.txt".format(self.timestamp),
                "sha256": _sha256(self.gt_path),
                "pose": "T_WC",
            }
        self.manifest_path = self.root / "manifests" / (date + ".json")
        self.manifest_path.parent.mkdir(parents=True)
        manifest = {
            "date": date,
            "split": "train" if split == "train" else "test",
            "complete": True,
            "frame_count": 1,
            "unique_timestamp_count": 1,
            "roi": {"T_BC": self.T_BC.tolist()},
            "frames": [row],
        }
        self.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.frame_key = "{}/{}".format(date, self.timestamp)

    def write_cache(self):
        cache_root = self.root / "cache"
        embedding_root = cache_root / "embeddings"
        embedding_root.mkdir(parents=True)
        array = np.zeros((64, 64, 256), dtype=np.float16)
        array[12, 19, :] = 1.5
        relative = Path("embeddings") / "frame.npy"
        np.save(cache_root / relative, array, allow_pickle=False)
        geometry = self._geometry()
        row = json.loads(self.manifest_path.read_text(encoding="utf-8"))["frames"][0]
        cache_record = {
            "frame_key": self.frame_key,
            "timestamp": self.timestamp,
            "image_path": row["image"]["path"],
            "image_sha256": row["image"]["sha256"],
            "input_manifest_path": str(self.manifest_path),
            "input_manifest_sha256": self.manifest_sha256,
            "cache_file": relative.as_posix(),
            "embedding_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "embedding_shape_hwc": [64, 64, 256],
            "embedding_dtype": "float16",
            "checkpoint_sha256": SAM_CHECKPOINT_SHA256,
            "sam_model": "vit_l",
            "resize_longest_side": 1024,
            "image_size_hw": [616, 808],
            "resized_size_hw": [781, 1024],
            "sam_input_size_hw": [1024, 1024],
            "scale_x": 1024 / 808,
            "scale_y": 781 / 616,
            "scale_longest": 1024 / 808,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 243,
            "feature_size_hw": [64, 64],
            "patch_size": 16,
            "valid_token_mask": geometry,
            "preprocessing": {"resize": "synthetic-test"},
        }
        cache_manifest = {
            "version": 1,
            "sam_model": "vit_l",
            "checkpoint_sha256": SAM_CHECKPOINT_SHA256,
            "encoder_size": 1024,
            "patch_size": 16,
            "feature_channels": 256,
            "stored_layout": "HWC",
            "stored_dtype": "float16",
            "stored_feature_size_hw": [64, 64],
            "feature_levels": [64, 32, 16],
            "source_manifests": [{
                "date": self.date,
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            }],
            "frames": {self.frame_key: cache_record},
        }
        self.cache_manifest_path = cache_root / "cache_manifest.json"
        self.cache_manifest_path.write_text(json.dumps(cache_manifest, sort_keys=True), encoding="utf-8")
        self.embedding_path = cache_root / relative

    @staticmethod
    def _geometry():
        rows = (np.arange(64, dtype=np.float64) + 0.5) * 16
        cols = (np.arange(64, dtype=np.float64) + 0.5) * 16
        mask = (rows[:, None] < 781) & (cols[None, :] < 1024)
        packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
        return {
            "shape": [64, 64],
            "encoding": "base64-packed-bits-little",
            "data": base64.b64encode(packed).decode("ascii"),
        }


class LeaderManifestDatasetTests(unittest.TestCase):
    def test_local_frozen_manifest_references_match_all_eight_hashes(self):
        repo_root = Path(__file__).resolve().parents[3]
        reference_root = repo_root / "work" / "nclt-manifests-8-reference-20260927"
        if not reference_root.is_dir():
            self.skipTest("frozen manifest reference directory is not present")
        for date, expected_sha in EXPECTED_MANIFEST_SHA256.items():
            path = reference_root / date / "manifest.json"
            raw = path.read_bytes()
            manifest = json.loads(raw.decode("utf-8"))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha, date)
            self.assertEqual(manifest["date"], date)
            self.assertEqual(len(manifest["frames"]), 300)

    def test_training_sample_uses_native_leader_voxels_pose_and_frozen_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = ManifestFixture(Path(temp) / "dataset", "train", "2012-01-22", True)
            fixture.write_cache()
            dataset = ManifestLEADERDataset(
                [fixture.manifest_path], fixture.root, "train",
                sam_cache_manifest=fixture.cache_manifest_path,
                strict_frozen=False,
            )
            sample = dataset[0]
            raw_points, raw_intensity = get_velo(str(fixture.scan_path))
            ranges = np.linalg.norm(raw_points, axis=1)
            keep = (ranges > 1.0) & (ranges < 100.0)
            points = raw_points[keep]
            label = np.asarray(raw_intensity, dtype=np.float32)[keep]
            polar = cartesian_to_polar_expansion(points, 0.2 * 1024)
            expected_features = np.column_stack((polar[:, 2], polar[:, 1], label)).astype(np.float32)
            expected_coords, expected_features = ME.utils.sparse_quantize(
                coordinates=polar, features=expected_features, quantization_size=0.2,
            )
            np.testing.assert_array_equal(sample["points"], points)
            np.testing.assert_array_equal(sample["coords"], expected_coords)
            np.testing.assert_array_equal(sample["feats"], expected_features)
            np.testing.assert_allclose(sample["T_WB"], fixture.T_WB, rtol=0, atol=2e-7)
            self.assertEqual(sample["sam_features"].shape, (256, 64, 64))
            self.assertEqual(sample["sam_features"].dtype, torch.float32)
            self.assertEqual(tuple(sample["image_valid_mask"].shape), (1, 64, 64))
            self.assertEqual(int(sample["image_valid_mask"].sum()), 49 * 64)
            self.assertEqual(sample["image_bounds"].tolist(), [1024.0, 781.0])
            expected_intrinsics = CAM5_INTRINSICS.astype(np.float32).copy()
            expected_intrinsics[0] *= 1024 / 808
            expected_intrinsics[1] *= 781 / 616
            torch.testing.assert_close(sample["intrinsics"], torch.from_numpy(expected_intrinsics))
            self.assertTrue(np.allclose(sample["camera_from_lidar"].numpy(), np.linalg.inv(fixture.T_BC), atol=1e-6))
            batch = collate_leader_frames([sample])
            self.assertEqual(tuple(batch["coords"].shape), (len(expected_coords), 4))
            self.assertEqual(tuple(batch["feats"].shape), tuple(expected_features.shape))
            self.assertEqual(tuple(batch["T"].shape), (1, 4, 4))
            self.assertEqual(batch["frame_keys"], (fixture.frame_key,))

    def test_dev_sample_and_batch_never_expose_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = ManifestFixture(Path(temp) / "dataset", "dev", "2012-02-12", False)
            fixture.write_cache()
            dataset = ManifestLEADERDataset(
                [fixture.manifest_path], fixture.root, "dev",
                sam_cache_manifest=fixture.cache_manifest_path,
                strict_frozen=False,
            )
            sample = dataset[0]
            batch = collate_leader_frames([sample])
            self.assertNotIn("T_WB", sample)
            self.assertNotIn("T", batch)
            self.assertIsNone(dataset.records[0].T_WB)

    def test_dev_manifest_with_any_gt_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = ManifestFixture(Path(temp) / "dataset", "dev", "2012-02-12", False)
            manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
            manifest["frames"][0]["ground_truth"] = {"path": "must-not-be-read.txt"}
            fixture.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not expose GT"):
                ManifestLEADERDataset([fixture.manifest_path], fixture.root, "dev", strict_frozen=False)

    def test_scan_and_embedding_content_are_hash_checked_on_read(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = ManifestFixture(Path(temp) / "dataset", "train", "2012-01-22", True)
            fixture.write_cache()
            dataset = ManifestLEADERDataset(
                [fixture.manifest_path], fixture.root, "train",
                sam_cache_manifest=fixture.cache_manifest_path,
                strict_frozen=False,
            )
            with fixture.scan_path.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "input SHA-256 mismatch"):
                dataset[0]
            fixture.scan_path.write_bytes(fixture.scan_path.read_bytes()[:-1])
            dataset = ManifestLEADERDataset(
                [fixture.manifest_path], fixture.root, "train",
                sam_cache_manifest=fixture.cache_manifest_path,
                strict_frozen=False,
            )
            array = np.load(fixture.embedding_path, allow_pickle=False)
            array[0, 0, 0] = 3.0
            np.save(fixture.embedding_path, array, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "embedding SHA-256 mismatch"):
                dataset[0]

    def test_manifest_paths_reject_windows_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "2012-01-22").mkdir()
            with self.assertRaisesRegex(ValueError, "clean relative paths"):
                _safe_child(root, "2012-01-22", "..\\outside.bin")

    def test_local_jan22_frame_matches_native_leader_preprocessing(self):
        repo_root = Path(__file__).resolve().parents[3]
        timestamp = 1327251800599458
        source_manifest_path = (repo_root / "work" / "nclt-manifests-8-reference-20260927"
                                / "2012-01-22" / "manifest.json")
        source_scan = (repo_root / "glace-local" / "data" / "scans" / "2012-01-22"
                       / "velodyne_sync" / (str(timestamp) + ".bin"))
        source_image = (repo_root / "glace-local" / "data" / "train_scene" / "train" / "rgb"
                        / (str(timestamp) + ".jpg"))
        source_pose = (repo_root / "glace-local" / "data" / "train_scene" / "train" / "poses"
                       / (str(timestamp) + ".txt"))
        if not all(path.is_file() for path in (source_manifest_path, source_scan, source_image, source_pose)):
            self.skipTest("local Jan22 reference frame is not present")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            date_root = root / "2012-01-22"
            scan_dir = date_root / "velodyne_sync"
            image_dir = date_root / "cam5"
            gt_dir = date_root / "groundtruth_interpolated"
            for directory in (scan_dir, image_dir, gt_dir):
                directory.mkdir(parents=True)
            scan_path = scan_dir / (str(timestamp) + ".bin")
            image_path = image_dir / (str(timestamp) + ".jpg")
            gt_path = gt_dir / (str(timestamp) + "_T_WC.txt")
            shutil.copyfile(source_scan, scan_path)
            shutil.copyfile(source_image, image_path)
            shutil.copyfile(source_pose, gt_path)
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            row = next(frame for frame in source_manifest["frames"]
                       if int(frame["timestamp_us"]) == timestamp)
            row["scan"]["sha256"] = _sha256(scan_path)
            row["image"]["sha256"] = _sha256(image_path)
            row["ground_truth"]["path"] = "groundtruth_interpolated/{}_T_WC.txt".format(timestamp)
            row["ground_truth"]["sha256"] = _sha256(gt_path)
            payload = {
                "date": "2012-01-22",
                "split": "train",
                "complete": True,
                "frame_count": 1,
                "unique_timestamp_count": 1,
                "roi": source_manifest["roi"],
                "frames": [row],
            }
            manifest_path = Path(temp) / "jan22-one-frame.json"
            manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            dataset = ManifestLEADERDataset([manifest_path], root, "train", strict_frozen=False)
            sample = dataset[0]
            raw_points, raw_intensity = get_velo(str(scan_path))
            ranges = np.linalg.norm(raw_points, axis=1)
            keep = (ranges > 1.0) & (ranges < 100.0)
            points = raw_points[keep]
            labels = np.asarray(raw_intensity, dtype=np.float32)[keep]
            polar = cartesian_to_polar_expansion(points, 0.2 * 1024)
            expected_features = np.column_stack((polar[:, 2], polar[:, 1], labels)).astype(np.float32)
            expected_coords, expected_features = ME.utils.sparse_quantize(
                coordinates=polar, features=expected_features, quantization_size=0.2,
            )
            np.testing.assert_array_equal(sample["points"], points)
            np.testing.assert_array_equal(sample["coords"], expected_coords)
            np.testing.assert_array_equal(sample["feats"], expected_features)
            t_wc = np.loadtxt(gt_path, dtype=np.float64).reshape(4, 4)
            t_bc = np.asarray(source_manifest["roi"]["T_BC"], dtype=np.float64)
            np.testing.assert_allclose(sample["T_WB"], t_wc @ np.linalg.inv(t_bc), rtol=0, atol=5e-6)
            self.assertTrue(np.array_equal(sample["T_corr"], np.eye(4, dtype=np.float32)))
            self.assertEqual(sample["coords"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
