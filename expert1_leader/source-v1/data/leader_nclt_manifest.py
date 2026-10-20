import base64
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import MinkowskiEngine as ME
import numpy as np
import torch

from data.robotcar_sdk.python.velodyne import get_velo
from utils.pose_util import cartesian_to_polar_expansion


TRAIN_DATES = ("2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11")
DEV_DATES = ("2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26")
EXPECTED_MANIFEST_SHA256 = {
    "2012-01-22": "44d9ffe138f284e2be147997d92147a686f8255fee81fa0f770c036672589582",
    "2012-02-02": "274e27444b10a64c974ffbc4cf2ea3f159f03059c4387f382c50af0c87576833",
    "2012-02-18": "58e7ff432d94c36ca8f07a05d170c9d15084a77331f2de490c1d38ab352da9a6",
    "2012-05-11": "aedec0cb9eae24fee4b05b13bb4ebd298b886c211f0244a8793e0b2c9a36d4e0",
    "2012-02-12": "7e0369a8cf90be044650daa5f0f0e90923b674d7ca08724c122ab96e0e3a06ad",
    "2012-02-19": "3d91e68296357b64c10128f8156b14374b7b1564bc7289a174367c2e1308b458",
    "2012-03-31": "b5a8d08517269891dfa001cf6812d45cf3c63adda47d3d75b04374973e088064",
    "2012-05-26": "9442142a38347c4ea984367dcb0d9bf11504382ed6cecc414b669dce865401f8",
}
SAM_CHECKPOINT_SHA256 = "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
TRAIN_SAM_CACHE_MANIFEST_SHA256 = "92e0f127edc06c0488c9301e17f42e6a1b74f9b649523db13148d237bed4736a"
IMAGE_SIZE_HW = (616, 808)
SAM_ENCODER_SIZE = 1024
SAM_FEATURE_SIZE = 64
SAM_FEATURE_CHANNELS = 256
DEFAULT_VOXEL_SIZE = 0.2
DEFAULT_HORIZONTAL_RESOLUTION = 1024
CAM5_INTRINSICS = np.asarray(
    [
        [199.716592, 0.0, 413.180976],
        [0.0, 199.716592, 310.834312],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, date: str, relative_path: str) -> Path:
    value = str(relative_path)
    relative = PurePosixPath(value)
    if ("\\" in value or ":" in value or "\x00" in value or relative.is_absolute()
            or not relative.parts or any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError("manifest paths must be clean relative paths")
    base = root.resolve(strict=True)
    date_root = (base / date).resolve(strict=True)
    if date_root != base and base not in date_root.parents:
        raise ValueError("date path escapes the dataset root")
    target = date_root.joinpath(*relative.parts).resolve(strict=True)
    if target != base and base not in target.parents:
        raise ValueError("manifest path escapes the dataset root")
    return target


def _resize_shape(image_size_hw: Sequence[int]) -> Tuple[int, int]:
    height, width = (int(image_size_hw[0]), int(image_size_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    scale = SAM_ENCODER_SIZE / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def _decode_valid_mask(metadata: Mapping[str, object]) -> np.ndarray:
    packed = metadata.get("valid_token_mask")
    if not isinstance(packed, Mapping) or packed.get("encoding") != "base64-packed-bits-little":
        raise ValueError("SAM record has no supported valid-token mask")
    if list(packed.get("shape", ())) != [SAM_FEATURE_SIZE, SAM_FEATURE_SIZE]:
        raise ValueError("SAM valid-token mask has an unexpected shape")
    raw = base64.b64decode(str(packed.get("data", "")).encode("ascii"), validate=True)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if bits.size < SAM_FEATURE_SIZE * SAM_FEATURE_SIZE:
        raise ValueError("SAM valid-token mask is truncated")
    mask = bits[: SAM_FEATURE_SIZE * SAM_FEATURE_SIZE].reshape(SAM_FEATURE_SIZE, SAM_FEATURE_SIZE).astype(bool)
    resized_height, resized_width = (int(value) for value in metadata["resized_size_hw"])
    cell = SAM_ENCODER_SIZE / SAM_FEATURE_SIZE
    row_centers = (np.arange(SAM_FEATURE_SIZE, dtype=np.float64) + 0.5) * cell
    col_centers = (np.arange(SAM_FEATURE_SIZE, dtype=np.float64) + 0.5) * cell
    expected = (row_centers[:, None] < resized_height) & (col_centers[None, :] < resized_width)
    if not np.array_equal(mask, expected):
        raise ValueError("SAM valid-token mask disagrees with resize and padding metadata")
    return mask


@dataclass(frozen=True)
class FrameRecord:
    date: str
    timestamp_us: int
    frame_key: str
    manifest_sha256: str
    manifest_path: Path
    scan_path: Path
    scan_sha256: str
    image_path: Path
    image_sha256: str
    T_BC: np.ndarray
    T_WB: Optional[np.ndarray]


class SAMFeatureStore:
    def __init__(
        self,
        cache_manifest_path: Path,
        records: Sequence[FrameRecord],
        manifest_hashes: Mapping[str, str],
        strict: bool = True,
        expected_manifest_sha256: Optional[str] = None,
        memory_capacity: int = 16,
    ):
        self.path = Path(cache_manifest_path).resolve(strict=True)
        raw = self.path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("SAM cache manifest SHA-256 differs from its frozen value")
        self.manifest = json.loads(raw.decode("utf-8"))
        if (self.manifest.get("checkpoint_sha256") != SAM_CHECKPOINT_SHA256
                or self.manifest.get("sam_model") != "vit_l"
                or self.manifest.get("encoder_size") != SAM_ENCODER_SIZE
                or self.manifest.get("feature_channels") != SAM_FEATURE_CHANNELS
                or self.manifest.get("stored_layout") != "HWC"
                or self.manifest.get("stored_dtype") != "float16"
                or self.manifest.get("stored_feature_size_hw") != [64, 64]
                or self.manifest.get("feature_levels") != [64, 32, 16]):
            raise ValueError("SAM cache does not match the frozen ViT-L preprocessing contract")
        sources = self.manifest.get("source_manifests")
        expected_paths = {record.date: record.manifest_path.resolve(strict=True) for record in records}
        source_hashes = {}
        if not isinstance(sources, list) or len(sources) != len(expected_paths):
            raise ValueError("SAM cache source manifests do not cover the selected dates")
        for item in sources:
            source_path = Path(str(item.get("path", ""))).resolve(strict=True)
            date = str(item.get("date") or source_path.parent.name)
            if (date not in expected_paths or source_path != expected_paths[date]
                    or date in source_hashes):
                raise ValueError("SAM cache source manifest path or date differs")
            source_hashes[date] = str(item.get("sha256"))
        if source_hashes != dict(manifest_hashes):
            raise ValueError("SAM cache source manifests differ from the selected frame manifests")
        expected = {record.frame_key: record for record in records}
        frames = self.manifest.get("frames")
        if not isinstance(frames, dict) or set(frames) != set(expected):
            raise ValueError("SAM cache keys do not exactly match the requested NCLT frames")
        self.frames = frames
        self.expected = expected
        self.root = self.path.parent.resolve(strict=True)
        self.memory_capacity = max(0, int(memory_capacity))
        self.memory = OrderedDict()
        self.validated = set()
        for key, record in expected.items():
            cache_record = frames[key]
            if (cache_record.get("frame_key") != key
                    or cache_record.get("timestamp") != record.timestamp_us
                    or cache_record.get("image_sha256") != record.image_sha256
                    or cache_record.get("input_manifest_sha256") != record.manifest_sha256
                    or cache_record.get("checkpoint_sha256") != SAM_CHECKPOINT_SHA256
                    or cache_record.get("sam_model") != "vit_l"
                    or cache_record.get("resize_longest_side") != SAM_ENCODER_SIZE
                    or cache_record.get("patch_size") != 16
                    or cache_record.get("feature_size_hw") != [64, 64]
                    or cache_record.get("embedding_shape_hwc") != [64, 64, SAM_FEATURE_CHANNELS]
                    or cache_record.get("embedding_dtype") != "float16"):
                raise ValueError("SAM cache record provenance mismatch for {}".format(key))
            self._validate_metadata(key, cache_record)
        if strict and expected_manifest_sha256 is None and not records:
            raise ValueError("strict cache validation requires selected records")

    def _validate_metadata(self, key: str, record: Mapping[str, object]) -> None:
        image_size = tuple(int(value) for value in record.get("image_size_hw", ()))
        resized_size = tuple(int(value) for value in record.get("resized_size_hw", ()))
        if image_size != IMAGE_SIZE_HW or resized_size != _resize_shape(image_size):
            raise ValueError("SAM resize geometry mismatch for {}".format(key))
        if tuple(int(value) for value in record.get("sam_input_size_hw", ())) != (1024, 1024):
            raise ValueError("SAM input canvas mismatch for {}".format(key))
        expected_scale = (resized_size[1] / image_size[1], resized_size[0] / image_size[0])
        observed_scale = (float(record.get("scale_x", -1)), float(record.get("scale_y", -1)))
        if not np.allclose(expected_scale, observed_scale, rtol=0, atol=1e-12):
            raise ValueError("SAM resize scale mismatch for {}".format(key))
        if (int(record.get("pad_left", -1)) != 0 or int(record.get("pad_top", -1)) != 0
                or int(record.get("pad_right", -1)) != 1024 - resized_size[1]
                or int(record.get("pad_bottom", -1)) != 1024 - resized_size[0]):
            raise ValueError("SAM padding mismatch for {}".format(key))
        _decode_valid_mask(record)
        relative = PurePosixPath(str(record.get("cache_file", "")))
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError("SAM cache file path is not a safe relative path")
        path = (self.root.joinpath(*relative.parts)).resolve(strict=True)
        if self.root not in path.parents:
            raise ValueError("SAM cache file escapes the cache root")

    def validate_all_embeddings(self) -> None:
        for key in self.expected:
            self.load(key)
        self.memory.clear()

    def load(self, frame_key: str):
        if frame_key in self.memory:
            value = self.memory.pop(frame_key)
            self.memory[frame_key] = value
            return value
        record = self.frames[frame_key]
        relative = PurePosixPath(str(record["cache_file"]))
        path = self.root.joinpath(*relative.parts)
        array = np.load(str(path), allow_pickle=False)
        if array.shape != (64, 64, SAM_FEATURE_CHANNELS) or array.dtype != np.float16:
            raise ValueError("SAM embedding shape or dtype mismatch for {}".format(frame_key))
        if not np.isfinite(array).all():
            raise ValueError("SAM embedding contains nonfinite values for {}".format(frame_key))
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if digest != record.get("embedding_sha256"):
            raise ValueError("SAM embedding SHA-256 mismatch for {}".format(frame_key))
        tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(_decode_valid_mask(record).copy()).unsqueeze(0)
        value = (tensor, mask, record)
        self.validated.add(frame_key)
        if self.memory_capacity:
            self.memory[frame_key] = value
            while len(self.memory) > self.memory_capacity:
                self.memory.popitem(last=False)
        return value


class ManifestLEADERDataset:
    def __init__(
        self,
        manifest_paths: Sequence[Path],
        data_root: Path,
        split: str,
        sam_cache_manifest: Optional[Path] = None,
        strict_frozen: bool = True,
        expected_cache_sha256: Optional[str] = None,
        voxel_size: float = DEFAULT_VOXEL_SIZE,
        horizontal_res: int = DEFAULT_HORIZONTAL_RESOLUTION,
        min_range: float = 1.0,
        max_range: float = 100.0,
        cache_memory_frames: int = 16,
    ):
        if split not in ("train", "dev"):
            raise ValueError("split must be train or dev")
        self.split = split
        self.manifest_split = "train" if split == "train" else "test"
        self.expected_dates = TRAIN_DATES if split == "train" else DEV_DATES
        if not manifest_paths:
            raise ValueError("four frozen NCLT manifests are required")
        self.data_root = Path(data_root).resolve(strict=True)
        self.voxel_size = float(voxel_size)
        self.horizontal_res = int(horizontal_res)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        if self.voxel_size <= 0 or self.horizontal_res <= 0 or self.min_range < 0 or self.max_range <= self.min_range:
            raise ValueError("invalid LEADER voxel or range settings")
        expected_hashes = EXPECTED_MANIFEST_SHA256 if strict_frozen else None
        records = []
        hashes = {}
        seen = set()
        ordered = []
        for manifest_path in manifest_paths:
            path = Path(manifest_path).resolve(strict=True)
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            date = str(payload.get("date", ""))
            if date not in self.expected_dates or date in hashes:
                raise ValueError("unexpected or duplicate NCLT manifest date: {}".format(date))
            digest = hashlib.sha256(raw).hexdigest()
            if expected_hashes and digest != expected_hashes[date]:
                raise ValueError("frozen manifest SHA-256 mismatch for {}".format(date))
            if payload.get("split") != self.manifest_split:
                raise ValueError("manifest split differs for {}".format(date))
            rows = payload.get("frames")
            if not isinstance(rows, list):
                raise ValueError("manifest frames must be an array")
            if strict_frozen and (payload.get("complete") is not True or len(rows) != 300
                                  or payload.get("frame_count") != 300
                                  or payload.get("unique_timestamp_count") != 300):
                raise ValueError("frozen date manifest must contain exactly 300 complete frames")
            t_bc = np.asarray(payload["roi"]["T_BC"], dtype=np.float64)
            if (t_bc.shape != (4, 4) or not np.isfinite(t_bc).all()
                    or not np.allclose(t_bc[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(t_bc[:3, :3].T @ t_bc[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(t_bc[:3, :3]), 1.0, rtol=0, atol=1e-5)):
                raise ValueError("manifest T_BC is not a valid rigid transform")
            t_cb = np.linalg.inv(t_bc)
            for row in rows:
                timestamp = int(row["timestamp_us"])
                if int(row.get("camera_id", -1)) != 5 or int(row["scan"]["timestamp_us"]) != timestamp:
                    raise ValueError("manifest row is not a synchronized Cam5/scan pair")
                key = "{}/{}".format(date, timestamp)
                if key in seen:
                    raise ValueError("duplicate frame key: {}".format(key))
                seen.add(key)
                image = row["image"]
                scan = row["scan"]
                if self.split == "train":
                    gt = row.get("ground_truth")
                    if not isinstance(gt, dict) or gt.get("pose") != "T_WC":
                        raise ValueError("train manifest requires T_WC ground truth")
                    gt_path = _safe_child(self.data_root, date, gt["path"])
                    if sha256_file(gt_path) != gt.get("sha256"):
                        raise ValueError("training GT SHA-256 mismatch for {}".format(key))
                    t_wc = np.asarray(np.loadtxt(str(gt_path), dtype=np.float64), dtype=np.float64).reshape(4, 4)
                    if (not np.isfinite(t_wc).all()
                            or not np.allclose(t_wc[3], [0, 0, 0, 1], atol=1e-8)
                            or not np.allclose(t_wc[:3, :3].T @ t_wc[:3, :3], np.eye(3), atol=1e-5)
                            or not np.isclose(np.linalg.det(t_wc[:3, :3]), 1.0, rtol=0, atol=1e-5)):
                        raise ValueError("training T_WC is not a finite rigid transform for {}".format(key))
                    t_wb = t_wc @ t_cb
                else:
                    if "ground_truth" in row or "pose" in row:
                        raise ValueError("online dev manifest must not expose GT")
                    t_wb = None
                records.append(FrameRecord(
                    date=date,
                    timestamp_us=timestamp,
                    frame_key=key,
                    manifest_sha256=digest,
                    manifest_path=path,
                    scan_path=_safe_child(self.data_root, date, scan["path"]),
                    scan_sha256=str(scan["sha256"]),
                    image_path=_safe_child(self.data_root, date, image["path"]),
                    image_sha256=str(image["sha256"]),
                    T_BC=t_bc.copy(),
                    T_WB=None if t_wb is None else t_wb.astype(np.float64),
                ))
            hashes[date] = digest
            ordered.append(date)
        if strict_frozen and tuple(ordered) != self.expected_dates:
            raise ValueError("strict frozen loading requires all four manifests in frozen order")
        if not strict_frozen and (len(set(ordered)) != len(ordered)
                                  or any(date not in self.expected_dates for date in ordered)):
            raise ValueError("manifest dates are invalid")
        self.records = tuple(records)
        self.keys = tuple(record.frame_key for record in self.records)
        self.manifest_hashes = hashes
        self.sam_store = None
        if sam_cache_manifest is not None:
            if split == "train":
                expected_cache_sha = expected_cache_sha256 or (
                    TRAIN_SAM_CACHE_MANIFEST_SHA256 if strict_frozen else None
                )
            else:
                expected_cache_sha = expected_cache_sha256
            self.sam_store = SAMFeatureStore(
                Path(sam_cache_manifest),
                self.records,
                self.manifest_hashes,
                strict=strict_frozen,
                expected_manifest_sha256=expected_cache_sha,
                memory_capacity=cache_memory_frames,
            )
        self._validated_files = {}

    def __len__(self) -> int:
        return len(self.records)

    def _verify_file(self, path: Path, expected_sha: str) -> None:
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        prior = self._validated_files.get(str(path))
        if prior != fingerprint:
            if sha256_file(path) != expected_sha:
                raise ValueError("NCLT input SHA-256 mismatch: {}".format(path))
            self._validated_files[str(path)] = fingerprint

    def verify_all_input_hashes(self) -> None:
        for record in self.records:
            self._verify_file(record.scan_path, record.scan_sha256)
            self._verify_file(record.image_path, record.image_sha256)

    def validate_all_sam_embeddings(self) -> None:
        if self.sam_store is None:
            raise ValueError("this dataset has no SAM cache")
        self.sam_store.validate_all_embeddings()

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[int(index)]
        self._verify_file(record.scan_path, record.scan_sha256)
        self._verify_file(record.image_path, record.image_sha256)
        points, labels = get_velo(str(record.scan_path))
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        ranges = np.linalg.norm(points, axis=1)
        keep = (ranges > self.min_range) & (ranges < self.max_range)
        points = np.ascontiguousarray(points[keep])
        labels = np.ascontiguousarray(labels[keep])
        if not len(points):
            raise ValueError("NCLT scan contains no points within the frozen range")
        polar = cartesian_to_polar_expansion(points, self.voxel_size * self.horizontal_res)
        features = np.column_stack((polar[:, 2], polar[:, 1], labels[:, 0])).astype(np.float32)
        coordinates, features = ME.utils.sparse_quantize(
            coordinates=polar,
            features=features,
            quantization_size=self.voxel_size,
        )
        sample = {
            "frame_key": record.frame_key,
            "date": record.date,
            "timestamp_us": record.timestamp_us,
            "coords": np.asarray(coordinates, dtype=np.int32),
            "feats": np.asarray(features, dtype=np.float32),
            "points": points,
            "T_corr": np.eye(4, dtype=np.float32),
            "T_BC": np.asarray(record.T_BC, dtype=np.float32),
            "manifest_sha256": record.manifest_sha256,
            "scan_sha256": record.scan_sha256,
            "image_sha256": record.image_sha256,
        }
        if self.split == "train":
            sample["T_WB"] = np.asarray(record.T_WB, dtype=np.float32)
        if self.sam_store is not None:
            embedding, valid_mask, metadata = self.sam_store.load(record.frame_key)
            height, width = IMAGE_SIZE_HW
            resized_height, resized_width = (int(value) for value in metadata["resized_size_hw"])
            intrinsic = CAM5_INTRINSICS.astype(np.float32).copy()
            intrinsic[0, :] *= resized_width / width
            intrinsic[1, :] *= resized_height / height
            sample.update({
                "sam_features": embedding,
                "intrinsics": torch.from_numpy(intrinsic),
                "camera_from_lidar": torch.from_numpy(np.linalg.inv(record.T_BC).astype(np.float32)),
                "image_bounds": torch.tensor([resized_width, resized_height], dtype=torch.float32),
                "image_valid_mask": valid_mask,
                "sam_metadata": metadata,
            })
        return sample


def collate_leader_frames(samples: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty NCLT batch")
    has_pose = ["T_WB" in sample for sample in samples]
    if any(has_pose) and not all(has_pose):
        raise ValueError("train and dev frames cannot share a batch")
    coords = [np.asarray(sample["coords"], dtype=np.int32) for sample in samples]
    features = [np.asarray(sample["feats"], dtype=np.float32) for sample in samples]
    result = {
        "coords": ME.utils.batched_coordinates(coords),
        "feats": torch.from_numpy(np.concatenate(features, axis=0)).float(),
        "points": tuple(sample["points"] for sample in samples),
        "T_corr": torch.from_numpy(np.stack([sample["T_corr"] for sample in samples])).float(),
        "frame_keys": tuple(str(sample["frame_key"]) for sample in samples),
        "dates": tuple(str(sample["date"]) for sample in samples),
        "timestamps_us": tuple(int(sample["timestamp_us"]) for sample in samples),
    }
    if all(has_pose):
        result["T"] = torch.from_numpy(np.stack([sample["T_WB"] for sample in samples])).float()
    multimodal = ["sam_features", "intrinsics", "camera_from_lidar", "image_bounds", "image_valid_mask"]
    if any(name in sample for sample in samples for name in multimodal):
        if not all(all(name in sample for name in multimodal) for sample in samples):
            raise ValueError("every frame in a multimodal batch must have calibrated SAM features")
        result.update({
            "sam_features": torch.stack([sample["sam_features"] for sample in samples]),
            "intrinsics": torch.stack([sample["intrinsics"] for sample in samples]),
            "camera_from_lidar": torch.stack([sample["camera_from_lidar"] for sample in samples]),
            "image_bounds": torch.stack([sample["image_bounds"] for sample in samples]),
            "image_valid_mask": torch.stack([sample["image_valid_mask"] for sample in samples]),
        })
    return result
