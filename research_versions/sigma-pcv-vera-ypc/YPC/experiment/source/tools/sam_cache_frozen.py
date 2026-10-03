import base64
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


FROZEN_SAM_SHA256 = "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
SAM_LONGEST_SIDE = 1024
SAM_ENCODER_SIZE = 1024
SAM_PATCH_SIZE = 16
SAM_FEATURE_CHANNELS = 256
EXPECTED_IMAGE_SIZE_HW = (616, 808)
CACHE_MANIFEST_NAME = "cache_manifest.json"


def _require_free_space(path: Path, minimum_free_bytes: int) -> None:
    if minimum_free_bytes <= 0:
        return
    available = int(shutil.disk_usage(str(path)).free)
    if available < int(minimum_free_bytes) + 16 * 1024 * 1024:
        raise RuntimeError("SAM cache stopped before the filesystem free-space floor")


def resize_longest_side_shape(image_size_hw: Sequence[int], target_length: int = SAM_LONGEST_SIDE) -> Tuple[int, int]:
    height, width = _image_size_hw(image_size_hw)
    scale = float(target_length) / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def make_frame_geometry(
    image_size_hw: Sequence[int],
    resized_size_hw: Sequence[int],
    encoder_size: int = SAM_ENCODER_SIZE,
    feature_size: int = 64,
) -> Dict[str, Any]:
    height, width = _image_size_hw(image_size_hw)
    resized_height, resized_width = _image_size_hw(resized_size_hw)
    if resized_height > encoder_size or resized_width > encoder_size:
        raise ValueError("resized image exceeds the SAM encoder canvas")
    if encoder_size % feature_size:
        raise ValueError("encoder size must be divisible by the feature size")
    if height <= 0 or width <= 0 or resized_height <= 0 or resized_width <= 0:
        raise ValueError("image sizes must be positive")
    meta = {
        "image_size_hw": [height, width],
        "resized_size_hw": [resized_height, resized_width],
        "sam_input_size_hw": [int(encoder_size), int(encoder_size)],
        "feature_size_hw": [int(feature_size), int(feature_size)],
        "scale_x": float(resized_width) / width,
        "scale_y": float(resized_height) / height,
        "scale_longest": float(encoder_size) / max(height, width),
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": int(encoder_size - resized_width),
        "pad_bottom": int(encoder_size - resized_height),
    }
    mask = _make_valid_mask(meta, (feature_size, feature_size))
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    meta["valid_token_mask"] = {
        "shape": [int(feature_size), int(feature_size)],
        "encoding": "base64-packed-bits-little",
        "data": base64.b64encode(packed).decode("ascii"),
    }
    return meta


def pixel_to_feature_grid(
    xy_pixels: Any,
    image_size: Optional[Sequence[int]],
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]] = 64,
) -> Any:
    height, width = _image_size_hw(meta["image_size_hw"])
    if image_size is not None and _image_size_hw(image_size) != (height, width):
        raise ValueError("image_size does not match cache metadata")
    feature_height, feature_width = _feature_size_hw(level)
    encoder_height, encoder_width = _image_size_hw(meta["sam_input_size_hw"])
    points = _as_numpy_points(xy_pixels)
    if points.shape[-1] != 2:
        raise ValueError("xy_pixels must end in an (x, y) coordinate pair")
    scales = np.asarray([meta["scale_x"], meta["scale_y"]], dtype=np.float64)
    cell_sizes = np.asarray([encoder_width / feature_width, encoder_height / feature_height], dtype=np.float64)
    grid = points.astype(np.float64, copy=False) * scales / cell_sizes - 0.5
    if _is_torch_tensor(xy_pixels):
        return xy_pixels.float().new_tensor(grid)
    return grid


def valid_token_mask(meta: Mapping[str, Any], level: Union[int, Sequence[int]] = 64) -> np.ndarray:
    feature_height, feature_width = _feature_size_hw(level)
    base_size = _feature_size_hw(meta.get("feature_size_hw", [64, 64]))
    packed = meta.get("valid_token_mask")
    if (feature_height, feature_width) == base_size and packed:
        if packed.get("encoding") != "base64-packed-bits-little":
            raise ValueError("unsupported valid token mask encoding")
        bits = base64.b64decode(packed["data"].encode("ascii"))
        values = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder="little")
        count = feature_height * feature_width
        if len(values) < count:
            raise ValueError("valid token mask data is truncated")
        return values[:count].reshape(feature_height, feature_width).astype(bool)
    return _make_valid_mask(meta, (feature_height, feature_width))


def bbox_to_feature_grid_bounds(
    bbox_xyxy: Sequence[float],
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]] = 64,
) -> Optional[np.ndarray]:
    box = np.asarray(_as_numpy_points(bbox_xyxy), dtype=np.float64).reshape(-1)
    if box.shape != (4,):
        raise ValueError("bbox_xyxy must contain x_min, y_min, x_max, y_max")
    if not np.isfinite(box).all():
        return None
    height, width = _image_size_hw(meta["image_size_hw"])
    x0, y0, x1, y1 = box.tolist()
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(width - 1), x1)
    y1 = min(float(height - 1), y1)
    if x1 < x0 or y1 < y0:
        return None
    corners = pixel_to_feature_grid(
        np.asarray([[x0, y0], [x1, y1]], dtype=np.float64),
        (height, width),
        meta,
        level,
    )
    return np.asarray([corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]], dtype=np.float64)


def enumerate_bbox_grid_tokens(
    bbox_xyxy: Sequence[float],
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]] = 64,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    bounds = bbox_to_feature_grid_bounds(bbox_xyxy, meta, level)
    if bounds is None:
        return np.empty((0, 2), dtype=np.int64)
    return _enumerate_bbox_grid_tokens_from_bounds(bounds, meta, level, mask)


def _enumerate_bbox_grid_tokens_from_bounds(
    bounds: np.ndarray,
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]],
    mask: Optional[np.ndarray],
) -> np.ndarray:
    feature_height, feature_width = _feature_size_hw(level)
    x0, y0, x1, y1 = bounds.tolist()
    epsilon = 1e-9
    col0 = max(0, int(math.ceil(x0 - epsilon)))
    row0 = max(0, int(math.ceil(y0 - epsilon)))
    col1 = min(feature_width - 1, int(math.floor(x1 + epsilon)))
    row1 = min(feature_height - 1, int(math.floor(y1 + epsilon)))
    if col1 < col0 or row1 < row0:
        return np.empty((0, 2), dtype=np.int64)
    rows, cols = np.mgrid[row0 : row1 + 1, col0 : col1 + 1]
    positions = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    token_mask = valid_token_mask(meta, level) if mask is None else np.asarray(mask, dtype=bool)
    if token_mask.shape != (feature_height, feature_width):
        raise ValueError("mask shape does not match the selected feature level")
    return positions[token_mask[positions[:, 0], positions[:, 1]]]


def bilinear_grid_neighbors(
    xy_pixels: Sequence[float],
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]] = 64,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    point = np.asarray(_as_numpy_points(xy_pixels), dtype=np.float64).reshape(-1)
    if point.shape != (2,):
        raise ValueError("xy_pixels must contain one (x, y) point")
    height, width = _image_size_hw(meta["image_size_hw"])
    if not np.isfinite(point).all() or point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
    feature_height, feature_width = _feature_size_hw(level)
    grid_x, grid_y = pixel_to_feature_grid(point, (height, width), meta, level).tolist()
    grid_x = min(max(grid_x, 0.0), float(feature_width - 1))
    grid_y = min(max(grid_y, 0.0), float(feature_height - 1))
    x0, y0 = int(math.floor(grid_x)), int(math.floor(grid_y))
    x1, y1 = min(x0 + 1, feature_width - 1), min(y0 + 1, feature_height - 1)
    dx, dy = grid_x - x0, grid_y - y0
    candidates = (
        ((y0, x0), (1.0 - dx) * (1.0 - dy)),
        ((y0, x1), dx * (1.0 - dy)),
        ((y1, x0), (1.0 - dx) * dy),
        ((y1, x1), dx * dy),
    )
    token_mask = valid_token_mask(meta, level) if mask is None else np.asarray(mask, dtype=bool)
    if token_mask.shape != (feature_height, feature_width):
        raise ValueError("mask shape does not match the selected feature level")
    combined: Dict[Tuple[int, int], float] = {}
    for (row, col), weight in candidates:
        if weight > 0.0 and token_mask[row, col]:
            combined[(row, col)] = combined.get((row, col), 0.0) + weight
    total = sum(combined.values())
    if total <= 0.0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
    positions = np.asarray(list(combined.keys()), dtype=np.int64)
    weights = np.asarray([weight / total for weight in combined.values()], dtype=np.float32)
    return positions, weights


def collect_bbox_tokens(
    feature_chw: Any,
    bbox_xyxy: Sequence[float],
    meta: Mapping[str, Any],
    level: Union[int, Sequence[int]] = 64,
    fallback_xy: Optional[Sequence[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Any, Any, Any, bool]:
    feature_height, feature_width = _feature_size_hw(level)
    if getattr(feature_chw, "ndim", None) != 3 or tuple(feature_chw.shape[-2:]) != (feature_height, feature_width):
        raise ValueError("feature_chw must have shape [channels, height, width] for the selected level")
    bounds = bbox_to_feature_grid_bounds(bbox_xyxy, meta, level)
    if bounds is None:
        positions = np.empty((0, 2), dtype=np.int64)
        used_fallback = False
        token_mask = None
    else:
        token_mask = valid_token_mask(meta, level) if mask is None else np.asarray(mask, dtype=bool)
        positions = _enumerate_bbox_grid_tokens_from_bounds(bounds, meta, level, token_mask)
        used_fallback = positions.shape[0] == 0
    if used_fallback:
        if fallback_xy is None:
            x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
            fallback_xy = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        positions, weights = bilinear_grid_neighbors(fallback_xy, meta, level, token_mask)
    else:
        weights = np.ones((positions.shape[0],), dtype=np.float32)
    import torch

    grid_yx = torch.as_tensor(positions, dtype=torch.long, device=feature_chw.device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=feature_chw.device)
    if positions.shape[0] == 0:
        tokens = feature_chw.new_empty((0, feature_chw.shape[0]))
    else:
        tokens = feature_chw[:, grid_yx[:, 0], grid_yx[:, 1]].transpose(0, 1).contiguous()
    return tokens, weight_tensor, grid_yx, used_fallback


def feature_pyramid(base_chw: Any) -> Dict[int, Any]:
    import torch.nn.functional as functional

    if getattr(base_chw, "ndim", None) != 3 or tuple(base_chw.shape[-2:]) != (64, 64):
        raise ValueError("base_chw must have shape [channels, 64, 64]")
    base_fp32 = base_chw.float().unsqueeze(0)
    level32 = functional.avg_pool2d(base_fp32, kernel_size=2, stride=2)[0].to(dtype=base_chw.dtype)
    level16 = functional.avg_pool2d(base_fp32, kernel_size=4, stride=4)[0].to(dtype=base_chw.dtype)
    return {64: base_chw, 32: level32, 16: level16}


def _rgb_normalization_values(value: Any) -> List[float]:
    values = value.detach().cpu().reshape(-1)
    if values.numel() != 3:
        raise ValueError("SAM normalization buffers must contain exactly three RGB values")
    return [float(item) for item in values.tolist()]


def build_sam_cache(
    manifest_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    data_root: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    cache_dir: Union[str, os.PathLike],
    device: str = "cuda:0",
    expected_checkpoint_sha256: Optional[str] = FROZEN_SAM_SHA256,
    batch_size: int = 1,
    expected_image_size_hw: Optional[Sequence[int]] = EXPECTED_IMAGE_SIZE_HW,
    minimum_free_bytes: int = 0,
) -> str:
    import torch
    from PIL import Image
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = [Path(path) for path in manifest_path] if isinstance(manifest_path, (list, tuple)) else [Path(manifest_path)]
    if not paths:
        raise ValueError("at least one input manifest is required")
    cache_root = Path(cache_dir)
    _require_free_space(cache_root.parent, minimum_free_bytes)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_root / CACHE_MANIFEST_NAME
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_sha.lower() != expected_checkpoint_sha256.lower():
        raise ValueError("SAM checkpoint SHA-256 does not match the frozen weight")
    input_manifests = [_read_manifest(path) for path in paths]
    records = []
    source_records = []
    for input_path, input_manifest, input_sha in input_manifests:
        source_records.append({"path": str(input_path), "sha256": input_sha})
        records.extend(_frame_specs(input_manifest, input_path, input_sha))
    if not records:
        raise ValueError("input manifests contain no frames")
    unique_keys = [record["frame_key"] for record in records]
    if len(unique_keys) != len(set(unique_keys)):
        raise ValueError("frame keys must be unique across input manifests")
    if expected_image_size_hw is not None:
        expected_image_size_hw = _image_size_hw(expected_image_size_hw)
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("existing cache was built with a different SAM checkpoint")
        if cache_manifest.get("sam_model") != "vit_l" or cache_manifest.get("encoder_size") != SAM_ENCODER_SIZE:
            raise ValueError("existing cache metadata is incompatible")
    else:
        cache_manifest = {
            "version": 1,
            "sam_model": "vit_l",
            "checkpoint_sha256": checkpoint_sha,
            "encoder_size": SAM_ENCODER_SIZE,
            "patch_size": SAM_PATCH_SIZE,
            "feature_channels": SAM_FEATURE_CHANNELS,
            "stored_layout": "HWC",
            "stored_dtype": "float16",
            "stored_feature_size_hw": [64, 64],
            "source_manifests": [],
            "frames": {},
        }
    cache_manifest.setdefault("source_manifests", [])
    known_sources = {item["sha256"] for item in cache_manifest["source_manifests"]}
    for item in source_records:
        if item["sha256"] not in known_sources:
            cache_manifest["source_manifests"].append(item)
            known_sources.add(item["sha256"])
    cache_manifest.setdefault("frames", {})
    model = sam_model_registry["vit_l"](checkpoint=str(checkpoint_path))
    model.eval()
    model.requires_grad_(False)
    transform = ResizeLongestSide(SAM_LONGEST_SIDE)
    encoder_size = int(model.image_encoder.img_size)
    if encoder_size != SAM_ENCODER_SIZE:
        raise ValueError("Meta SAM ViT-L encoder must use a 1024-pixel input canvas")
    if tuple(model.pixel_mean.shape[-2:]) != (1, 1) or tuple(model.pixel_std.shape[-2:]) != (1, 1):
        raise ValueError("unexpected SAM image normalization buffers")
    preprocessing = {
        "resize": "segment_anything.ResizeLongestSide(1024)",
        "normalization": "Meta SAM Sam.preprocess",
        "pixel_mean_rgb": _rgb_normalization_values(model.pixel_mean),
        "pixel_std_rgb": _rgb_normalization_values(model.pixel_std),
        "input_range": "RGB float32 [0,255]",
        "padding": "right_and_bottom_zero_after_normalization",
    }
    if cache_manifest.get("preprocessing") not in (None, preprocessing):
        raise ValueError("existing cache preprocessing metadata is incompatible")
    model.to(device)
    cache_manifest["preprocessing"] = preprocessing
    cache_manifest["feature_levels"] = [64, 32, 16]
    pending = []
    prepared = []
    for record in records:
        _require_free_space(cache_root, minimum_free_bytes)
        image_path = _resolve_image_path(Path(data_root), record["image_path"], record.get("date"))
        image_sha = _sha256_file(image_path)
        if record.get("image_sha256") and image_sha.lower() != record["image_sha256"].lower():
            raise ValueError("input image SHA-256 mismatch for {}".format(record["frame_key"]))
        prior = cache_manifest["frames"].get(record["frame_key"])
        cache_file = _cache_file_for_key(cache_root, record["frame_key"])
        if prior and prior.get("image_sha256") == image_sha and prior.get("cache_file") == str(cache_file.relative_to(cache_root)):
            if _cached_embedding_is_valid(cache_root, prior):
                continue
        with Image.open(str(image_path)) as image:
            rgb = image.convert("RGB")
            image_size_hw = (int(rgb.height), int(rgb.width))
            if expected_image_size_hw is not None and image_size_hw != expected_image_size_hw:
                raise ValueError("unexpected image size {} for {}".format(image_size_hw, record["frame_key"]))
            image_array = np.asarray(rgb, dtype=np.uint8)
        resized = transform.apply_image(image_array)
        resized_size_hw = (int(resized.shape[0]), int(resized.shape[1]))
        official_shape = tuple(transform.get_preprocess_shape(*image_size_hw, SAM_LONGEST_SIDE))
        if resized_size_hw != official_shape:
            raise RuntimeError("official ResizeLongestSide output shape mismatch")
        tensor = torch.as_tensor(resized, device=device).permute(2, 0, 1).float()
        prepared.append(model.preprocess(tensor))
        pending.append({
            **record,
            "image_path": record["image_path"],
            "image_sha256": image_sha,
            "image_size_hw": image_size_hw,
            "resized_size_hw": resized_size_hw,
            "cache_file": str(cache_file.relative_to(cache_root)),
            "input_manifest_sha256": record["input_manifest_sha256"],
        })
        if len(pending) == batch_size:
            _require_free_space(cache_root, minimum_free_bytes)
            _encode_pending(model, pending, prepared, cache_manifest, cache_root, checkpoint_sha, encoder_size, cache_manifest_path)
            pending = []
            prepared = []
    if pending:
        _require_free_space(cache_root, minimum_free_bytes)
        _encode_pending(model, pending, prepared, cache_manifest, cache_root, checkpoint_sha, encoder_size, cache_manifest_path)
    return str(cache_manifest_path)


def load_sam_cache(cache_manifest_path: Union[str, os.PathLike], frame_key: str) -> Tuple[Any, Dict[str, Any]]:
    import torch

    manifest_path = Path(cache_manifest_path)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        record = cache_manifest["frames"][str(frame_key)]
    except KeyError as exc:
        raise KeyError("unknown SAM cache frame key: {}".format(frame_key)) from exc
    path = manifest_path.parent / record["cache_file"]
    array = np.load(str(path), allow_pickle=False)
    if array.shape != (64, 64, SAM_FEATURE_CHANNELS) or array.dtype != np.float16:
        raise ValueError("cached SAM embedding has an unexpected shape or dtype")
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if digest != record.get("embedding_sha256"):
        raise ValueError("cached SAM embedding SHA-256 mismatch")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, record


def _image_size_hw(value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("image size must be (height, width)")
    return int(value[0]), int(value[1])


def _feature_size_hw(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        size = int(value)
        return size, size
    return _image_size_hw(value)


def _make_valid_mask(meta: Mapping[str, Any], size_hw: Sequence[int]) -> np.ndarray:
    feature_height, feature_width = _feature_size_hw(size_hw)
    resized_height, resized_width = _image_size_hw(meta["resized_size_hw"])
    encoder_height, encoder_width = _image_size_hw(meta["sam_input_size_hw"])
    rows = (np.arange(feature_height, dtype=np.float64) + 0.5) * (encoder_height / feature_height)
    cols = (np.arange(feature_width, dtype=np.float64) + 0.5) * (encoder_width / feature_width)
    return (rows[:, None] < resized_height) & (cols[None, :] < resized_width)


def _as_numpy_points(points: Any) -> np.ndarray:
    if _is_torch_tensor(points):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def _sha256_file(path: Union[str, os.PathLike]) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file_if_exists(path: Path) -> Optional[str]:
    return _sha256_file(path) if path.is_file() else None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_manifest(path: Path) -> Tuple[Path, Dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input manifest must be a JSON object")
    return path.resolve(), value, hashlib.sha256(raw).hexdigest()


def _frame_specs(manifest: Mapping[str, Any], manifest_path: Path, manifest_sha: str) -> List[Dict[str, Any]]:
    frame_items = manifest.get("frames")
    if isinstance(frame_items, dict):
        frame_items = [dict(value, frame_key=value.get("frame_key", key)) for key, value in frame_items.items()]
    if not isinstance(frame_items, list):
        raise ValueError("input manifest must contain a frames array or object")
    date = str(manifest.get("date", "")).strip()
    output = []
    for index, item in enumerate(frame_items):
        if not isinstance(item, dict):
            raise ValueError("each input manifest frame must be an object")
        image_value = item.get("image_path", item.get("image"))
        image_path = image_value.get("path") if isinstance(image_value, dict) else image_value
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("each input manifest frame must contain image or image_path")
        timestamp = item.get("timestamp", item.get("timestamp_us", item.get("actual_timestamp")))
        frame_key = item.get("frame_key")
        if frame_key is None:
            identity = str(timestamp) if timestamp is not None else Path(image_path).stem
            frame_key = "{}/{}".format(date, identity) if date else identity
        output.append({
            "frame_key": str(frame_key),
            "date": date or None,
            "timestamp": None if timestamp is None else int(timestamp),
            "image_path": image_path,
            "image_sha256": item.get(
                "image_sha256",
                item.get("sha256", image_value.get("sha256") if isinstance(image_value, dict) else None),
            ),
            "input_manifest_path": str(manifest_path),
            "input_manifest_sha256": manifest_sha,
            "manifest_index": int(index),
        })
    return output


def _resolve_image_path(data_root: Path, image_path: str, date: Optional[str] = None) -> Path:
    root = data_root.resolve()
    candidate = Path(image_path)
    if candidate.is_absolute():
        candidates = [candidate.resolve()]
    else:
        relative_candidates = [candidate]
        parts = candidate.parts
        if parts and parts[0].lower() == "cam5":
            relative_candidates.append(Path("images") / "Cam5" / Path(*parts[1:]))
        date_root = root / date if date and root.name != date else root
        roots = [date_root] if date else []
        roots.append(root)
        candidates = [(base / relative).resolve() for base in roots for relative in relative_candidates]
    contained = []
    for resolved in candidates:
        try:
            within_root = os.path.commonpath([str(root), str(resolved)]) == str(root)
        except ValueError:
            within_root = False
        if not within_root:
            raise ValueError("manifest image path escapes data_root")
        contained.append(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(str(contained[0]))


def _cache_file_for_key(cache_root: Path, frame_key: str) -> Path:
    key_hash = hashlib.sha256(frame_key.encode("utf-8")).hexdigest()
    return cache_root / "frames" / (key_hash + ".npy")


def _cached_embedding_is_valid(cache_root: Path, record: Mapping[str, Any]) -> bool:
    path = cache_root / record["cache_file"]
    if not path.is_file():
        return False
    try:
        array = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        array.shape == (64, 64, SAM_FEATURE_CHANNELS)
        and array.dtype == np.float16
        and hashlib.sha256(array.tobytes(order="C")).hexdigest() == record.get("embedding_sha256")
    )


def _encode_pending(
    model: Any,
    pending: Sequence[Mapping[str, Any]],
    prepared: Sequence[Any],
    cache_manifest: Dict[str, Any],
    cache_root: Path,
    checkpoint_sha: str,
    encoder_size: int,
    cache_manifest_path: Path,
) -> None:
    import torch

    batch = torch.stack(list(prepared), dim=0)
    with torch.inference_mode():
        embeddings = model.image_encoder(batch)
    if tuple(embeddings.shape[1:]) != (SAM_FEATURE_CHANNELS, 64, 64):
        raise RuntimeError("Meta SAM ViT-L returned an unexpected embedding shape")
    for item, embedding in zip(pending, embeddings):
        array = embedding.detach().to(device="cpu", dtype=torch.float16).permute(1, 2, 0).contiguous().numpy()
        geometry = make_frame_geometry(item["image_size_hw"], item["resized_size_hw"], encoder_size, 64)
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        cache_path = cache_root / item["cache_file"]
        _atomic_numpy(cache_path, array)
        record = {
            "frame_key": item["frame_key"],
            "timestamp": item["timestamp"],
            "image_path": item["image_path"],
            "image_sha256": item["image_sha256"],
            "input_manifest_path": item["input_manifest_path"],
            "input_manifest_sha256": item["input_manifest_sha256"],
            "cache_file": item["cache_file"],
            "embedding_sha256": digest,
            "embedding_shape_hwc": [64, 64, SAM_FEATURE_CHANNELS],
            "embedding_dtype": "float16",
            "checkpoint_sha256": checkpoint_sha,
            "sam_model": "vit_l",
            "resize_longest_side": SAM_LONGEST_SIDE,
            "image_size_hw": geometry["image_size_hw"],
            "resized_size_hw": geometry["resized_size_hw"],
            "sam_input_size_hw": geometry["sam_input_size_hw"],
            "scale_x": geometry["scale_x"],
            "scale_y": geometry["scale_y"],
            "scale_longest": geometry["scale_longest"],
            "pad_left": geometry["pad_left"],
            "pad_top": geometry["pad_top"],
            "pad_right": geometry["pad_right"],
            "pad_bottom": geometry["pad_bottom"],
            "feature_size_hw": geometry["feature_size_hw"],
            "patch_size": SAM_PATCH_SIZE,
            "valid_token_mask": geometry["valid_token_mask"],
            "preprocessing": cache_manifest["preprocessing"],
        }
        cache_manifest["frames"][item["frame_key"]] = record
    _atomic_json(cache_manifest_path, cache_manifest)
