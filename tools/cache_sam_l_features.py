import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


OFFICIAL_SAM_VIT_L_SHA256 = "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    frames = source["frames"]
    if not frames:
        raise ValueError("Manifest contains no frames")
    for key, record in frames.items():
        if set(record) != {"image", "K", "T_camera_lidar"}:
            raise ValueError(f"Unexpected or missing manifest fields: {key}")

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != OFFICIAL_SAM_VIT_L_SHA256:
        raise ValueError(f"Expected official SAM ViT-L checkpoint SHA-256, got {checkpoint_hash}")
    sam = sam_model_registry["vit_l"](checkpoint=str(checkpoint_path))
    sam.image_encoder.to(args.device).eval()
    transform = ResizeLongestSide(1024)
    output_frames = {}
    for index, (key, record) in enumerate(sorted(frames.items()), 1):
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = source_path.parent / image_path
        image_path = image_path.resolve()
        image_hash = sha256_file(image_path)
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        feature_path = output_dir / f"sam-vit-l-v1-{checkpoint_hash[:16]}-{image_hash[:16]}-{key_hash}.npy"
        with Image.open(image_path) as source_image:
            rgb = np.asarray(source_image.convert("RGB"))
        resized = transform.apply_image(rgb)
        if not feature_path.exists():
            tensor = torch.as_tensor(resized).permute(2, 0, 1).float()
            tensor = sam.preprocess(tensor).unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=args.device == "cuda"):
                features = sam.image_encoder(tensor)
            features = features[0].float().cpu().numpy().astype(np.float16)
            if features.shape != (256, 64, 64) or not np.isfinite(features).all():
                raise ValueError(f"Invalid SAM output: {key}")
            temporary_path = feature_path.with_suffix(".tmp")
            with temporary_path.open("wb") as stream:
                np.save(stream, features, allow_pickle=False)
            os.replace(temporary_path, feature_path)
        else:
            features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
            if features.shape != (256, 64, 64) or not np.isfinite(features).all():
                raise ValueError(f"Invalid existing SAM cache: {feature_path}")
        output_frames[key] = {
            "image": os.path.relpath(image_path, output_dir).replace("\\", "/"),
            "K": record["K"],
            "T_camera_lidar": record["T_camera_lidar"],
            "sam_features": feature_path.name,
            "sam_checkpoint_sha256": checkpoint_hash,
            "image_sha256": image_hash,
            "resized_size": [resized.shape[1], resized.shape[0]],
        }
        print(f"{index}/{len(frames)} {key}", flush=True)

    output = {
        "sam_model": "vit_l",
        "sam_checkpoint_sha256": checkpoint_hash,
        "source_manifest_sha256": sha256_file(source_path),
        "frame_count": len(frames),
        "frames": output_frames,
    }
    temporary_manifest = output_dir / "manifest.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_manifest, output_dir / "manifest.json")


if __name__ == "__main__":
    main()
