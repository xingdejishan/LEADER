import argparse
import json
import os
from pathlib import Path

import torch

from data.leader_nclt_manifest import (
    DEV_DATES,
    EXPECTED_MANIFEST_SHA256,
    SAM_CHECKPOINT_SHA256,
    ManifestLEADERDataset,
    sha256_file,
)
from tools.sam_cache_frozen import build_sam_cache


def prepare(args):
    if args.device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise RuntimeError("SAM extraction requires exactly one visible GPU")
    data_root = Path(args.data_root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    if sha256_file(checkpoint) != SAM_CHECKPOINT_SHA256:
        raise ValueError("SAM checkpoint differs from the frozen ViT-L weight")
    paths = []
    expected_keys = set()
    for date in DEV_DATES:
        path = (manifest_root / date / "manifest.json").resolve(strict=True)
        if sha256_file(path) != EXPECTED_MANIFEST_SHA256[date]:
            raise ValueError("development manifest hash differs for {}".format(date))
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("date") != date or value.get("split") != "test" or len(value.get("frames", [])) != 300:
            raise ValueError("development manifest is incomplete for {}".format(date))
        paths.append(path)
        expected_keys.update("{}/{}".format(date, frame["timestamp_us"]) for frame in value["frames"])
    if len(expected_keys) != 1200:
        raise ValueError("development frame keys are not unique")
    if output.exists():
        existing_path = output / "cache_manifest.json"
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if not set(existing.get("frames", {})).issubset(expected_keys):
                raise ValueError("existing SAM cache contains frames outside the development set")
            existing_sources = {item.get("sha256") for item in existing.get("source_manifests", [])}
            if existing_sources != {EXPECTED_MANIFEST_SHA256[date] for date in DEV_DATES}:
                raise ValueError("existing SAM cache has different source manifests")
        elif any(output.iterdir()):
            raise ValueError("SAM cache output directory has files but no cache manifest")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    cache_path = Path(build_sam_cache(
        paths, data_root, checkpoint, output,
        device=args.device,
        expected_checkpoint_sha256=SAM_CHECKPOINT_SHA256,
        batch_size=1,
        minimum_free_bytes=int(float(args.minimum_free_gib) * 1024 ** 3),
    ))
    dataset = ManifestLEADERDataset(
        paths, data_root, split="dev", sam_cache_manifest=cache_path,
        strict_frozen=True, expected_cache_sha256=sha256_file(cache_path),
        cache_memory_frames=1,
    )
    if len(dataset) != 1200:
        raise ValueError("SAM cache does not cover all development frames")
    dataset.sam_store.validate_all_embeddings()
    result = {
        "cache_manifest_path": str(cache_path),
        "cache_manifest_sha256": sha256_file(cache_path),
        "frame_count": len(dataset),
        "manifest_sha256": {date: dataset.manifest_hashes[date] for date in DEV_DATES},
        "ground_truth_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
