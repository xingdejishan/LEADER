import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResidualProbe(nn.Module):
    def __init__(self, in_dim=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_permutation(nviews, rng):
    permutation = np.arange(len(nviews), dtype=np.int64)
    for count in np.unique(nviews):
        rows = np.flatnonzero(nviews == count)
        permutation[rows] = rows[rng.permutation(len(rows))]
    return permutation


def validate_frame(frame_id, visual_path, lidar_path):
    with np.load(visual_path) as visual_data, np.load(lidar_path) as lidar_data:
        required_visual = {"image", "valid", "view_count", "per_view_valid"}
        missing = required_visual.difference(visual_data.files)
        if missing:
            raise ValueError(f"{visual_path} is missing {sorted(missing)}")
        image = np.asarray(visual_data["image"], dtype=np.float32)
        valid = np.asarray(visual_data["valid"], dtype=bool)
        nviews = np.asarray(visual_data["view_count"], dtype=np.int64)
        per_view_valid = np.asarray(visual_data["per_view_valid"], dtype=bool)
        prediction = np.asarray(lidar_data["prediction"], dtype=np.float32)
        target = np.asarray(lidar_data["target"], dtype=np.float32)
    if image.ndim != 2 or image.shape[1] != 128:
        raise ValueError(f"{frame_id}: expected image [N,128], got {image.shape}")
    if len({len(image), len(valid), len(nviews), len(prediction), len(target)}) != 1:
        raise ValueError(f"{frame_id}: visual and LiDAR cache lengths differ")
    if per_view_valid.shape[1] != len(image):
        raise ValueError(f"{frame_id}: per_view_valid is not [6,N]")
    if not np.array_equal(nviews, per_view_valid.sum(axis=0)):
        raise ValueError(f"{frame_id}: view_count disagrees with per_view_valid")
    if not np.array_equal(valid, nviews > 0):
        raise ValueError(f"{frame_id}: valid disagrees with view_count")
    if prediction.ndim != 2 or prediction.shape[1] < 3 or target.shape != (len(image), 3):
        raise ValueError(f"{frame_id}: unexpected prediction/target shapes")
    if not np.isfinite(image).all() or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError(f"{frame_id}: cache contains non-finite values")
    if np.any((nviews < 0) | (nviews > 6)):
        raise ValueError(f"{frame_id}: view count outside six-view range")
    return image, target - prediction[:, :3], nviews


def build_cache(args, rows, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        frame_id = row["frame_id"]
        visual_path = Path(args.visual_cache) / (frame_id + ".npz")
        lidar_path = Path(args.lidar_cache) / (frame_id + ".npz")
        if not visual_path.exists() or not lidar_path.exists():
            raise FileNotFoundError(f"{frame_id}: expected {visual_path} and {lidar_path}")
        target_file = cache_dir / (frame_id + ".npz")
        if target_file.exists() and not args.rebuild_cache:
            continue
        visual, residual, nviews = validate_frame(frame_id, visual_path, lidar_path)
        permutation = stratified_permutation(
            nviews, np.random.default_rng(args.shuffle_seed + int(frame_id[-6:]))
        )
        np.savez_compressed(target_file, visual=visual, residual=residual,
                            nviews=nviews, permutation=permutation)


def load_batch(path, mode, rng, batch_size, device):
    with np.load(path) as data:
        visual = data["visual"]
        residual = data["residual"]
        permutation = data["permutation"]
        rows = rng.integers(0, len(residual), size=min(batch_size, len(residual)))
        if mode == "shuffled":
            rows = permutation[rows]
        x = torch.from_numpy(visual[rows]).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(residual[rows]).to(device=device, dtype=torch.float32)
    return x, y


def train_probe(paths, mode, seed, steps, batch_size, lr, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    probe = ResidualProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        path = paths[int(rng.integers(0, len(paths)))]
        x, y = load_batch(path, mode, rng, batch_size, device)
        loss = (probe(x) - y).pow(2).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return probe


def evaluate(probe, paths, mode, device):
    squared_errors, squared_base, counts = [], [], []
    with torch.inference_mode():
        for path in paths:
            with np.load(path) as data:
                visual = data["visual"]
                residual = data["residual"]
                nviews = data["nviews"]
                if mode == "shuffled":
                    visual = visual[data["permutation"]]
                prediction = probe(torch.from_numpy(visual).to(device=device, dtype=torch.float32))
                prediction = prediction.cpu().numpy()
            squared_errors.append(((residual - prediction) ** 2).sum(axis=1))
            squared_base.append((residual ** 2).sum(axis=1))
            counts.append(nviews)
    errors = np.concatenate(squared_errors)
    base = np.concatenate(squared_base)
    counts = np.concatenate(counts)

    def metrics(mask):
        return {
            "baseline_rmse": float(np.sqrt(base[mask].mean())),
            "probe_rmse": float(np.sqrt(errors[mask].mean())),
            "relative_reduction": float(1.0 - errors[mask].mean() / max(base[mask].mean(), 1e-12)),
            "count": int(mask.sum()),
        }

    report = {
        "all": metrics(np.ones(len(counts), dtype=bool)),
        "with_view": metrics(counts > 0),
    }
    for count in range(4):
        mask = counts == count
        if mask.any():
            report[f"{count}_views"] = metrics(mask)
    return report


def split_report(paths):
    counts = []
    for path in paths:
        with np.load(path) as data:
            counts.append(data["nviews"])
    values = np.concatenate(counts)
    histogram = np.bincount(values, minlength=7)
    return {
        "count": int(len(values)),
        "view_count_histogram": histogram.tolist(),
        "view_count_fraction": (histogram / max(len(values), 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", default="2089,2090,2091")
    parser.add_argument("--shuffle-seed", type=int, default=2089)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for residual probe")
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    if args.train_limit is not None:
        train_rows = train_rows[:args.train_limit]
    if args.val_limit is not None:
        val_rows = val_rows[:args.val_limit]
    if not train_rows or not val_rows:
        raise ValueError("Both train and val must contain at least one frame")
    checkpoint_file = Path(args.checkpoint) / "model.safetensors"
    extra_file = Path(args.checkpoint) / "extra.json"
    if not checkpoint_file.exists() or not extra_file.exists():
        raise FileNotFoundError("checkpoint must contain model.safetensors and extra.json")
    checkpoint_extra = json.load(open(extra_file, encoding="utf-8"))
    cache_dir = Path(args.cache_dir or (Path(args.output) / "cache"))
    build_cache(args, train_rows, cache_dir)
    build_cache(args, val_rows, cache_dir)
    train_paths = [cache_dir / (row["frame_id"] + ".npz") for row in train_rows]
    val_paths = [cache_dir / (row["frame_id"] + ".npz") for row in val_rows]
    device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    results = []
    for seed in seeds:
        real = train_probe(train_paths, "real", seed, args.steps, args.batch_size, args.lr, device)
        shuffled = train_probe(train_paths, "shuffled", seed, args.steps, args.batch_size, args.lr, device)
        real_report = evaluate(real, val_paths, "real", device)
        shuffled_report = evaluate(shuffled, val_paths, "shuffled", device)
        results.append({
            "seed": seed,
            "real": real_report,
            "shuffled": shuffled_report,
            "real_beats_shuffled": real_report["with_view"]["probe_rmse"] < shuffled_report["with_view"]["probe_rmse"],
            "real_beats_shuffled_all": real_report["all"]["probe_rmse"] < shuffled_report["all"]["probe_rmse"],
            "real_beats_baseline_with_view": real_report["with_view"]["probe_rmse"] < real_report["with_view"]["baseline_rmse"],
            "shuffled_beats_baseline_with_view": shuffled_report["with_view"]["probe_rmse"] < shuffled_report["with_view"]["baseline_rmse"],
        })
    output = {
        "protocol": "frozen LEADER residual probe; cached representative voxel DeDoDe/PCA128 mean over valid views",
        "checkpoint": {"model_safetensors_sha256": sha256(checkpoint_file), "extra": checkpoint_extra},
        "inputs": {"visual_cache": str(Path(args.visual_cache)), "lidar_cache": str(Path(args.lidar_cache))},
        "train_frames": len(train_rows),
        "val_frames": len(val_rows),
        "train_voxels": split_report(train_paths),
        "val_voxels": split_report(val_paths),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "results": results,
        "summary": {
            "all_seeds_real_beats_shuffled_with_view": all(item["real_beats_shuffled"] for item in results),
            "all_seeds_real_beats_baseline_with_view": all(item["real_beats_baseline_with_view"] for item in results),
            "all_seeds_shuffled_beats_baseline_with_view": all(item["shuffled_beats_baseline_with_view"] for item in results),
        },
    }
    os.makedirs(args.output, exist_ok=True)
    with open(Path(args.output) / "residual_probe.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
