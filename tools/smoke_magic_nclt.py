import argparse
import json
import os
from pathlib import Path

import torch

from data.leader_nclt_manifest import (
    TRAIN_DATES,
    TRAIN_SAM_CACHE_MANIFEST_SHA256,
    ManifestLEADERDataset,
    collate_leader_frames,
)
from tools.magic_nclt_runtime import (
    batch_forward,
    configure_training_mode,
    load_pretrained_model,
    model_parameter_groups,
)
from tools.resource_guard import DiskGuard, GIB
from tools.train_magic_nclt import _move_optimizer_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--train-sam-cache-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("real-frame smoke requires exactly one visible GPU")
    root_guard = DiskGuard(args.base_checkpoint.resolve(strict=True).parent, 4 * GIB)
    root_guard.require()
    scratch = args.scratch.resolve(strict=True)
    scratch_guard = DiskGuard(scratch, 50 * GIB)
    scratch_guard.require(2 * GIB)
    dataset = ManifestLEADERDataset(
        [args.manifest_root / date / "manifest.json" for date in TRAIN_DATES],
        args.data_root,
        split="train",
        sam_cache_manifest=args.train_sam_cache_manifest,
        strict_frozen=True,
        expected_cache_sha256=TRAIN_SAM_CACHE_MANIFEST_SHA256,
        cache_memory_frames=1,
    )
    if (len(dataset) != 1200 or args.batch_size <= 0 or args.sample_index < 0
            or args.sample_index + args.batch_size > len(dataset)):
        raise ValueError("real-frame smoke requires a frozen training frame")
    samples = [dataset[index] for index in range(args.sample_index, args.sample_index + args.batch_size)]
    batch = collate_leader_frames(samples)
    device = torch.device("cuda:0")
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    model, checkpoint = load_pretrained_model(args.base_checkpoint, True, device)
    center = torch.as_tensor(checkpoint["center_t"], dtype=torch.float32, device=device)
    del checkpoint
    optimizer = torch.optim.Adam(model_parameter_groups(model, 1e-5, 1e-4, 1e-4))
    configure_training_mode(model)
    loss, raw, details = batch_forward(model, batch, center, device, True)
    if not torch.isfinite(loss) or not torch.isfinite(raw):
        raise FloatingPointError("real-frame smoke loss is nonfinite")
    loss.backward()
    gradient_groups = {}
    for group in optimizer.param_groups:
        values = [parameter.grad.detach().abs().sum().item()
                  for parameter in group["params"] if parameter.grad is not None]
        gradient_groups[group["name"]] = float(sum(values))
        if not values or not any(value > 0 for value in values):
            raise RuntimeError("real-frame smoke has no gradient for {}".format(group["name"]))
    optimizer.step()
    torch.cuda.synchronize(device)
    model.to("cpu")
    _move_optimizer_state(optimizer, torch.device("cpu"))
    model.to(device)
    _move_optimizer_state(optimizer, device)
    optimizer.zero_grad(set_to_none=True)
    second_loss, _, _ = batch_forward(model, batch, center, device, True)
    second_loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    checkpoint_probe = scratch / "leader-deepfusion-checkpoint-probe-{}.pt".format(os.getpid())
    if checkpoint_probe.exists():
        raise FileExistsError("checkpoint probe path already exists")
    try:
        scratch_guard.require(2 * GIB)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                   str(checkpoint_probe))
        checkpoint_bytes = checkpoint_probe.stat().st_size
    finally:
        if checkpoint_probe.exists():
            checkpoint_probe.unlink()
    result = {
        "frame_keys": [sample["frame_key"] for sample in samples],
        "sparse_inputs": int(batch["feats"].shape[0]),
        "sparse_outputs": int(details["predictions"].shape[0]),
        "trr_loss": float(loss.detach().cpu()),
        "raw_loss": float(raw.detach().cpu()),
        "post_state_migration_loss": float(second_loss.detach().cpu()),
        "gradient_abs_sum": gradient_groups,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "full_checkpoint_probe_bytes": checkpoint_bytes,
        "root_free_bytes_after": root_guard.require(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
