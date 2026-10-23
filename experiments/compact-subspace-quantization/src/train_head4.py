"""Subspace training with quantisation in the loop.

The pose-weighted objective lifts the 8-bit subspace to the PCA baseline, but it
degrades faster under 4-bit storage (1454 vs 1461), because a task-aligned basis packs
more information into fewer directions and coarser levels hurt more.  Here the 4-bit
quantiser sits inside the training graph with a straight-through estimator, so the basis
is chosen for the storage format it will actually be written in.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
DATA = Path("/data2/zhangcy/leader-v2-full-trajectory-20261001")
TRAIN_DATES = ("2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11")


def load_training_frame(rt, date, index, key):
    from PIL import Image
    import torch
    ds = rt.datasets[date]
    sample = dict(ds[index])
    sample.pop("T_WB", None)
    with Image.open(ds.records[index].image_path) as image:
        rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        rgb = rgb.permute(2, 0, 1).float() / 255.0
    return {"record": (date, index, key, index), "sample": sample, "rgb": rgb}


def fake_quant(code, levels):
    scale = code.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / levels
    q = torch.clamp(torch.round(code / scale), -levels, levels) * scale
    return code + (q - code).detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--per-date", type=int, default=3000)
    parser.add_argument("--basis", default="/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--w-frame", type=float, default=1.0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--w-rel", type=float, default=0.2)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    global torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch as _torch
    torch = _torch
    import torch.nn.functional as F
    import e2e_online as online
    from encoder import LinearHead

    archive = dict(np.load(args.basis))
    head = LinearHead(archive["mean"], archive["pca64_int8"]).cuda()
    basis_ref = torch.as_tensor(archive["pca64_int8"], device="cuda", dtype=torch.float32)
    if args.resume:
        state = torch.load(args.resume, map_location="cuda")
        head.weight.data = state["weight"].cuda()
        print("resumed", args.resume, flush=True)
    levels = float(2 ** (args.bits - 1) - 1)

    boot = online.boot_online()
    model = boot["model"]
    rt = boot["runtime"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    from data.leader_nclt_manifest import ManifestLEADERDataset
    records = []
    for date in TRAIN_DATES:
        manifest = DATA / "NCLT" / date / "manifest.json"
        sam_manifest = DATA / "sam" / date / "cache_manifest.json"
        ds = ManifestLEADERDataset([manifest], DATA / "NCLT", "train", strict_frozen=False,
                                   sam_cache_manifest=sam_manifest, cache_memory_frames=0)
        rt.datasets[date] = ds
        rng = np.random.default_rng(4242 + abs(hash(date)) % 1000)
        pick = rng.choice(len(ds.records), size=min(args.per_date, len(ds.records)), replace=False)
        records.extend((date, int(i), ds.records[int(i)].frame_key) for i in pick)
    print("training frames", len(records), "bits", args.bits, flush=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    log = (out / "train.log").open("a", buffering=1)
    order = np.arange(len(records))
    np.random.default_rng(5).shuffle(order)
    step = 0
    started = time.time()
    history = []
    while step < args.steps:
        for position in order:
            if step >= args.steps:
                break
            date, index, key = records[position]
            try:
                boot["memo"]["velo"].clear()
                boot["memo"]["sha256"].clear()
                with torch.no_grad():
                    loaded = load_training_frame(rt, date, index, key)
                    initial = rt.initial(loaded)
                    geometry = rt.geometry(loaded, initial[1].detach().cpu())
                    frame = rt.frame(loaded, *initial, geometry)
                    teacher = frame["sam"].detach()
                    if tuple(teacher.shape) != (256, 64, 64):
                        continue
                    observed_t = model.observation_batch(
                        [teacher], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    ranges = frame["points"].float().norm(dim=1)
                    weight = (ranges / ranges.mean().clamp_min(1e-6)).clamp(0.05, 20.0) ** args.alpha
                    weight = weight / weight.mean()
                with torch.enable_grad():
                    code = head(teacher)
                    quantized = fake_quant(code, levels)
                    expanded = head.expand(quantized)[0].float()
                    observed_s = model.observation_batch(
                        [expanded], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    point_loss = 1.0 - (observed_s * observed_t).sum(-1)
                    loss_desc = (weight * point_loss).mean()
                    loss_frame = point_loss.max()
                    flat_t = teacher.permute(1, 2, 0).reshape(-1, 256)
                    ref = (flat_t - head.mean) @ basis_ref
                    loss_rel = F.mse_loss(code.reshape(code.shape[1], -1).t(), ref)
                    loss = loss_desc + args.w_frame * loss_frame + args.w_rel * loss_rel
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                step += 1
                history.append(float(loss))
                log.write(json.dumps(dict(step=step, key=key, loss=float(loss),
                                          desc=float(loss_desc), rank=float(loss_frame),
                                          rel=float(loss_rel))) + "\n")
                if step % 20 == 0:
                    print("step %4d loss %.5f desc %.5f worst %.5f rel %.5f %.1fs" % (
                        step, np.mean(history[-20:]), float(loss_desc), float(loss_frame),
                        float(loss_rel), time.time() - started), flush=True)
                    torch.save({"weight": head.weight.detach().cpu(),
                                "mean": head.mean.detach().cpu(), "step": step}, out / "head.pt")
                del frame, teacher, observed_s, observed_t
            except RuntimeError as error:
                if "non-empty list of Tensors" not in str(error):
                    raise
                continue
    torch.save({"weight": head.weight.detach().cpu(), "mean": head.mean.detach().cpu(),
                "step": step}, out / "head.pt")
    print("HEAD4_DONE", step, flush=True)


if __name__ == "__main__":
    main()
