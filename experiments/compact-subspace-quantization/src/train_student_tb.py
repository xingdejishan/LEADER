"""Train the image student to predict the task-trained 64-d basis, not PCA coordinates.

Same recipe that already works on PCA coordinates (the student sees the image, learns the
code, is supervised on the 32-d observed descriptor V7 consumes), but the target coordinate
system is the pose-weighted, rotated task basis that scored 1472 frames at 4 bits.  The
adapter is fixed to that basis, so the network only has to produce the 64 numbers.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per-date", type=int, default=2000)
    parser.add_argument("--head", default="/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/alpha2/head.pt")
    parser.add_argument("--rotation", default="/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/pca_heads/rotation.npy")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--w-frame", type=float, default=1.0)
    parser.add_argument("--w-code", type=float, default=0.5)
    parser.add_argument("--backbone", default="small")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn.functional as F
    import e2e_online as online
    from encoder import CompactEncoder, expand_with, sam_preprocess

    head_state = torch.load(args.head, map_location="cpu")
    Q = torch.as_tensor(np.load(args.rotation).astype(np.float32))
    weight = (head_state["weight"].float() @ Q).cuda()
    mean = head_state["mean"].float().cuda()
    print("task basis", tuple(weight.shape), flush=True)

    encoder = CompactEncoder(args.backbone, args.width).cuda()
    if args.resume:
        encoder.load_state_dict(torch.load(args.resume, map_location="cuda")["encoder"])
        print("resumed", args.resume, flush=True)

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
        rng = np.random.default_rng(909 + abs(hash(date)) % 1000)
        pick = rng.choice(len(ds.records), size=min(args.per_date, len(ds.records)), replace=False)
        records.extend((date, int(i), ds.records[int(i)].frame_key) for i in pick)
    print("training frames", len(records), flush=True)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    log = (out / "train.log").open("a", buffering=1)
    order = np.arange(len(records))
    np.random.default_rng(3).shuffle(order)
    step, started, history = 0, time.time(), []
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
                    weight_p = (ranges / ranges.mean().clamp_min(1e-6)).clamp(0.05, 20.0) ** args.alpha
                    weight_p = weight_p / weight_p.mean()
                    flat_t = teacher.permute(1, 2, 0).reshape(-1, 256)
                    target = ((flat_t - mean) @ weight).t().reshape(1, 64, 64, 64)
                with torch.enable_grad():
                    x = sam_preprocess(frame["rgb"], frame["hw"])
                    code = encoder(x)
                    expanded = expand_with(code, weight, mean)[0].float()
                    observed_s = model.observation_batch(
                        [expanded], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    point_loss = 1.0 - (observed_s * observed_t).sum(-1)
                    loss_desc = (weight_p * point_loss).mean()
                    loss_frame = point_loss.max()
                    loss_code = F.mse_loss(code, target)
                    loss = loss_desc + args.w_frame * loss_frame + args.w_code * loss_code
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    optimizer.step()
                step += 1
                history.append(float(loss))
                log.write(json.dumps(dict(step=step, key=key, loss=float(loss),
                                          desc=float(loss_desc), worst=float(loss_frame),
                                          code=float(loss_code))) + "\n")
                if step % 20 == 0:
                    print("step %4d loss %.5f desc %.5f worst %.5f code %.5f %.1fs" % (
                        step, np.mean(history[-20:]), float(loss_desc), float(loss_frame),
                        float(loss_code), time.time() - started), flush=True)
                    torch.save({"encoder": encoder.state_dict(), "weight": weight.cpu(),
                                "mean": mean.cpu(), "step": step}, out / "student.pt")
                del frame, teacher, observed_s, observed_t
            except RuntimeError as error:
                if "non-empty list of Tensors" not in str(error):
                    raise
                continue
    torch.save({"encoder": encoder.state_dict(), "weight": weight.cpu(),
                "mean": mean.cpu(), "step": step}, out / "student.pt")
    print("STUDENT_TB_DONE", step, flush=True)


if __name__ == "__main__":
    main()
