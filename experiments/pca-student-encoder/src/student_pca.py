"""Student that predicts PCA-64 coordinates directly from RGB.

Design: the network outputs 64 numbers per token in the *same* subspace that the closed-form
PCA basis spans, so reconstruction to 256 dimensions is a fixed linear map (no learned
adapter) and 4-bit storage behaves exactly like 4-bit PCA storage.  Only the image -> code
mapping is learned.

Supervision, all against frozen teachers:
  code         regress the PCA-64 coordinates of the cached SAM tokens (per-channel standardised)
  descriptor   match what V7 actually consumes: the 32-d normalised descriptor produced by the
               frozen observation module from (tokens, rgb, pixels), sampled at the projected
               points.  This is a much smaller and better-conditioned target than the 256-d space.
  energy       reproduce V7's appearance energy ranking over sampled candidate poses.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
DATA = Path("/data2/zhangcy/leader-v2-full-trajectory-20261001")
BASIS = Path("/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz")
V5PLAN = "/home/zhangcy/projects/leader-v5-leader-style-20261003/v5-training-plan.json"
TRAIN_DATES = ("2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11")
PATCH = 16
GRID = 64


def load_training_frame(rt, date, index, key):
    from PIL import Image
    import torch
    ds = rt.datasets[date]
    sample = dict(ds[index])
    sample.pop("T_WB", None)
    with Image.open(ds.records[index].image_path) as image:
        rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
    return {"record": (date, index, key, index), "sample": sample, "rgb": rgb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per-date", type=int, default=4000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--w-code", type=float, default=1.0)
    parser.add_argument("--w-desc", type=float, default=2.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn.functional as F
    import e2e_online as online
    from encoder import CompactEncoder, sam_preprocess

    basis = np.load(BASIS)
    mean = basis["mean"].astype(np.float32)
    projection = basis["pca64_int8"].astype(np.float32)
    print("loading V7", flush=True)
    boot = online.boot_online()
    model, rt, c = boot["model"], boot["runtime"], boot["c"]
    settings = boot["inference_method"]
    selfverify = c.selfverify
    center = rt.center.detach().double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    project = torch.from_numpy(projection).cuda()
    channel_mean = torch.from_numpy(mean).cuda()

    def to_code(tokens):
        """(256,64,64) -> (64,64,64) PCA coordinates."""
        x = tokens.permute(1, 2, 0).reshape(-1, 256) - channel_mean
        return (x @ project).t().reshape(64, GRID, GRID)

    def from_code(code):
        """(64,64,64) -> (256,64,64) reconstruction."""
        x = code.reshape(64, -1).t() @ project.t() + channel_mean
        return x.reshape(GRID, GRID, 256).permute(2, 0, 1).contiguous()

    from data.leader_nclt_manifest import ManifestLEADERDataset
    plan = json.loads(Path(V5PLAN).read_text())
    records = []
    for date in TRAIN_DATES:
        manifest = DATA / "NCLT" / date / "manifest.json"
        sam_manifest = DATA / "sam" / date / "cache_manifest.json"
        ds = ManifestLEADERDataset([manifest], DATA / "NCLT", "train", strict_frozen=False,
                                   sam_cache_manifest=sam_manifest, cache_memory_frames=0)
        rt.datasets[date] = ds
        truth = json.loads((DATA / "train-truth" / (date + ".json")).read_text())
        rng = np.random.default_rng(20261023 + abs(hash(date)) % 1000)
        pick = rng.choice(len(ds.records), size=min(args.per_date, len(ds.records)), replace=False)
        for index in sorted(pick):
            key = ds.records[index].frame_key
            stamp = key.split("/")[1]
            if stamp in truth:
                records.append((date, index, key, truth[stamp]["T_WB"]))
    print("training frames", len(records), flush=True)

    encoder = CompactEncoder("small", args.width).cuda()
    if args.resume:
        state = torch.load(args.resume, map_location="cuda")
        encoder.load_state_dict(state["encoder"])
        print("resumed step", state.get("step"), flush=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # per-channel scale of the teacher codes, measured on a few frames
    log = (out / "train.log").open("a", buffering=1)
    history = []
    step = 0
    started = time.time()
    order = np.arange(len(records))
    np.random.default_rng(11).shuffle(order)

    while step < args.steps:
        for position in order:
            if step >= args.steps:
                break
            date, index, key, gt_list = records[position]
            gt = torch.tensor(gt_list, dtype=torch.float64, device="cuda")
            try:
                boot["memo"]["velo"].clear()
                boot["memo"]["sha256"].clear()
                with torch.no_grad():
                    loaded = load_training_frame(rt, date, index, key)
                    initial = rt.initial(loaded)
                    geometry = rt.geometry(loaded, initial[1].detach().cpu())
                    frame = rt.frame(loaded, *initial, geometry)
                teacher_tokens = frame["sam"].detach()
                if tuple(teacher_tokens.shape) != (256, GRID, GRID):
                    continue
                hw = frame["hw"]
                valid_h = int(np.ceil(hw[0] / PATCH))
                valid_w = int(np.ceil(hw[1] / PATCH))
                rgb = frame["rgb"]
                with torch.no_grad():
                    teacher_code = to_code(teacher_tokens)
                    warm, _ = model.warm_start(frame["initial"], frame["source"], frame["target"])
                    warm = warm.double()
                    observed_teacher = model.observation_batch(
                        [teacher_tokens], [rgb], [frame["pixels"]], [hw])[0]
                    context_teacher = selfverify.frame_context(
                        model, frame, observed_teacher, warm, settings)

                with torch.enable_grad():
                    x = sam_preprocess(rgb, hw)
                    with torch.cuda.amp.autocast():
                        code = encoder(x)[0]
                    observed_student = model.observation_batch(
                        [from_code(code.float())], [rgb], [frame["pixels"]], [hw])[0]
                    loss_desc = 1.0 - F.cosine_similarity(
                        observed_student, observed_teacher, dim=-1).mean()

                    scale = teacher_code.float().abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                    loss_code = F.l1_loss(code.float() / scale, teacher_code.float() / scale) + \
                        (1.0 - F.cosine_similarity(code.float().permute(1, 2, 0).reshape(-1, 64),
                                                   teacher_code.float().permute(1, 2, 0).reshape(-1, 64),
                                                   dim=-1)).mean()

                    rng = np.random.default_rng(step)
                    gt_local = gt.clone()
                    gt_local[:3, 3] = gt_local[:3, 3] - center
                    poses = [gt_local, frame["initial"].double()]
                    for metres, degrees in ((0.35, 1.5), (0.9, 3.0), (2.5, 8.0)):
                        for _ in range(3):
                            delta = torch.tensor(rng.normal(0, degrees * np.pi / 180, 3),
                                                 device="cuda").ravel()
                            shift = torch.tensor(rng.normal(0, metres, 3), device="cuda")
                            pose = torch.eye(4, dtype=torch.float64, device="cuda")
                            from models.far import so3_exp
                            pose[:3, :3] = so3_exp(delta.double()) @ gt_local[:3, :3]
                            pose[:3, 3] = gt_local[:3, 3] + shift.double()
                            poses.append(pose)
                    context_student = selfverify.frame_context(
                        model, frame, observed_student, warm, settings)
                    e_t, e_s = [], []
                    for pose in poses:
                        with torch.no_grad():
                            e_t.append(float(selfverify.energy(model, pose, context_teacher)[0]))
                        e_s.append(selfverify.energy(model, pose, context_student)[0])
                    e_t = torch.tensor(e_t, dtype=e_s[0].dtype, device=e_s[0].device)
                    e_s = torch.stack(e_s)
                    spread = e_t.std().clamp_min(1e-6)
                    order_t = torch.argsort(e_t)
                    terms = []
                    for i in range(len(e_t)):
                        for j in range(i + 1, len(e_t)):
                            if e_t[order_t[j]] - e_t[order_t[i]] > 0.05:
                                terms.append(F.relu(0.025 + e_s[order_t[i]] - e_s[order_t[j]]) / spread)
                    loss_rank = torch.stack(terms).mean() if terms else e_s.sum() * 0
                    loss_energy = F.smooth_l1_loss(e_s / spread, e_t / spread)

                    loss = (args.w_code * loss_code + args.w_desc * loss_desc
                            + args.w_energy * (loss_energy + loss_rank))
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                step += 1
                entry = dict(step=step, key=key, loss=float(loss), code=float(loss_code),
                             desc=float(loss_desc), energy=float(loss_energy),
                             rank=float(loss_rank))
                history.append(entry)
                log.write(json.dumps(entry) + "\n")
                if step % 10 == 0:
                    recent = history[-10:]
                    print("step %4d loss %.4f code %.4f desc %.4f energy %.3f rank %.4f  %.0fs" % (
                        step, np.mean([r["loss"] for r in recent]), np.mean([r["code"] for r in recent]),
                        np.mean([r["desc"] for r in recent]), np.mean([r["energy"] for r in recent]),
                        np.mean([r["rank"] for r in recent]), time.time() - started), flush=True)
                    torch.save({"encoder": encoder.state_dict(), "step": step, "args": vars(args)},
                               out / "encoder.pt")
                del frame, teacher_tokens, observed_teacher, observed_student
                del context_teacher, context_student
            except RuntimeError as error:
                if "non-empty list of Tensors" in str(error) or "empty" in str(error):
                    log.write(json.dumps(dict(skip=key, error=str(error))) + "\n")
                    continue
                raise
    torch.save({"encoder": encoder.state_dict(), "step": step, "args": vars(args)},
               out / "encoder.pt")
    print("TRAIN_DONE", step, flush=True)


if __name__ == "__main__":
    main()
