"""Train the compact visual encoder against frozen V7 (pose-response distillation).

Teacher: cached SAM ViT-L tokens (64x64x256).  Student: RGB -> 64x64x64 code ->
adapter -> 64x64x256, evaluated through V7's own appearance energy, which is the
quantity the refiner actually optimises.  Two signals:

  relational   keep the token-similarity structure of the teacher
  pose-response  the energy landscape E(T) over sampled candidate poses must rank
                 the true pose above wrong poses exactly as the teacher does

No ground truth is read by the scoring process; training reads train-truth only,
on the four training dates, which are disjoint from the four evaluation dates.
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
V5PLAN = "/home/zhangcy/projects/leader-v5-leader-style-20261003/v5-training-plan.json"
TRAIN_DATES = ("2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11")
PATCH = 16
GRID = 64


def rotation_deg(a, b):
    import torch
    r = a[:3, :3] @ b[:3, :3].T
    c = ((r.trace() - 1) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(c) * 180.0 / np.pi


def perturb(pose, rng, metres, degrees):
    import torch
    from models.far import so3_exp
    delta = torch.tensor([rng.normal(0, degrees * np.pi / 180, 3)]).ravel()
    shift = torch.tensor(rng.normal(0, metres, 3))
    out = torch.eye(4, dtype=pose.dtype, device=pose.device)
    rot = so3_exp(delta.double().to(pose.device).to(pose.dtype)) @ pose[:3, :3]
    out[:3, :3] = rot
    out[:3, 3] = pose[:3, 3] + shift.to(pose.device).to(pose.dtype)
    return out


def load_training_frame(rt, date, index, key):
    """Same as runtime.load, but the split carries T_WB: drop it before it reaches V7."""
    from PIL import Image
    import torch
    ds = rt.datasets[date]
    sample = dict(ds[index])
    sample.pop("T_WB", None)
    with Image.open(ds.records[index].image_path) as image:
        rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        rgb = rgb.permute(2, 0, 1).float() / 255.0
    return {"record": (date, index, key, index), "sample": sample, "rgb": rgb}


def candidate_poses(gt, initial, rng):
    """Near / mid / far perturbations around truth, plus the real initial guess."""
    poses = [gt.clone(), initial.clone()]
    for metres, degrees in ((0.35, 1.5), (0.8, 3.0), (2.5, 8.0)):
        for _ in range(3):
            poses.append(perturb(gt, rng, metres, degrees))
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per-date", type=int, default=2000)
    parser.add_argument("--backbone", default="small")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--w-rel", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-rank", type=float, default=1.0)
    parser.add_argument("--w-cos", type=float, default=0.2)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, PROJECT)
    import torch
    import torch.nn.functional as F
    import e2e_online as online
    sys.path.insert(0, "/home/zhangcy/Documents/compact") if False else None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from encoder import CompactEncoder, sam_preprocess

    print("booting V7", flush=True)
    boot = online.boot_online()
    model = boot["model"]
    rt = boot["runtime"]
    c = boot["c"]
    settings = boot["inference_method"]
    center = rt.center.detach().double()
    selfverify = c.selfverify
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    # ---- training-date datasets, same loaders as the eval runtime ----
    from data.leader_nclt_manifest import ManifestLEADERDataset
    plan = json.loads(Path(V5PLAN).read_text())
    records = []
    for date in TRAIN_DATES:
        manifest = DATA / "NCLT" / date / "manifest.json"
        sam_manifest = DATA / "sam" / date / "cache_manifest.json"
        ds = ManifestLEADERDataset([manifest], DATA / "NCLT", "train",
                                   strict_frozen=False,
                                   sam_cache_manifest=sam_manifest,
                                   cache_memory_frames=0)
        rt.datasets[date] = ds
        truth = json.loads((DATA / "train-truth" / (date + ".json")).read_text())
        rng = np.random.default_rng(20261023 + abs(hash(date)) % 1000)
        pick = rng.choice(len(ds.records), size=min(args.per_date, len(ds.records)),
                          replace=False)
        for index in sorted(pick):
            rec = ds.records[index]
            key = rec.frame_key
            stamp = key.split("/")[1]
            if stamp not in truth:
                continue
            records.append((date, index, key, len(records), truth[stamp]["T_WB"]))
    print("training frames", len(records), flush=True)

    encoder = CompactEncoder(args.backbone, args.width).cuda()
    if args.resume:
        state = torch.load(args.resume, map_location="cuda")
        encoder.load_state_dict(state["encoder"])
        print("resumed", args.resume, "step", state.get("step"), flush=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    log = (out / "train.log").open("a", buffering=1)
    history = []
    step = 0
    started = time.time()
    order = np.arange(len(records))
    np.random.default_rng(7).shuffle(order)

    while step < args.steps:
        for position in order:
            if step >= args.steps:
                break
            date, index, key, _, gt_list = records[position]
            gt = torch.tensor(gt_list, dtype=torch.float64, device="cuda")
            try:
                boot["memo"]["velo"].clear()
                boot["memo"]["sha256"].clear()
                with torch.no_grad():
                    loaded = load_training_frame(rt, date, index, key)
                    initial = rt.initial(loaded)
                    geometry = rt.geometry(loaded, initial[1].detach().cpu())
                    frame = rt.frame(loaded, *initial, geometry)
                teacher = frame["sam"].detach()
                if tuple(teacher.shape) != (256, GRID, GRID):
                    continue
                hw = frame["hw"]
                valid_h = int(np.ceil(hw[0] / PATCH))
                valid_w = int(np.ceil(hw[1] / PATCH))
                if valid_h < 8 or valid_w < 8:
                    continue
                rgb = frame["rgb"]
                warm, _ = model.warm_start(frame["initial"], frame["source"], frame["target"])
                warm = warm.double()

                with torch.enable_grad():
                    x = sam_preprocess(rgb, hw)
                    with torch.cuda.amp.autocast():
                        code = encoder(x)
                        expanded = encoder.expand(code)
                    student = expanded[0].float()
                    observed_s = model.observation_batch(
                        [student], [rgb], [frame["pixels"]], [hw])[0]
                    with torch.no_grad():
                        observed_t = model.observation_batch(
                            [teacher], [rgb], [frame["pixels"]], [hw])[0]
                        context_t = selfverify.frame_context(
                            model, frame, observed_t, warm, settings)
                    context_s = selfverify.frame_context(
                        model, frame, observed_s, warm, settings)

                    # ---- relational distillation on valid tokens ----
                    rng = np.random.default_rng(step)
                    rows = rng.choice(valid_h, size=min(args.tokens // valid_w, valid_h),
                                      replace=False)
                    sub_t = teacher[:, rows][:, :, :valid_w].permute(1, 2, 0).reshape(-1, 256)
                    sub_s = student[:, rows][:, :, :valid_w].permute(1, 2, 0).reshape(-1, 256)
                    if len(sub_t) > args.tokens:
                        pick_tok = rng.choice(len(sub_t), size=args.tokens, replace=False)
                        sub_t, sub_s = sub_t[pick_tok], sub_s[pick_tok]
                    tn = F.normalize(sub_t, dim=-1)
                    sn = F.normalize(sub_s, dim=-1)
                    loss_rel = F.mse_loss(sn @ sn.T, tn @ tn.T)
                    loss_cos = (1.0 - (sn * tn).sum(-1)).mean()

                    # ---- pose-response distillation through the frozen energy ----
                    gt_local = gt.clone()
                    gt_local[:3, 3] = gt_local[:3, 3] - center
                    initial_local = frame["initial"].double()
                    poses = candidate_poses(gt_local, initial_local, rng)
                    e_t, e_s = [], []
                    for pose in poses:
                        with torch.no_grad():
                            e_t.append(float(selfverify.energy(model, pose, context_t)[0]))
                        e_s.append(selfverify.energy(model, pose.to(student.dtype)
                                                     if False else pose, context_s)[0])
                    e_t = torch.tensor(e_t, dtype=e_s[0].dtype, device=e_s[0].device)
                    e_s = torch.stack(e_s)
                    spread = e_t.std().clamp_min(1e-6)
                    loss_energy = F.smooth_l1_loss(e_s / spread, e_t / spread)

                    margin = 0.05
                    order_t = torch.argsort(e_t)
                    rank_terms = []
                    for i in range(len(e_t)):
                        for j in range(i + 1, len(e_t)):
                            if e_t[order_t[j]] - e_t[order_t[i]] > margin:
                                rank_terms.append(F.relu(margin * 0.5
                                                         + e_s[order_t[i]] - e_s[order_t[j]]) / spread)
                    loss_rank = torch.stack(rank_terms).mean() if rank_terms else loss_energy * 0

                    loss = (args.w_rel * loss_rel + args.w_cos * loss_cos
                            + args.w_energy * loss_energy + args.w_rank * loss_rank)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                step += 1
                entry = dict(step=step, key=key, loss=float(loss), rel=float(loss_rel),
                             cos=float(loss_cos), energy=float(loss_energy),
                             rank=float(loss_rank), e_t=[round(v, 3) for v in e_t.tolist()],
                             e_s=[round(float(v), 3) for v in e_s.detach().tolist()])
                history.append(entry)
                log.write(json.dumps(entry) + "\n")
                if step % 10 == 0:
                    recent = history[-10:]
                    print("step %4d loss %.4f rel %.4f cos %.5f energy %.4f rank %.4f  %.1fs"
                          % (step, np.mean([r["loss"] for r in recent]),
                             np.mean([r["rel"] for r in recent]),
                             np.mean([r["cos"] for r in recent]),
                             np.mean([r["energy"] for r in recent]),
                             np.mean([r["rank"] for r in recent]),
                             time.time() - started), flush=True)
                    torch.save({"encoder": encoder.state_dict(), "step": step,
                                "args": vars(args)}, out / "encoder.pt")
                del frame, teacher, observed_s, observed_t, context_s, context_t
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
