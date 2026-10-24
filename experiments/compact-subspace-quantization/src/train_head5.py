"""Learn the 64-d subspace by its effect on the pose V7 would recover.

Distance weighting is only a proxy for rotational leverage.  The quantity that matters is
how far the compressed features move the pose that V7's own optimiser converges to.  At the
reference pose T_SAM (the pose obtained with the original SAM features) the first-order
effect of a feature perturbation is

    dxi = -(H + lambda I)^-1 dg,   dg = g_compressed - g_sam,  H = dg/dz at z = 0

with z the six-dimensional twist used by V7's energy.  g is obtained by autograd, H by
central differences of g (the energy is not quadratic), and only dg carries a graph, so the
loss back-propagates into the projection without differentiating through the refiner.
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


def gradient(selfverify, model, pose, context, scales):
    value, grad, _, _, _ = selfverify.value_gradient(model, pose, context, scales, True)
    return value, grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--per-date", type=int, default=3000)
    parser.add_argument("--basis", default="/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--rot-unit-deg", type=float, default=0.5)
    parser.add_argument("--trans-unit-m", type=float, default=0.05)
    parser.add_argument("--w-desc", type=float, default=0.1)
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
    if args.resume:
        state = torch.load(args.resume, map_location="cuda")
        head.weight.data = state["weight"].cuda()
    levels = float(2 ** (args.bits - 1) - 1)

    boot = online.boot_online()
    model = boot["model"]
    rt = boot["runtime"]
    selfverify = boot["c"].selfverify
    settings = boot["inference_method"]
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
    eye = torch.eye(6, device="cuda", dtype=torch.float64)

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
                    refined, _, _, _ = selfverify.refine(model, frame, observed_t,
                                                         boot["inference_method"])
                    pose_ref = refined.detach().double()
                    warm, _ = model.warm_start(frame["initial"], frame["source"], frame["target"])
                with torch.enable_grad():
                    context_t = selfverify.frame_context(model, frame, observed_t, warm.double(),
                                                         settings)
                    scales = pose_ref.new_tensor([settings["tau_R"]] * 3
                                                 + [settings["tau_t"]] * 3)
                    move = selfverify.move
                    g_t = selfverify.value_gradient(model, pose_ref, context_t, scales, False)[1]
                    g_t = g_t.detach()
                    columns = []
                    for k in range(6):
                        column = []
                        for sign in (1.0, -1.0):
                            z = torch.zeros(6, dtype=pose_ref.dtype, device=pose_ref.device)
                            z[k] = sign * args.eps
                            local = move(pose_ref, z * scales)
                            grad = selfverify.value_gradient(model, local, context_t,
                                                             scales, False)[1]
                            column.append(grad.detach())
                        columns.append((column[0] - column[1]) / (2 * args.eps))
                    H = torch.stack(columns, dim=1).detach()

                with torch.enable_grad():
                    code = head(teacher)
                    quantized = fake_quant(code, levels)
                    expanded = head.expand(quantized)[0].float()
                    observed_s = model.observation_batch(
                        [expanded], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    context_s = selfverify.frame_context(model, frame, observed_s, warm.double(),
                                                         settings)
                    g_s = gradient(selfverify, model, pose_ref, context_s, scales)[1]
                    dg = g_s - g_t
                    H_reg = H + args.lam * eye
                    dxi = -torch.linalg.solve(H_reg, dg.reshape(6, 1)).reshape(6)
                    physical = dxi * scales.double()
                    rot = physical[:3] * 180.0 / np.pi
                    trans = physical[3:]
                    loss_pose = (rot / args.rot_unit_deg).pow(2).mean() + \
                                (trans / args.trans_unit_m).pow(2).mean()
                    loss_desc = (1.0 - (observed_s * observed_t).sum(-1)).mean()
                    loss = loss_pose + args.w_desc * loss_desc
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                step += 1
                history.append(float(loss_pose))
                log.write(json.dumps(dict(step=step, key=key, pose=float(loss_pose),
                                          desc=float(loss_desc),
                                          pred_rot_deg=float(rot.norm()),
                                          pred_trans_m=float(trans.norm()))) + "\n")
                if step % 10 == 0:
                    print("step %4d pose %.4f desc %.5f |dR| %.3f deg |dt| %.4f m  %.1fs" % (
                        step, np.mean(history[-10:]), float(loss_desc),
                        float(rot.norm()), float(trans.norm()), time.time() - started),
                        flush=True)
                    torch.save({"weight": head.weight.detach().cpu(),
                                "mean": head.mean.detach().cpu(), "step": step}, out / "head.pt")
                del frame, teacher, observed_s, observed_t, context_s, context_t
            except RuntimeError as error:
                if "non-empty list of Tensors" not in str(error):
                    raise
                continue
    torch.save({"weight": head.weight.detach().cpu(), "mean": head.mean.detach().cpu(),
                "step": step}, out / "head.pt")
    print("HEAD5_DONE", step, flush=True)


if __name__ == "__main__":
    main()
