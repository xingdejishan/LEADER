"""Evaluate the PCA-space student encoder on the 2502-frame NCLT pool.

The student emits 64 numbers per token in the frozen PCA-64 subspace; reconstruction is the
closed-form PCA expansion, so 4-bit storage behaves exactly like 4-bit PCA storage.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SCORER = "/home/zhangcy/projects/r-unified-eval-20261018-v1"
SAMPLES = Path("/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json")
BASIS = Path("/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz")
BITS = (8, 4)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(temporary), str(path))


def check_environment(output):
    if os.getuid() != 1003 or sys.prefix != "/home/zhangcy/.conda/envs/leader":
        raise RuntimeError("Use the zhangcy-owned leader environment")
    output = Path(output).resolve()
    if (Path("/home/zhangcy/projects").resolve() not in output.parents
            and Path("/data2/zhangcy").resolve() not in output.parents):
        raise ValueError("Output is outside own projects")
    parent = output
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_uid != os.getuid():
        raise ValueError("Output parent is not owned by zhangcy")
    free = os.statvfs(str(parent))
    if free.f_bavail * free.f_frsize < 1024 ** 3:
        raise RuntimeError("Real free space is below 1 GiB")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    return output


def predict(output, gpu, shard, shards, encoder_path, width, learned_adapter=False, sparse=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    status = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if int(status[0]) < 10240 or int(status[1]) > 20:
        raise RuntimeError("Selected GPU is not sufficiently free; no model was loaded")

    selection = json.loads(SAMPLES.read_text())["samples"]
    mine = [row for index, row in enumerate(selection) if index % shards == shard]

    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import e2e_online as online
    from encoder import CompactEncoder, sam_preprocess

    basis = np.load(BASIS)
    mean = torch.from_numpy(basis["mean"].astype(np.float32)).cuda()
    projection = torch.from_numpy(basis["pca64_int8"].astype(np.float32)).cuda()

    def expand(code):
        x = code.reshape(64, -1).t() @ projection.t() + mean
        return x.reshape(64, 64, 256).permute(2, 0, 1).contiguous()

    print("booting V7 gpu=%s shard=%d/%d frames=%d" % (gpu, shard, shards, len(mine)), flush=True)
    boot = online.boot_online()
    model, rt = boot["model"], boot["runtime"]
    torch.set_num_threads(1)
    lookup = {record[2]: record for record in rt.records}
    center = rt.center.detach().cpu().double().numpy()

    encoder = CompactEncoder("small", width).cuda()
    state = torch.load(encoder_path, map_location="cuda")
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    target = output / ("shard%d" % shard)
    target.mkdir(parents=True, exist_ok=True)
    stream_path = target / "predictions.jsonl"
    existing = []
    if stream_path.exists():
        existing = [json.loads(line) for line in stream_path.read_text().splitlines() if line.strip()]
    if [row["key"] for row in existing] != [row["key"] for row in mine[:len(existing)]]:
        raise RuntimeError("Resume prefix does not match selection")

    with stream_path.open("a" if existing else "x", buffering=1) as stream:
        for position, item in enumerate(mine[len(existing):], start=len(existing)):
            key = item["key"]
            boot["memo"]["velo"].clear()
            boot["memo"]["sha256"].clear()
            loaded = rt.load(lookup[key])
            with torch.no_grad():
                initial = rt.initial(loaded)
                geometry = rt.geometry(loaded, initial[1].detach().cpu())
                frame = rt.frame(loaded, *initial, geometry)
            sam = frame["sam"]
            with torch.no_grad():
                code = encoder(sam_preprocess(frame["rgb"], frame["hw"]))[0].float()
            variants = {"fp16": sam}
            pixels = frame["pixels"].detach().cpu().numpy().astype(np.float64)
            hw = frame["hw"]
            valid_h = int(np.ceil(hw[0] / 16.0))
            valid_w = int(np.ceil(hw[1] / 16.0))
            cells = np.zeros((64, 64), dtype=bool)
            for gx, gy in pixels * np.array([hw[1] / 1024.0, hw[0] / 1024.0])[None, :] * (64.0 / 1024.0):
                x = int(np.clip(np.floor(gx), 0, 63))
                y = int(np.clip(np.floor(gy), 0, 63))
                cells[y, x] = True
                cells[y, min(x + 1, 63)] = True
                cells[min(y + 1, 63), x] = True
                cells[min(y + 1, 63), min(x + 1, 63)] = True
            mask = torch.from_numpy(cells[:valid_h, :valid_w]).cuda()
            for bits in BITS:
                levels = float(2 ** (bits - 1) - 1)
                scale = code.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) / levels
                quantised = torch.clamp(torch.round(code / scale), -levels, levels) * scale
                if sparse:
                    kept = quantised[:, :valid_h, :valid_w] * mask
                    quantised = torch.zeros_like(quantised)
                    quantised[:, :valid_h, :valid_w] = kept
                if learned_adapter:
                    variants["student_b%d" % bits] = encoder.expand(quantised.unsqueeze(0))[0]
                else:
                    variants["student_b%d" % bits] = expand(quantised)
            for name, features in variants.items():
                frame["sam"] = features
                tick = time.monotonic()
                try:
                    observed = model.observation_batch(
                        [frame["sam"]], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    refined, _, _, info = boot["c"].selfverify.refine(
                        model, frame, observed, boot["inference_method"])
                except RuntimeError as error:
                    if "non-empty list of Tensors" not in str(error):
                        raise
                    variants[name] = dict(error=str(error), status="empty_point_set_failure")
                    continue
                torch.cuda.synchronize()
                pose = refined.detach().cpu().double().numpy().copy()
                pose[:3, 3] += center
                if not np.isfinite(pose).all():
                    raise FloatingPointError("Non-finite output pose")
                variants[name] = dict(pose=pose.tolist(), seconds=time.monotonic() - tick,
                                      rounds=int(info["rounds"]), accepted=int(info["accepted"]),
                                      status=info["status"])
                del observed, refined, info
            stream.write(json.dumps(dict(item, variants=variants)) + "\n")
            print("%d/%d %s" % (position + 1, len(mine), key), flush=True)
    print("SHARD_COMPLETE", shard, flush=True)


def score(output):
    sys.path.insert(0, SCORER)
    import score_unified as scorer
    rows = []
    for directory in sorted(Path(output).glob("shard*")):
        path = directory / "predictions.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    world = scorer.load_world()
    truth = dict(zip(world["hard_keys"], world["hard_gt"]))
    truth.update(zip(world["ord_keys"], world["ord_gt"]))
    names = ["fp16"] + ["student_b%d" % bits for bits in BITS]
    rows = [row for row in rows if row["key"] in truth
            and all("pose" in row["variants"].get(name, {}) for name in names)]
    metrics = {}
    for cohort, pick in (
        ("all", lambda r: True),
        ("hard", lambda r: r["group"] == "hard"),
        ("ordinary", lambda r: r["group"] == "ordinary"),
        ("baseline_over_10m", lambda r: r["stratum"] == "baseline_over_10m"),
        ("baseline_under_5m", lambda r: r["group"] == "hard" and r["stratum"] == "baseline_under_5m"),
    ):
        selected = [row for row in rows if pick(row)]
        if not selected:
            continue
        gt = np.stack([truth[row["key"]] for row in selected])
        values = {}
        for name in names:
            poses = np.asarray([row["variants"][name]["pose"] for row in selected])
            t = scorer.pos_err(poses, gt)
            r = scorer.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
            values[name] = dict(n=len(selected), MPE_m=float(t.mean()), MOE_deg=float(r.mean()),
                                success_1m_2deg=int(((t <= 1) & (r <= 2)).sum()),
                                over_10m=int((t > 10).sum()))
        metrics[cohort] = values

    gt = np.stack([truth[row["key"]] for row in rows])
    base = np.asarray([row["variants"]["fp16"]["pose"] for row in rows])
    bt = scorer.pos_err(base, gt)
    br = scorer.rot_deg(base[:, :3, :3], gt[:, :3, :3])
    base_ok = (bt <= 1) & (br <= 2)
    rng = np.random.RandomState(20261023)
    changes = {}
    for name in names[1:]:
        poses = np.asarray([row["variants"][name]["pose"] for row in rows])
        t = scorer.pos_err(poses, gt)
        r = scorer.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
        ok = (t <= 1) & (r <= 2)
        resample = rng.randint(0, len(rows), size=(5000, len(rows)))
        changes[name] = dict(
            delta_MPE_m=float((t - bt).mean()), delta_MOE_deg=float((r - br).mean()),
            delta_MPE_bootstrap_95pct_m=np.percentile((t - bt)[resample].mean(1), [2.5, 97.5]).tolist(),
            delta_MOE_bootstrap_95pct_deg=np.percentile((r - br)[resample].mean(1), [2.5, 97.5]).tolist(),
            success_lost=int((base_ok & ~ok).sum()), success_gained=int((~base_ok & ok).sum()),
            new_over_10m=int(((bt <= 10) & (t > 10)).sum()),
            max_extra_position_m=float((t - bt).max()),
            max_extra_rotation_deg=float((r - br).max()))
    report = dict(frames=len(rows), metrics=metrics, changes=changes,
                  scope="2502-frame NCLT pool; V7 frontend only")
    write_json(Path(output) / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "score"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--encoder", default="")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learned-adapter", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    output = check_environment(args.output)
    if args.mode == "predict":
        predict(output, args.gpu, args.shard, args.shards, args.encoder, args.width, args.learned_adapter, args.sparse)
    else:
        score(output)


if __name__ == "__main__":
    main()
