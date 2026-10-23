"""Evaluate the compact encoder on the 2502-frame NCLT pool against the SAM baseline.

Variants per frame, all through the same frozen V7 refiner:
  fp16          cached SAM ViT-L tokens                      (baseline)
  compact_fp16  student code, FP16
  compact_int8  student code, per-channel symmetric INT8      (the storage format)

Shards are disjoint frame index ranges so several GPUs can run without the driver's
20-minute guard firing.  Scoring reads protocol truth, inference never does.
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


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(temporary), str(path))


def check_environment(output):
    if os.getuid() != 1003 or sys.prefix != "/home/zhangcy/.conda/envs/leader":
        raise RuntimeError("Use the zhangcy-owned leader environment")
    output = Path(output).resolve()
    if Path("/home/zhangcy/projects").resolve() not in output.parents:
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


def rotation_difference(a, b):
    from scipy.spatial.transform import Rotation
    return float(np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


def predict(output, gpu, shard, shards, encoder_path, arch):
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

    from PIL import Image
    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn.functional as F
    import e2e_online as online
    from encoder import CompactEncoder, sam_preprocess

    print("booting V7 gpu=%s shard=%d/%d frames=%d" % (gpu, shard, shards, len(mine)), flush=True)
    boot = online.boot_online()
    model = boot["model"]
    rt = boot["runtime"]
    torch.set_num_threads(1)
    lookup = {record[2]: record for record in rt.records}
    missing = [row["key"] for row in mine if row["key"] not in lookup]
    if missing:
        raise RuntimeError("keys missing from runtime: " + repr(missing[:5]))
    center = rt.center.detach().cpu().double().numpy()

    encoder = CompactEncoder(arch[0], arch[1]).cuda()
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
            if tuple(sam.shape) != (256, 64, 64):
                raise RuntimeError("Unexpected SAM tensor layout")

            variants = {}
            with torch.no_grad():
                x = sam_preprocess(frame["rgb"], frame["hw"])
                code = encoder(x)
                expanded = encoder.expand(code)[0].float()
                scale = code.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / 127.0
                quantized = torch.clamp(torch.round(code / scale), -127, 127) * scale
                expanded_q = encoder.expand(quantized)[0].float()
            candidates = {"fp16": sam, "compact_fp16": expanded, "compact_int8": expanded_q}
            for name, features in candidates.items():
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
                    print("FRAME_VARIANT_FAILURE %s %s" % (key, name), flush=True)
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
            row = dict(item, variants=variants)
            if len(variants) == 3 and all("pose" in value for value in variants.values()):
                row["compact_shift_m"] = float(np.linalg.norm(
                    np.asarray(variants["compact_fp16"]["pose"])[:3, 3]
                    - np.asarray(variants["fp16"]["pose"])[:3, 3]))
                row["compact_shift_deg"] = rotation_difference(
                    np.asarray(variants["compact_fp16"]["pose"]),
                    np.asarray(variants["fp16"]["pose"]))
            row["code_shape"] = list(code.shape)
            stream.write(json.dumps(row) + "\n")
            print("%d/%d %s compact_shift=%.3fm/%.2fdeg" % (
                position + 1, len(mine), key, row.get("compact_shift_m", -1.0),
                row.get("compact_shift_deg", -1.0)), flush=True)
    print("SHARD_COMPLETE", shard, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "score"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--encoder", default="")
    parser.add_argument("--backbone", default="small")
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    output = check_environment(args.output)
    if args.mode == "predict":
        predict(output, args.gpu, args.shard, args.shards, args.encoder,
                (args.backbone, args.width))
    else:
        score(output)


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
    rows = [row for row in rows if row["key"] in truth
            and all("pose" in row["variants"].get(name, {})
                    for name in ("fp16", "compact_fp16", "compact_int8"))]
    metrics = {}
    for cohort, pick in (
        ("all", lambda r: True),
        ("hard", lambda r: r["group"] == "hard"),
        ("ordinary", lambda r: r["group"] == "ordinary"),
        ("baseline_over_10m", lambda r: r["stratum"] == "baseline_over_10m"),
        ("baseline_5_to_10m", lambda r: r["stratum"] == "baseline_5_to_10m"),
        ("baseline_under_5m", lambda r: r["group"] == "hard" and r["stratum"] == "baseline_under_5m"),
    ):
        selected = [row for row in rows if pick(row)]
        if not selected:
            continue
        gt = np.stack([truth[row["key"]] for row in selected])
        values = {}
        for name in ("fp16", "compact_fp16", "compact_int8"):
            poses = np.asarray([row["variants"][name]["pose"] for row in selected])
            t = scorer.pos_err(poses, gt)
            r = scorer.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
            values[name] = dict(n=len(selected), MPE_m=float(t.mean()), MOE_deg=float(r.mean()),
                                median_position_m=float(np.median(t)),
                                median_rotation_deg=float(np.median(r)),
                                success_1m_2deg=int(((t <= 1) & (r <= 2)).sum()),
                                over_10m=int((t > 10).sum()))
        metrics[cohort] = values

    gt = np.stack([truth[row["key"]] for row in rows])
    base = np.asarray([row["variants"]["fp16"]["pose"] for row in rows])
    bt = scorer.pos_err(base, gt)
    br = scorer.rot_deg(base[:, :3, :3], gt[:, :3, :3])
    baseline_ok = (bt <= 1) & (br <= 2)
    changes = {}
    rng = np.random.RandomState(20261023)
    for name in ("compact_fp16", "compact_int8"):
        poses = np.asarray([row["variants"][name]["pose"] for row in rows])
        t = scorer.pos_err(poses, gt)
        r = scorer.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
        ok = (t <= 1) & (r <= 2)
        resample = rng.randint(0, len(rows), size=(5000, len(rows)))
        changes[name] = dict(
            delta_MPE_m=float((t - bt).mean()), delta_MOE_deg=float((r - br).mean()),
            delta_MPE_bootstrap_95pct_m=np.percentile((t - bt)[resample].mean(1), [2.5, 97.5]).tolist(),
            delta_MOE_bootstrap_95pct_deg=np.percentile((r - br)[resample].mean(1), [2.5, 97.5]).tolist(),
            success_lost=len([1 for a, b in zip(baseline_ok, ok) if a and not b]),
            success_gained=len([1 for a, b in zip(baseline_ok, ok) if b and not a]),
            new_over_10m=len([1 for a, b in zip(bt, t) if a <= 10 < b]),
            max_extra_position_m=float((t - bt).max()),
            max_extra_rotation_deg=float((r - br).max()),
        )
    report = dict(frames=len(rows), metrics=metrics, changes=changes,
                  scope="2502-frame NCLT pool; frozen V7 frontend only; frozen visual candidates not replayed")
    write_json(Path(output) / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
