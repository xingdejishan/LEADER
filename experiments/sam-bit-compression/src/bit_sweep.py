"""Bit-width sweep on the PCA-64 SAM code, evaluated on the 2502-frame NCLT pool.

Per frame the 256-d SAM token map is projected onto the frozen PCA-64 basis fitted on the
four training dates, quantised to B bits per coefficient with a per-channel symmetric scale,
dequantised, projected back, and fed to the frozen V7.  All variants share the same scan,
initial pose and image; only the stored code changes.
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
BITS = (8, 6, 5, 4, 3)


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


def predict(output, gpu, shard, shards):
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
    import torch
    import e2e_online as online

    basis = dict(np.load(BASIS))
    mean = basis["mean"].astype(np.float32)
    projection = basis["pca64_int8"].astype(np.float32)
    print("basis pca64 energy %.5f" % float(basis["energy_pca"]), flush=True)

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

    def restore(sam, bits):
        array = sam.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        code = (array - mean) @ projection
        levels = float(2 ** (bits - 1) - 1)
        scale = np.maximum(np.max(np.abs(code), axis=(0, 1), keepdims=True) / levels, 1e-12)
        quantised = np.clip(np.rint(code / scale), -levels, levels)
        decoded = ((quantised * scale).astype(np.float32) @ projection.T + mean).astype(np.float16)
        return torch.from_numpy(decoded.transpose(2, 0, 1).copy()).to(
            device=sam.device, dtype=sam.dtype)

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
            for bits in BITS:
                frame["sam"] = restore(sam, bits)
                tick = time.monotonic()
                try:
                    observed = model.observation_batch(
                        [frame["sam"]], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                    refined, _, _, info = boot["c"].selfverify.refine(
                        model, frame, observed, boot["inference_method"])
                except RuntimeError as error:
                    if "non-empty list of Tensors" not in str(error):
                        raise
                    variants["pca64_b%d" % bits] = dict(error=str(error),
                                                        status="empty_point_set_failure")
                    print("FRAME_VARIANT_FAILURE %s b%d" % (key, bits), flush=True)
                    continue
                torch.cuda.synchronize()
                pose = refined.detach().cpu().double().numpy().copy()
                pose[:3, 3] += center
                if not np.isfinite(pose).all():
                    raise FloatingPointError("Non-finite output pose")
                variants["pca64_b%d" % bits] = dict(
                    pose=pose.tolist(), seconds=time.monotonic() - tick,
                    rounds=int(info["rounds"]), accepted=int(info["accepted"]),
                    status=info["status"])
                del observed, refined, info
            row = dict(item, variants=variants)
            stream.write(json.dumps(row) + "\n")
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
    names = ["pca64_b%d" % bits for bits in BITS]
    rows = [row for row in rows if row["key"] in truth
            and all("pose" in row["variants"].get(name, {}) for name in names)]
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
        for name in names:
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
    reference = np.asarray([row["variants"]["pca64_b8"]["pose"] for row in rows])
    bt = scorer.pos_err(reference, gt)
    br = scorer.rot_deg(reference[:, :3, :3], gt[:, :3, :3])
    base_ok = (bt <= 1) & (br <= 2)
    rng = np.random.RandomState(20261023)
    changes = {}
    for name in names:
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
            max_extra_rotation_deg=float((r - br).max()),
            bits=int(name.split("_b")[1]),
            bytes_per_frame=int(64 * 64 * 64 * int(name.split("_b")[1]) / 8),
        )
    reference_note = ("paired deltas are against the 8-bit PCA-64 code, not against SAM FP16; "
                      "the SAM FP16 reference is in the parent experiment reports")
    report = dict(frames=len(rows), reference="pca64_b8", note=reference_note,
                  metrics=metrics, changes=changes,
                  scope="2502-frame NCLT pool; V7 frontend only; frozen visual candidates not replayed")
    write_json(Path(output) / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "score"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    output = check_environment(args.output)
    if args.mode == "predict":
        predict(output, args.gpu, args.shard, args.shards)
    else:
        score(output)


if __name__ == "__main__":
    main()
