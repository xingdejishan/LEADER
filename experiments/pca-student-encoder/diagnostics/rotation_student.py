"""Rotation-update diagnostic with the compact student features instead of SAM.

Same statistic as the SAM version: does the refiner's rotation update point toward the true
correction (anti-parallel to the needed rotation, so cosine close to -1) and how much of the
error does it remove.  4-bit code, matching the deployed storage.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SCORER = "/home/zhangcy/projects/r-unified-eval-20261018-v1"
SAMPLES = "/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json"
BASIS = "/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz"
CHECKPOINT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
sys.path.insert(0, PROJECT)
sys.path.insert(0, SCORER)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import e2e_online as online  # noqa: E402
import score_unified as scorer  # noqa: E402
from encoder import CompactEncoder, sam_preprocess  # noqa: E402


def axis_angle(matrix):
    return Rotation.from_matrix(matrix).as_rotvec()


def main():
    world = scorer.load_world()
    truth = dict(zip(world["hard_keys"], world["hard_gt"]))
    samples = json.load(open(SAMPLES))["samples"]
    picked = [row for row in samples if row["group"] == "hard"][:LIMIT]

    basis = np.load(BASIS)
    mean = torch.from_numpy(basis["mean"].astype(np.float32)).cuda()
    projection = torch.from_numpy(basis["pca64_int8"].astype(np.float32)).cuda()

    def expand(code):
        x = code.reshape(64, -1).t() @ projection.t() + mean
        return x.reshape(64, 64, 256).permute(2, 0, 1).contiguous()

    boot = online.boot_online()
    model, rt = boot["model"], boot["runtime"]
    lookup = {record[2]: record for record in rt.records}
    center = rt.center.detach().cpu().double().numpy()
    encoder = CompactEncoder("small", 128).cuda()
    encoder.load_state_dict(torch.load(CHECKPOINT, map_location="cuda")["encoder"])
    encoder.eval()

    rows = []
    for index, item in enumerate(picked, 1):
        key = item["key"]
        boot["memo"]["velo"].clear()
        boot["memo"]["sha256"].clear()
        loaded = rt.load(lookup[key])
        with torch.no_grad():
            initial = rt.initial(loaded)
            geometry = rt.geometry(loaded, initial[1].detach().cpu())
            frame = rt.frame(loaded, *initial, geometry)
            code = encoder(sam_preprocess(frame["rgb"], frame["hw"]))[0].float()
            levels = float(2 ** (BITS - 1) - 1)
            scale = code.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) / levels
            quantised = torch.clamp(torch.round(code / scale), -levels, levels) * scale
            frame["sam"] = expand(quantised)
            observed = model.observation_batch(
                [frame["sam"]], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
            refined, _, _, info = boot["c"].selfverify.refine(
                model, frame, observed, boot["inference_method"])
        before = frame["initial"].detach().cpu().double().numpy()
        after = refined.detach().cpu().double().numpy()
        gt = truth[key]
        needed = axis_angle(before[:3, :3] @ gt[:3, :3].T)
        update = axis_angle(after[:3, :3] @ before[:3, :3].T)
        cosine = float(np.dot(needed, update) /
                       max(1e-12, np.linalg.norm(needed) * np.linalg.norm(update)))
        error_before = float(np.degrees(np.linalg.norm(needed)))
        error_after = float(np.degrees(np.linalg.norm(axis_angle(after[:3, :3] @ gt[:3, :3].T))))
        rows.append(dict(key=key, needed_deg=error_before, update_deg=float(np.degrees(
            np.linalg.norm(update))), cosine=cosine, before=error_before, after=error_after))
        if index % 25 == 0:
            print("  %d frames" % index, flush=True)

    needed = np.array([r["needed_deg"] for r in rows])
    update = np.array([r["update_deg"] for r in rows])
    cosine = np.array([r["cosine"] for r in rows])
    before = np.array([r["before"] for r in rows])
    after = np.array([r["after"] for r in rows])
    report = dict(
        frames=len(rows), bits=BITS,
        mean_before=float(before.mean()), mean_after=float(after.mean()),
        median_before=float(np.median(before)), median_after=float(np.median(after)),
        rotation_worse=int((after > before + 1e-9).sum()),
        rotation_better=int((after < before - 1e-9).sum()),
        cosine_median=float(np.median(cosine)), cosine_mean=float(cosine.mean()),
        share_cosine_negative=float((cosine < 0).mean()),
        update_median=float(np.median(update)), needed_median=float(np.median(needed)))
    print(json.dumps(report, indent=1))
    with open("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/rotation_student.json", "w") as f:
        json.dump(dict(report=report, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
