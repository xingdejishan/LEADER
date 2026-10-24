"""Does the V7 refine move rotation in the right direction?

For each frame compare the rotation update the refiner actually makes (R_after R_before^T)
with the rotation correction that was needed (R_before R_gt^T).  If the axes agree, the energy
carries usable rotation information and the problem is step size or acceptance; if they are
unrelated, the appearance term simply does not see rotation.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SCORER = "/home/zhangcy/projects/r-unified-eval-20261018-v1"
SAMPLES = "/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.path.insert(0, PROJECT)
sys.path.insert(0, SCORER)

import torch  # noqa: E402
import e2e_online as online  # noqa: E402
import score_unified as scorer  # noqa: E402


def axis_angle(matrix):
    rotation = Rotation.from_matrix(matrix)
    vector = rotation.as_rotvec()
    return vector, float(np.linalg.norm(vector))


def main():
    world = scorer.load_world()
    truth = dict(zip(world["hard_keys"], world["hard_gt"]))
    samples = json.load(open(SAMPLES))["samples"]
    picked = [row for row in samples if row["group"] == "hard"][:LIMIT]

    boot = online.boot_online()
    model, rt = boot["model"], boot["runtime"]
    lookup = {record[2]: record for record in rt.records}
    center = rt.center.detach().cpu().double().numpy()

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
            observed = model.observation_batch(
                [frame["sam"]], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
            refined, _, _, info = boot["c"].selfverify.refine(
                model, frame, observed, boot["inference_method"])
        before = frame["initial"].detach().cpu().double().numpy()
        after = refined.detach().cpu().double().numpy()
        gt = truth[key]
        needed, needed_deg = axis_angle(before[:3, :3] @ gt[:3, :3].T)
        update, update_deg = axis_angle(after[:3, :3] @ before[:3, :3].T)
        done = np.degrees(np.linalg.norm(needed)) - np.degrees(np.linalg.norm(
            after[:3, :3] @ gt[:3, :3].T))
        cosine = float(np.dot(needed, update) / max(1e-12, np.linalg.norm(needed) * np.linalg.norm(update)))
        rows.append(dict(key=key, needed_deg=float(np.degrees(needed_deg)),
                         update_deg=float(np.degrees(update_deg)), cosine=cosine,
                         improved_deg=float(done), rounds=int(info["rounds"])))
        if index % 25 == 0:
            print("  %d frames" % index, flush=True)

    needed = np.array([r["needed_deg"] for r in rows])
    update = np.array([r["update_deg"] for r in rows])
    cosine = np.array([r["cosine"] for r in rows])
    improved = np.array([r["improved_deg"] for r in rows])
    big = needed > 2.0
    report = dict(
        frames=len(rows),
        needed_median=float(np.median(needed)), update_median=float(np.median(update)),
        cosine_median=float(np.median(cosine)), cosine_mean=float(cosine.mean()),
        share_cosine_positive=float((cosine > 0).mean()),
        corr_update_needed=float(np.corrcoef(update, needed)[0, 1]),
        improved_frames=int((improved > 0).sum()),
        worsened_frames=int((improved < 0).sum()),
        mean_improvement_deg=float(improved.mean()),
        big_error_frames=int(big.sum()),
        big_mean_improvement_deg=float(improved[big].mean()) if big.any() else None,
        big_mean_cosine=float(cosine[big].mean()) if big.any() else None,
        big_mean_update_deg=float(update[big].mean()) if big.any() else None,
        big_mean_needed_deg=float(needed[big].mean()) if big.any() else None,
    )
    print(json.dumps(report, indent=1))
    with open("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/rotation_update.json", "w") as f:
        json.dump(dict(report=report, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
