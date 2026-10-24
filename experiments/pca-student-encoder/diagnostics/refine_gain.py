"""How much do position and rotation each improve from the LEADER initial pose to V7's refined pose?

Uses the cached SAM features (the strongest configuration) so the measurement is about the
refiner, not about feature compression.  Reports per-frame MPE/MOE before and after.
"""
import json
import sys

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SCORER = "/home/zhangcy/projects/r-unified-eval-20261018-v1"
SAMPLES = "/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 220
sys.path.insert(0, PROJECT)
sys.path.insert(0, SCORER)

import torch  # noqa: E402
import e2e_online as online  # noqa: E402
import score_unified as scorer  # noqa: E402


def main():
    world = scorer.load_world()
    truth = dict(zip(world["hard_keys"], world["hard_gt"]))
    truth.update(zip(world["ord_keys"], world["ord_gt"]))
    samples = json.load(open(SAMPLES))["samples"]
    hard = [row for row in samples if row["group"] == "hard"][:LIMIT]
    ordinary = [row for row in samples if row["group"] == "ordinary"][:LIMIT // 2]

    boot = online.boot_online()
    model, rt = boot["model"], boot["runtime"]
    lookup = {record[2]: record for record in rt.records}
    center = rt.center.detach().cpu().double().numpy()

    rows = []
    for cohort, items in (("hard", hard), ("ordinary", ordinary)):
        for item in items:
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
            start = frame["initial"].detach().cpu().double().numpy().copy()
            start[:3, 3] += center
            end = refined.detach().cpu().double().numpy().copy()
            end[:3, 3] += center
            gt = truth[key][None]
            before_t = float(scorer.pos_err(start[None], gt)[0])
            before_r = float(scorer.rot_deg(start[None][:, :3, :3], gt[:, :3, :3])[0])
            after_t = float(scorer.pos_err(end[None], gt)[0])
            after_r = float(scorer.rot_deg(end[None][:, :3, :3], gt[:, :3, :3])[0])
            rows.append(dict(cohort=cohort, key=key, before_t=before_t, before_r=before_r,
                             after_t=after_t, after_r=after_r,
                             rounds=int(info["rounds"]), accepted=int(info["accepted"])))
            if len(rows) % 25 == 0:
                print("  %d rows" % len(rows), flush=True)

    report = {}
    for cohort in ("hard", "ordinary", "all"):
        subset = rows if cohort == "all" else [r for r in rows if r["cohort"] == cohort]
        if not subset:
            continue
        bt = np.array([r["before_t"] for r in subset])
        br = np.array([r["before_r"] for r in subset])
        at = np.array([r["after_t"] for r in subset])
        ar = np.array([r["after_r"] for r in subset])
        report[cohort] = dict(
            n=len(subset),
            before_MPE=float(bt.mean()), after_MPE=float(at.mean()),
            before_MOE=float(br.mean()), after_MOE=float(ar.mean()),
            relative_MPE_gain=float(1 - at.mean() / max(bt.mean(), 1e-9)),
            relative_MOE_gain=float(1 - ar.mean() / max(br.mean(), 1e-9)),
            median_before_MOE=float(np.median(br)), median_after_MOE=float(np.median(ar)),
            median_before_MPE=float(np.median(bt)), median_after_MPE=float(np.median(at)),
            rotation_worse=int((ar > br + 1e-9).sum()),
            rotation_better=int((ar < br - 1e-9).sum()),
            position_worse=int((at > bt + 1e-9).sum()),
            position_better=int((at < bt - 1e-9).sum()),
        )
    print(json.dumps(report, indent=1))
    with open("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/refine_gain.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
