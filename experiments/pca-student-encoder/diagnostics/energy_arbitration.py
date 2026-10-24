"""Can the appearance energy arbitrate between candidates better than the LiDAR geometry score?

For each frame with a frozen candidate pool: build V7's frame context from the cached SAM
features, evaluate the appearance energy at every candidate pose, and compare two selectors

  geometry   pick the candidate with the highest frozen fair.geometry_score (what the MoE does)
  appearance pick the candidate with the lowest V7 appearance energy (what a learned visual
             feature could provide)

against protocol truth.  Reports selection accuracy and the score margin between the best good
candidate and the best decoy, which is the quantity a targeted feature has to widen.
"""
import glob
import json
import sys

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
V = "/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
sys.path.insert(0, PROJECT)
sys.path.insert(0, "/home/zhangcy/projects/r-unified-eval-20261018-v1")

import torch  # noqa: E402
import e2e_online as online  # noqa: E402
import score_unified as scorer  # noqa: E402


def main():
    world = scorer.load_world()
    truth = dict(zip(world["hard_keys"], world["hard_gt"]))
    files = sorted(glob.glob(V + "/*.json"))[:LIMIT]
    boot = online.boot_online()
    model, rt, c = boot["model"], boot["runtime"], boot["c"]
    settings = boot["inference_method"]
    lookup = {record[2]: record for record in rt.records}
    center = rt.center.detach().double()

    rows = []
    for path in files:
        record = json.load(open(path))
        key = record["key"]
        if key not in truth or key not in lookup:
            continue
        gt = torch.tensor(truth[key], dtype=torch.float64, device="cuda")
        poses, names, scores = [], [], []
        for name, value in record["poses"].items():
            if not name.startswith("candidate_") or value.get("pose") is None:
                continue
            poses.append(torch.tensor(value["pose"], dtype=torch.float64, device="cuda"))
            names.append(name)
            scores.append((value.get("fair") or {}).get("geometry_score"))
        if len(poses) < 2:
            continue
        boot["memo"]["velo"].clear()
        boot["memo"]["sha256"].clear()
        loaded = rt.load(lookup[key])
        with torch.no_grad():
            initial = rt.initial(loaded)
            geometry = rt.geometry(loaded, initial[1].detach().cpu())
            frame = rt.frame(loaded, *initial, geometry)
            observed = model.observation_batch(
                [frame["sam"]], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
            warm, _ = model.warm_start(frame["initial"], frame["source"], frame["target"])
            context = c.selfverify.frame_context(model, frame, observed, warm.double(), settings)
            energies = []
            for pose in poses:
                local = pose.clone()
                local[:3, 3] = local[:3, 3] - center
                energies.append(float(c.selfverify.energy(model, local, context)[0]))
        errs = []
        for pose in poses:
            p = pose.detach().cpu().double().numpy()[None]
            t = float(scorer.pos_err(p, truth[key][None])[0])
            r = float(scorer.rot_deg(p[:, :3, :3], truth[key][None][:, :3, :3])[0])
            errs.append((t, r, t <= 1 and r <= 2))
        valid_scores = [s for s in scores if s is not None]
        if len(valid_scores) != len(scores):
            continue
        pick_geom = int(np.argmax(scores))
        pick_app = int(np.argmin(energies))
        rows.append(dict(key=key, errs=errs, pick_geom=pick_geom, pick_app=pick_app,
                         e=energies, scores=scores))
        print("%d %s ok_pool=%s geom_ok=%s app_ok=%s" % (
            len(rows), key, any(e[2] for e in errs), errs[pick_geom][2], errs[pick_app][2]),
            flush=True)

    pool_ok = [r for r in rows if any(e[2] for e in r["errs"])]
    geom_hit = sum(r["errs"][r["pick_geom"]][2] for r in pool_ok)
    app_hit = sum(r["errs"][r["pick_app"]][2] for r in pool_ok)
    print("\nframes audited %d, pools containing a good candidate %d" % (len(rows), len(pool_ok)))
    print("geometry selector picks a good candidate: %d/%d (%.1f%%)" % (
        geom_hit, len(pool_ok), 100.0 * geom_hit / len(pool_ok)))
    print("appearance selector picks a good candidate: %d/%d (%.1f%%)" % (
        app_hit, len(pool_ok), 100.0 * app_hit / len(pool_ok)))

    margins = []
    for r in pool_ok:
        good = [r["e"][i] for i, e in enumerate(r["errs"]) if e[2]]
        bad = [r["e"][i] for i, e in enumerate(r["errs"]) if not e[2]]
        if bad:
            margins.append(min(bad) - min(good))
    margins = np.asarray(margins)
    print("appearance margin (best good vs best decoy, larger is better): "
          "median %.4f  p10 %.4f  negative %d/%d" % (
              np.median(margins), np.percentile(margins, 10),
              int((margins < 0).sum()), len(margins)))


if __name__ == "__main__":
    main()
