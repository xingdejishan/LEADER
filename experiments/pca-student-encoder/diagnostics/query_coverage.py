"""How much of the 64x64 feature grid does the downstream actually read?

V7 samples the token map at the pixels where the frame's LiDAR points project, and that pixel
set is fixed for the whole refinement (the observed descriptor is computed once).  This
measures, per frame, the fraction of grid cells whose 2x2 bilinear neighbourhood is touched by
any sampled pixel - the fraction a query-aligned store would have to keep.
"""
import json
import sys

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SAMPLES = "/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.path.insert(0, PROJECT)

import torch  # noqa: E402
import e2e_online as online  # noqa: E402


def main():
    samples = json.load(open(SAMPLES))["samples"]
    picked = [row for row in samples if row["group"] == "hard"][:LIMIT]
    boot = online.boot_online()
    rt = boot["runtime"]
    lookup = {record[2]: record for record in rt.records}

    fractions, pixel_counts, touched_rows = [], [], []
    for index, item in enumerate(picked, 1):
        key = item["key"]
        boot["memo"]["velo"].clear()
        boot["memo"]["sha256"].clear()
        loaded = rt.load(lookup[key])
        with torch.no_grad():
            initial = rt.initial(loaded)
            geometry = rt.geometry(loaded, initial[1].detach().cpu())
            frame = rt.frame(loaded, *initial, geometry)
        pixels = frame["pixels"].detach().cpu().numpy().astype(np.float64)
        hw = frame["hw"]
        scale = np.array([hw[1] / 1024.0, hw[0] / 1024.0])
        resized = pixels * scale[None, :]
        grid = resized * 64.0 / 1024.0
        valid_h = int(np.ceil(hw[0] / 16.0))
        valid_w = int(np.ceil(hw[1] / 16.0))
        cells = np.zeros((64, 64), dtype=bool)
        for gx, gy in grid:
            x = int(np.clip(np.floor(gx), 0, 63))
            y = int(np.clip(np.floor(gy), 0, 63))
            cells[y, min(x, 63)] = True
            cells[y, min(x + 1, 63)] = True
            cells[min(y + 1, 63), x] = True
            cells[min(y + 1, 63), min(x + 1, 63)] = True
        valid = cells[:valid_h, :valid_w]
        area = valid_h * valid_w
        fractions.append(float(valid.sum()) / area)
        pixel_counts.append(len(pixels))
        touched_rows.append(valid.sum(axis=1))
        if index % 25 == 0:
            print("  %d frames, mean coverage %.3f" % (index, float(np.mean(fractions))), flush=True)

    fractions = np.array(fractions)
    report = dict(frames=len(fractions), mean_coverage=float(fractions.mean()),
                  median_coverage=float(np.median(fractions)),
                  p10=float(np.percentile(fractions, 10)),
                  p90=float(np.percentile(fractions, 90)),
                  max_coverage=float(fractions.max()),
                  mean_pixels=float(np.mean(pixel_counts)),
                  grid="64x64, valid %s" % ("rows up to ceil(hw/16)"))
    print(json.dumps(report, indent=1))
    with open("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/query_coverage.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
