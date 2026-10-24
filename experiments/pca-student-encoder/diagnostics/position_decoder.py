"""Is the token map predictable from position alone?

A per-frame global code plus a position-conditioned decoder only works if the token map is
mostly a smooth function of image position.  For each frame we fit a low-order polynomial of
(x, y) per channel and report the explained variance, next to the error a 4-bit quantiser
introduces, and the redundancy across channels (rank of the centred map).
"""
import glob
import json
import sys

import numpy as np

CACHE = "/data2/zhangcy/leader-v2-full-trajectory-20261001/sam"
BASIS = "/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz"
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(20261023)

basis = np.load(BASIS)
mean = basis["mean"].astype(np.float32)
projection = basis["pca64_int8"].astype(np.float32)

dates = ["2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11"]
files = []
for date in dates:
    listing = sorted(glob.glob(CACHE + "/" + date + "/frames/*.npy"))
    if listing:
        files.extend(listing[i] for i in rng.choice(len(listing), size=min(FRAMES // len(dates), len(listing)),
                                                     replace=False))
print("frames", len(files))

rows = []
for path in files:
    array = np.load(path).astype(np.float32)
    if array.shape != (64, 64, 256):
        continue
    code = (array.reshape(-1, 256) - mean) @ projection          # (4096, 64)
    code = code.reshape(64, 64, 64)
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="ij"), -1)
    x = grid[..., 0].reshape(-1)
    y = grid[..., 1].reshape(-1)
    features = np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], -1)  # (4096, 6)
    target = code.reshape(-1, 64)
    coefficient, *_ = np.linalg.lstsq(features, target, rcond=None)
    predicted = features @ coefficient
    residual = target - predicted
    total_variance = float((target ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / max(total_variance, 1e-12)
    # 4-bit quantisation error on the same map, per channel range
    levels = 127.0
    scale = np.abs(target).max(axis=0, keepdims=True) / levels
    quantised = np.clip(np.rint(target / scale), -levels, levels) * scale
    quantisation_error = float(((quantised - target) ** 2).sum() / max(total_variance, 1e-12))
    centred = target - target.mean(0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centred, tol=1e-3 * np.abs(centred).max()))
    rows.append(dict(path=path.split("/")[-1], r2=float(r2),
                     residual_energy=float((residual ** 2).sum() / max(total_variance, 1e-12)),
                     quantisation_energy=quantisation_error, rank=rank))

r2 = np.array([r["r2"] for r in rows])
q = np.array([r["quantisation_energy"] for r in rows])
report = dict(frames=len(rows),
              r2_mean=float(r2.mean()), r2_median=float(np.median(r2)),
              r2_p10=float(np.percentile(r2, 10)), r2_max=float(r2.max()),
              quantisation_energy_mean=float(q.mean()),
              residual_over_quantisation=float((1 - r2).mean() / max(q.mean(), 1e-12)),
              rank_median=int(np.median([r["rank"] for r in rows])))
print(json.dumps(report, indent=1))
with open("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/position_decoder.json", "w") as stream:
    json.dump(report, stream, indent=1)
