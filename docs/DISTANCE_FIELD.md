# GPU distance field: what worked and what did not

Goal: replace the repeated KD-tree nearest-neighbour evaluation with a
precomputed truncated distance field, so that coarse scoring runs on the GPU and
the fine stage becomes a continuous optimisation. Target: 162 ms → <50 ms.

**Outcome: the GPU scoring is genuinely 10–18× faster, but every way of using it
to reduce the number of exact evaluations costs accuracy. A realistic landing is
~130 ms, not 50 ms.**

All numbers below are on a 10-frame stratified sample (2 per error band) with an
RTX 4090, `torch 1.12`. Accuracy columns are position error after the chain,
`med` = median over the sample, `<=1m` = fraction within 1 m.

---

## 1. The field

`D(x) = min(distance to nearest reference-map point, 1.0)`, built by rasterising
the local map into a voxel grid and running a Euclidean distance transform.
Gradients are taken from the **untruncated** EDT: the truncated field is exactly
flat wherever the true distance exceeds the truncation, so a gradient-based
optimiser would stall on it. This is a deliberate deviation from the original
design.

The local map spans roughly 110 × 100 × 22 m (60 m scan radius), so the grid is
large:

| voxel | cells | build time | field MAE vs exact | p95 | top-6 agreement |
|---|---|---|---|---|---|
| 0.4 m | 4.7 M | 0.8 s | 0.070–0.111 m | 0.29–0.33 | 4 / 6 |
| 0.3 m | 10.9 M | 2.1 s | 0.052–0.083 m | 0.22–0.24 | 5 / 6 |

Build time is per reference triple, so it must be moved offline; it cannot be
part of a 50 ms frame budget without a preload architecture.

## 2. A bug worth recording

The first version of the field looked numerically plausible but was worthless:
its correlation with the exact distance was **−0.05**, and — the tell — the error
did not change at all when the voxel size was varied from 0.4 m to 0.15 m.

Cause: `grid_sample` maps `grid[..., 0]` to the **last** spatial dimension of the
input tensor, so a field stored as `[x, y, z]` must be sampled with coordinates
given in `(z, y, x)` order. Passing them in `(x, y, z)` order silently samples a
transposed location. After the fix the correlation went to **0.885** and the MAE
to 0.093 m at 0.3 m voxels.

The "error independent of resolution" signature is a useful diagnostic: a
quantisation error would have scaled with the voxel size; a constant error means
the lookup itself is wrong.

## 3. Speed

| stage | exact KD (CPU) | GPU field | speedup |
|---|---|---|---|
| coarse, 60 candidates × 1500 pts | 27 ms | **1.5 ms** | 18× |
| fine, 270 poses × 4000 pts | 234 ms | **24 ms** | 10× |

## 4. Accuracy — three ways of using the field, all worse

| fine stage | med | ≤1 m | note |
|---|---|---|---|
| exact grid enumeration (baseline) | **0.78 m** | **50 %** | 45 poses/candidate, 4000 pts |
| GPU continuous optimisation (6 iters) | 1.36 m | 40 % | gradient descent on the field |
| field shortlist k=3 → exact re-score | 1.41 m | 40 % | field ranks, KD decides |
| field shortlist k=8 → exact re-score | 1.92 m | 30 % | larger shortlist is *worse* |

Two distinct failure modes:

1. **Continuous optimisation is worse than enumeration.** A gradient step stops
   in a local minimum of an approximate field; the 45-point grid compares the
   *true* cost at 45 places and takes the best. This confirms the earlier
   finding that the fine stage's value is enumeration, not precision.
2. **The shortlist misses the right pose.** The field error is 0.05–0.08 m while
   the cost differences between neighbouring poses inside one window are often
   smaller than that. Enlarging the shortlist makes it worse, not better,
   because the exact-cost minimum is not the same as the minimum-error pose —
   the same "the cost function's optimum is not the truth" effect seen
   throughout this work.

## 5. Where that leaves the budget

The only change that holds accuracy is the coarse stage on the GPU:

| stage | before | after |
|---|---|---|
| scan preparation | 21 ms | 21 ms |
| coarse | 31 ms | **1.5 ms** |
| fine (exact, pruned) | 110 ms | 110 ms |
| **total** | **162 ms** | **~132 ms** |

Reaching 50 ms requires the exact fine stage to fall from 110 ms to under 20 ms,
i.e. a further 5× cut in exact evaluations. Both attempts to do that are in the
table above and both lost accuracy.

The remaining untried option is an accurate enough field that the shortlist
never misses the best pose: at 0.1 m voxels the error would be ~0.02 m, but the
field would be ~300 M cells for a 110 m map — not viable in build time or
memory.

## 6. Reproducing

```
df_core.py    field construction, tensor-native sampling, batched refinement
df_run.py     grid vs continuous comparison, per-stage timing
hybrid.py     field shortlist + exact re-score
diag_field.py field accuracy versus voxel size
```

Run with `CUDA_VISIBLE_DEVICES=<free gpu>`. The scripts assume the server layout
used by the rest of this repository.
