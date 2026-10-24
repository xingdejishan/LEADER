# Rotation error diagnosis — pilot (2026-10-24)

## Scope

Source poses: `results/full-online-final.npz` on branch `hypothesis-arbitration`, commit `5a767d5`.
Ground truth: `evaluator-only-truth/<date>.json`, field `T_WB`.
Rotation metric: SO(3) geodesic angle, degrees.

Clean set: 104,473 frames — the 236 `replaced` frames of `scored.npz` and 19 further frames in the same
51.205 s window on 2012-02-12 are removed (offset parallel to the reference-minus-raw median, above 5 m).

Frames with t0 rotation error above 3 degrees: 4,622 in this recomputation.
`v7_rotation_distribution_clean_20261024.json` reports 4,624; that file additionally excludes four missing predictions.

## Sample

30 frames from those 4,622, stratified by date: 2012-02-12: 11, 2012-02-19: 6, 2012-03-31: 7, 2012-05-26: 6. Seed 20261024.
10 control frames from the clean set with t0 rotation error at or below 3 degrees, same stratification.

## Run

```
python e2e_online.py --keys-file pilot_keys.json --out results/pilot40.jsonl --gpu 1 --shard 0/1
```

Remaining settings at their defaults: parallel experts, k = 8.4, visual budget 8, fast PnP on, verify cache on, kernel backend, full selection mode.

Frame outcomes are counted two ways:

| quantity | value |
|---|---:|
| frames written | 40 |
| frames with a final pose | 40 |
| `status: success` | 38 |
| `status: degraded` | 2 |
| degraded reason | visual_no_hypothesis (both frames kept a geometric pose) |
| frames where the visual branch produced a pose | 38 |

## Fields in `pilot40.jsonl`

`T0`, `T_geometry`, `T_visual`, `T`, `arbitration` (route3 and visual costs, `thresh`, `sigma_c`, `k`, `chosen`, `switched`),
`visual.detail` (retrieval ids, scored-candidate metadata, hypothesis count, consensus count, PnP calls).

## High-error frames

| key | status | t0 rot | geo rot | visual rot | visual trans | final rot | chosen | cost r3 | cost vis | switch thr |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2012-02-12/1329071240326940 | success | 3.49 | 3.49 | 4.51 | 0.27 | 3.49 | route3 | 0.236 | 0.286 | 0.023 |
| 2012-02-12/1329073113110988 | degraded | 4.65 | 4.65 | - | - | 4.65 | route3 | - | - | - |
| 2012-02-12/1329068821693905 | success | 3.24 | 3.24 | 2.48 | 0.22 | 3.24 | route3 | 0.330 | 0.380 | 0.085 |
| 2012-02-12/1329069115121729 | success | 4.29 | 4.29 | 1.47 | 0.99 | 4.29 | route3 | 0.402 | 0.499 | 0.087 |
| 2012-02-12/1329072237824247 | success | 3.61 | 3.61 | 2.58 | 0.37 | 3.61 | route3 | 0.322 | 0.355 | 0.086 |
| 2012-02-12/1329072642864267 | success | 3.49 | 3.49 | 1.50 | 0.10 | 3.49 | route3 | 0.302 | 0.427 | 0.142 |
| 2012-02-12/1329071260528921 | success | 3.75 | 3.75 | 2.82 | 0.26 | 3.75 | route3 | 0.311 | 0.367 | 0.070 |
| 2012-02-12/1329071872588276 | success | 4.23 | 4.23 | 3.22 | 0.20 | 4.23 | route3 | 0.326 | 0.317 | 0.047 |
| 2012-02-12/1329071085511933 | success | 3.26 | 3.26 | 8.40 | 0.76 | 3.26 | route3 | 0.245 | 0.521 | 0.084 |
| 2012-02-12/1329071575159480 | success | 4.60 | 4.60 | 4.88 | 0.77 | 4.60 | route3 | 0.239 | 0.319 | 0.055 |
| 2012-02-12/1329070658870800 | success | 3.17 | 3.17 | 2.33 | 0.44 | 3.17 | route3 | 0.310 | 0.344 | 0.137 |
| 2012-02-19/1329676746640117 | success | 3.35 | 3.35 | 4.82 | 0.88 | 3.35 | route3 | 0.385 | 0.460 | 0.049 |
| 2012-02-19/1329676463424109 | success | 4.38 | 4.38 | 3.54 | 0.06 | 4.38 | route3 | 0.235 | 0.284 | 0.134 |
| 2012-02-19/1329677146663023 | success | 3.15 | 3.15 | 1.87 | 2.97 | 3.15 | route3 | 0.514 | 0.912 | 0.185 |
| 2012-02-19/1329676374419027 | success | 3.14 | 3.14 | 2.21 | 0.38 | 3.14 | route3 | 0.219 | 0.241 | 0.097 |
| 2012-02-19/1329679101776299 | degraded | 4.23 | 4.23 | - | - | 4.23 | route3 | - | - | - |
| 2012-02-19/1329675994797386 | success | 3.00 | 3.00 | 2.17 | 0.51 | 3.00 | route3 | 0.317 | 0.381 | 0.034 |
| 2012-03-31/1333220402107435 | success | 3.45 | 3.45 | 2.47 | 3.51 | 3.45 | route3 | 0.320 | 0.764 | 0.042 |
| 2012-03-31/1333221289975163 | success | 4.16 | 4.16 | 7.70 | 0.37 | 4.16 | route3 | 0.258 | 0.284 | 0.019 |
| 2012-03-31/1333221267573437 | success | 6.55 | 6.55 | 12.36 | 0.79 | 6.55 | route3 | 0.257 | 0.480 | 0.025 |
| 2012-03-31/1333219174413176 | success | 3.10 | 3.10 | 3.88 | 1.48 | 3.10 | route3 | 0.445 | 0.546 | 0.062 |
| 2012-03-31/1333219144210885 | success | 4.08 | 4.08 | 5.45 | 0.55 | 4.08 | route3 | 0.351 | 0.456 | 0.060 |
| 2012-03-31/1333218565766638 | success | 3.10 | 3.10 | 2.91 | 0.63 | 3.10 | route3 | 0.159 | 0.265 | 0.044 |
| 2012-03-31/1333219194814716 | success | 3.99 | 3.99 | 5.67 | 0.70 | 3.99 | route3 | 0.394 | 0.497 | 0.084 |
| 2012-05-26/1338077365535720 | success | 4.01 | 4.01 | 4.16 | 0.69 | 4.01 | route3 | 0.428 | 0.546 | 0.062 |
| 2012-05-26/1338077289729516 | success | 3.19 | 3.19 | 1.33 | 0.23 | 3.19 | route3 | 0.408 | 0.381 | 0.073 |
| 2012-05-26/1338077068911411 | success | 3.10 | 3.10 | 6.80 | 0.50 | 3.10 | route3 | 0.491 | 0.623 | 0.047 |
| 2012-05-26/1338079997550772 | success | 4.26 | 4.26 | 3.71 | 0.62 | 4.26 | route3 | 0.190 | 0.188 | 0.061 |
| 2012-05-26/1338077054710270 | success | 3.43 | 3.43 | 5.52 | 10.39 | 3.43 | route3 | 0.391 | 0.484 | 0.076 |
| 2012-05-26/1338075142552567 | success | 3.06 | 3.06 | 2.90 | 0.28 | 3.06 | route3 | 0.365 | 0.419 | 0.031 |


## Control frames

| key | status | t0 rot | geo rot | visual rot | visual trans | final rot | chosen | cost r3 | cost vis | switch thr |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2012-02-12/1329073079707671 | success | 2.57 | 2.57 | 2.17 | 0.26 | 2.57 | route3 | 0.159 | 0.177 | 0.024 |
| 2012-02-12/1329068543467296 | success | 0.80 | 0.80 | 3.33 | 0.45 | 0.80 | route3 | 0.302 | 0.322 | 0.044 |
| 2012-02-19/1329678269927939 | success | 2.63 | 2.63 | 1.84 | 0.33 | 2.63 | route3 | 0.266 | 0.262 | 0.102 |
| 2012-02-19/1329675092545806 | success | 1.05 | 1.05 | 0.66 | 0.15 | 1.05 | route3 | 0.176 | 0.172 | 0.039 |
| 2012-02-19/1329676819044248 | success | 1.59 | 1.59 | 1.57 | 0.27 | 1.59 | route3 | 0.395 | 0.440 | 0.227 |
| 2012-03-31/1333219964073792 | success | 1.09 | 1.09 | 4.69 | 2.48 | 1.09 | route3 | 0.437 | 0.689 | 0.158 |
| 2012-03-31/1333219530040300 | success | 1.78 | 1.78 | 3.25 | 0.04 | 1.78 | route3 | 0.421 | 0.393 | 0.118 |
| 2012-05-26/1338077581153386 | success | 0.71 | 0.71 | 1.44 | 1.88 | 0.71 | route3 | 0.255 | 0.535 | 0.096 |
| 2012-05-26/1338079968948444 | success | 2.36 | 2.36 | 2.14 | 0.10 | 2.36 | route3 | 0.164 | 0.155 | 0.070 |
| 2012-05-26/1338077547550642 | success | 1.13 | 1.13 | 3.29 | 0.92 | 1.13 | route3 | 0.251 | 0.541 | 0.047 |


## Counts

| quantity | value |
|---|---:|
| high-error frames | 30 |
| high-error frames with a visual pose | 28 |
| high-error frames without a visual pose | 2 |
| visual rotation better than t0 by more than 0.5 deg | 14 |
| median improvement among those (deg) | 0.958 |
| visual rotation worse than t0 by more than 0.5 deg | 10 |
| median degradation among those (deg) | 1.884 |
| visual rotation at or below 3 deg | 13 |
| visual rotation at or below 3 deg and visual translation at or below 1 m | 11 |
| same, and visual translation below the geometry translation | 2 |
| frames where the visual pose was chosen | 0 |
| frames where final rotation equals t0 rotation | 40 / 40 |
| frames where geometry rotation equals t0 rotation | 40 / 40 |
| visual translation error at or below 1 m | 24 / 28 |
| same, among the frames where visual rotation is better | 12 / 14 |
| median of cost_visual minus cost_route3 | 0.0775 |
| frames where cost_visual is below cost_route3 | 3 / 28 |
| control frames where final rotation equals t0 rotation | 10 / 10 |

## Counterfactual: switch threshold set to zero

Rule: take the visual pose whenever `cost_visual < cost_route3`. Measured on the same 40 frames.

| group | switched | frames above 3 deg before | frames above 3 deg after |
|---|---:|---:|---:|
| high-error | 3 | 30 | 29 |
| control | 4 | 0 | 1 |

Both groups are stratified by hand, so these counts are not a population estimate.

## Limits of this sample

- Sampled t0 rotation spans 3.005 to 6.551 degrees; 1 frame is above 5 degrees.
  The 292 frames between 5 and 10 degrees in the full set are not represented.
- Only the visual pose selected by the branch is stored. The scored-candidate list carries scores, inlier counts and residuals, not per-candidate poses.
