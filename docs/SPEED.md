# Where the search module's time actually goes

Measured on the same 1502 hard frames, by `experiments/ablate_points.py`. Accuracy
numbers are deterministic, so each is a single run; latency numbers are the median
of 2–4 repeats of a single-process, warm-cache, 200-frame contiguous segment,
because the machine is shared and single runs move by 10–20 %.

## 1. Three solvers, identical answers, 2.5x apart

All rows use the endpoint-covered subset at 4000/1500 points.

| solver | source | MPE | median | <=1 m | evals/frame | latency |
|---|---|---|---|---|---|---|
| `fast_lib.solve(prune=True)` | what `pipeline.localize` calls today | 1.5059 | 0.2920 | 85.1 % | 211 | **274 ms** |
| `fast_lib.solve(prune=False)` | full expansion | 1.5059 | 0.2920 | 85.1 % | 334 | — |
| `adaptive.solve_adaptive` | the solver `docs/SAMPLING.md` quotes | 1.5059 | 0.2920 | 85.1 % | 128 | **109 ms** |

The three agree on MPE, on the median, on the <=1 m rate and on all three hard
buckets. They do not agree on work. The pruning path expands about 4.4 of the six
windows because its bound prunes almost nothing, while the adaptive rule expands
1.39 and searches the same poses. The default entry point is therefore paying 2.5x
for an identical answer.

This also reconciles two numbers that look contradictory: 113.5 ms is the adaptive
solver, and 274 ms is what `pipeline.py` actually runs.

## 2. The subset endpoint is worth more than it looks

`points[::max(1, len // n)][:n]` degenerates into a prefix whenever the voxelised
scan holds fewer than 2n points. At n_fine = 4000 that is **312 of 1502 frames
(21 %)**; the median scan is 8602 points, so the other 79 % are unaffected.

Taking `np.linspace(0, len - 1, n)` instead costs nothing and moves accuracy:

| subset | MPE | median | <=1 m |
|---|---|---|---|
| stride (`[::len // n][:n]`) | 1.5166 | 0.3020 | 85.0 % |
| linspace (endpoint covered) | **1.5059** | **0.2920** | 85.1 % |

## 3. The point budget has a plateau, which the old rule hid

Adaptive solver, endpoint-covered subset, 1502 frames:

| n_fine | n_coarse | MPE | median | <=1 m | 5–8 m bucket | latency |
|---|---|---|---|---|---|---|
| 4000 | 1500 | 1.5059 | 0.2920 | 85.1 % | 62 % | 109 ms |
| 3000 | 1500 | 1.5074 | 0.2911 | 85.0 % | 62 % | 94 ms |
| 2000 | 1500 | 1.5040 | 0.2904 | 85.1 % | 62 % | 88 ms |
| 1500 | 1500 | 1.5070 | 0.2909 | 84.8 % | 62 % | 80 ms |
| 1500 | 1000 | **1.5056** | **0.2909** | **85.0 %** | **62 %** | **71 ms** |
| 1000 | 1000 | 1.5239 | 0.3020 | 84.8 % | 54 % | 64 ms |
| 500 | 1500 | 1.5131 | 0.3028 | 84.8 % | 62 % | — |

The plateau runs from 4000 down to 1500 fine points with no measurable loss, and
the coarse budget can fall from 1500 to 1000 for free as well. Below that it starts
to cost: at 1000/1000 the hardest bucket drops from 62 % to 54 %.

For contrast, the same n_fine = 2000 under the old stride rule gives MPE 1.5732,
median 0.3085, and a 5–8 m bucket of **29 %** instead of 62 %. That is the confound
in the earlier "fewer points is bad" result: the old rule was losing azimuthal
coverage at the same time as it was losing points.

## 4. Verdict on 50 ms

| configuration | latency | vs the shipped default |
|---|---|---|
| shipped default (prune, stride, 4000/1500) | 274 ms | 1.0x |
| adaptive + endpoint subset, 4000/1500 (the documented 113.5 ms) | 109 ms | 2.5x |
| adaptive + endpoint subset, 1500/1000 | **71 ms** | **3.8x** |
| adaptive + endpoint subset, 1000/1000 | 64 ms | 4.2x, accuracy starts to slip |

50 ms is not reached. The fixed cost outside the search is about 18 ms (map crop 8
plus scan decode and voxel 10), and the remaining ~53 ms is 183k k-d tree queries
plus a fresh tree build on every frame. Going further needs kernel work — reusing
the tree while the crop is unchanged, or moving the query off the CPU — not fewer
points.

## 5. Proposed change, not applied

`src/fast_lib.py`:

```python
def spread_subset(points, n):
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]
```

`src/pipeline.py`, in `localize()`: call `adaptive.solve_adaptive(...)` instead of
`F.solve(...)`, passing `n_fine=cfg.fine_points` and `n_coarse=cfg.coarse_points`.
It returns five values, not four, so the unpack needs
`s, d, cost, evals, expanded`. `solve_adaptive` takes no `topk` — it hard-codes six
candidates — so `Config.topk` and `Config.prune` become dead for the default path.

## Reproducing

```bash
python experiments/ablate_points.py acc adaptive linspace 1500 1000 1502
python experiments/ablate_points.py lat adaptive linspace 1500 1000 200
```

Both honour `CFS_DEPLOY` (which checkout to import `src/` from) and `CFS_FRAMES`
(the frame table directory), and append one row per run to
`experiments/speed_results.jsonl`. The subset rule is swapped by monkeypatching
`fast_lib.spread_subset`, so the module under test is never modified.
