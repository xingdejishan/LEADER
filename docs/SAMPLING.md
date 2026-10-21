# The point-subset defect and its fix

Every number elsewhere in this repository was produced with a defective point
subset. Fixing it improves accuracy at zero cost, so the older figures are
pessimistic and are superseded by the ones here.

---

## 1. What was wrong

The scan was voxel-downsampled and then truncated with `scan[:N]`. `voxel()`
returns points in acquisition order, which for a spinning LiDAR is azimuthal, so
`scan[:4000]` keeps a contiguous slice of the sweep rather than a sample of it.

Measured azimuthal coverage (36 bins over 360 degrees):

| frame | full scan | after voxel | first 4000 |
|---|---|---|---|
| 3 | 35/36 | 35/36 | **7/36** |
| 1500 | 35/36 | 35/36 | **7/36** |
| 1125 | 34/36 | 34/36 | **7/36** |
| 750 | 32/36 | 31/36 | 31/36 (only 3104 points) |

So for most frames the "4000-point budget" was a ~70-degree wedge — roughly a
fifth of the scene. The same wedge is used for every pose, so the comparison
stays self-consistent, but the cost estimate is far noisier than the budget
implies and is spatially biased.

This also explains a puzzle: a 500-point screen kept the full-search winner only
11.5 % of the time, *below* the 17.8 % expected from picking 8 of 45 at random.
A noise-driven ranking would be unbiased; a wedge-driven one systematically
favours some poses.

## 2. The fix

Take an evenly strided subset instead:

```python
def spread_subset(points, n):
    if len(points) <= n:
        return points
    return points[::max(1, len(points) // n)][:n]
```

Zero cost — the slicing measures 0.00 ms. An azimuth-sorted variant is slightly
better still but needs an argsort of the whole scan.

## 3. Effect on accuracy (full 1502 frames)

| subset | MPE | median | ≤1 m | windows expanded |
|---|---|---|---|---|
| prefix (previous) | 1.5638 | 0.3513 | 83 % | 1.64 |
| **stride (adopted)** | **1.5166** | 0.3020 | **85 %** | 1.41 |
| azimuth-sorted | 1.5036 | 0.2909 | 85 % | 1.38 |

Hard buckets, stride versus prefix: 1–3 m 87 % vs 86 %, 3–5 m 63 % vs 65 %,
5–8 m 58 % vs 58 %. So the aggregate improves clearly, the 3–5 m band gives back
two points, and the rest is a wash.

Note this beats the old "exhaustive" reference figure of MPE 1.5403 — that number
was itself computed on the wedge.

## 4. Effect on latency

End-to-end for the search module only (initial pose + raw scan in, corrected pose
out), 200-frame contiguous segment, 4-thread queries, warm map cache:

| stage | ms |
|---|---|
| scan decode + range filter + voxel | 7.2 |
| map crop (cache hit) | 1.0 |
| KD-tree build | 5.1 |
| amortised map build on cache misses (~34 % of frames) | ~14 |
| coarse (17 + 63 evaluations at 1500 points) | 25.3 |
| six candidate centres at 4000 points | 10.2 |
| fine (1.05 windows × 45 poses × 4000 points) | 46.5 |
| **total** | **113.5** (p90 150) |

The fix itself contributes 0.00 ms. The gap between this and an earlier 21 ms
preprocessing figure is machine load and cache-miss rate, not the sampling
change.

**Scope warning.** This 113.5 ms excludes both neighbours in the chain: the
upstream V7 network (~52 ms) and the downstream CoFAR refiner (~296 ms). The
whole chain is therefore ~460 ms. The 50 ms target discussed for this module
applies to the middle segment only.

## 5. Where the remaining time goes, and what was already ruled out

- Fine stage 46.5 ms for essentially one window (45 poses × 4000 points =
  180k nearest-neighbour queries).
- A GPU brute-force kernel was measured and is **slower** than the CPU k-d tree
  on dense maps: 140 ms vs 77 ms at 39k map points, 90 ms vs 74 ms at 23k. It
  only wins below ~8-9k map points (6-41 ms vs 35-67). Brute force discards the
  tree's pruning, which reduces 39k candidates per query point to about twenty
  node visits.
- Progressive point screening inside a window fails: at 500 points the paired
  standard error is ~0.0054 while the gap between the best and second-best pose
  is ~0.0011, so the screen ranks noise.

## 6. Reproducing

`experiments/spread_test.py <prefix|stride|azimuth>` runs the A/B above on the
full 1502 frames.

Note that `analysis/log_fine.py` and everything derived from `fine_stats.npz`
(the statistics in docs/ADAPTIVE.md) were produced with the **prefix** subset.
`src/fast_lib.py` and `src/adaptive.py` now use `spread_subset` by default.
