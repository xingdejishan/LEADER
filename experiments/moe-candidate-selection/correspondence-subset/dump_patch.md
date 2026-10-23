# How the correspondence dump was produced

`match_prov.py` is copied to a scratch path and patched in three places, so the frozen
script itself is untouched:

1. `OUTDIR` is redirected to a scratch directory;
2. `solution_from()` stores `(ids, world, pixels, ref)` for the aggregated channel,
   keyed by the current frame index;
3. an `atexit` hook writes the dict to `corr_dump.npy`.

Then: `--arm v8_noprune --limit 40 --batched 1 --fast-pnp 1`.

First pass, 40 frames (2012-02-12), 5841 bound correspondences, 11111 matches,
475 PnP calls, 35 accepted candidates.

# First result (40 frames, one date — biased sample, not a verdict)

| set | Oracle Recall@13 (1 m / 2 deg) | median candidates |
|---|---|---|
| frozen candidates | 0 / 40 | 9 |
| subset PnP (DBSCAN 2 m, one PnP per cluster) | 2 / 40 | **0** |

The subset version produced a median of **zero** qualified candidates: splitting the
correspondences into coherent 3D clusters leaves each cluster below the frozen
qualification bar (15 unique inliers, 6 coverage cells, inlier ratio 0.2). So the first
cut measures cluster starvation, not the hypothesis. The next version must either relax
the per-subset qualification or use fewer, larger clusters, and must run on a stratified
sample rather than the first 40 frames of a single date.
