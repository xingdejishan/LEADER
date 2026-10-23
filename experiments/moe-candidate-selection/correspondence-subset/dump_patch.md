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

# Attribution of the 2 hits (added after review)

The first cut mixed two sources: one PnP per DBSCAN cluster, and one PnP over all
correspondences. Re-ran with a per-source label.

| source | frames with a qualified solution | correct poses (1 m / 2 deg) |
|---|---|---|
| per-cluster PnP | 0 / 40 | 0 |
| global PnP | 6 / 40 | 2 |

Attribution of the correct poses: cluster only 0, global only 2, both 0, neither 38.

So the 2/40 reported earlier belongs entirely to the global solve; the clustered solve
produced no qualified solution on any of the 40 frames.

# Open item: solver used

The frozen pipeline produced these candidate lists with the deterministic solver
(`--fast-pnp 1`). The attribution run called `common.solve_pnp`, which is EPnP +
RANSAC 2000 + LM refinement. Same correspondences, different solver. The 24-frame
bit-identical check between the two solvers is recorded in
`sp-v11-accelerated-20261017-v1/FREEZE.md`; the two disagree on this 40-frame sample in
the sense that the frozen candidate list contains none of the 2 poses that the RANSAC
solve reached.

Not measured: whether this holds on a stratified sample, on other dates, or with the
deterministic solver called on the same numbers in the same way.

# Strict solver comparison on identical correspondences

Same `(world, pixels, ids)` per frame, same intrinsics, same extrinsics, same
qualification bar (15 unique inliers, 6 coverage cells, ratio 0.2). A = the deterministic
fast solver from `match_prov.py`; B = EPnP + RANSAC 2000 + LM (`common.solve_pnp`).

| | solutions | qualified | correct (1 m / 2 deg) | median per call | total |
|---|---|---|---|---|---|
| A fast | 29 / 40 | 4 | 0 | 2.5 ms | 92 ms |
| B RANSAC | 31 / 40 | 6 | 2 | 10.2 ms | 693 ms |

Correct poses: both 0, only B 2, only A 0, neither 38.

Per-frame detail on the two B-correct frames:

- frame 1: A produced a **qualified** solution (64 inliers, 7 cells) that is not correct.
- frame 5: A produced a solution that is **not qualified** (22 inliers, 5 cells).

B across seeds 20 / 7 / 99: correct on 3 of 3 seeds for those 2 frames, 0 of 3 for the
other 38.

# Confound that remains

The correspondence dump was produced with `--arm v8_noprune`; the frozen candidate table
compared against earlier is `match-frozen_nopruneopt`. The comparison above is
solver-vs-solver on one and the same correspondence set, so it is internally consistent;
the earlier baseline (0/40) was measured against a table built from a different arm.
