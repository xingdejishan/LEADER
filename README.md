# Coarse-to-fine along-track search for LiDAR re-localisation

A training-free module that corrects the **along-track and lateral translation
error** of a localisation pose. It starts from an upstream pose (LEADER/V7 after
one local refinement pass), enumerates candidate positions on a static LiDAR
reference map, and returns a translation correction.

Rotation is **not** touched. The module only ever searches within ±8 m of the
incoming pose, so it cannot recover a globally wrong retrieval.

## Files

```
src/fast_lib.py        the pieces: scan decode, voxel, map cache, query, solve()
src/adaptive.py        adaptive fine-window expansion (recommended; replaces the pruning)
src/pipeline.py        the chain: record + initial pose -> corrected pose
src/run_frames.py      CLI driver over a frame list
src/exact_gpu.py       shared-neighbourhood exact scorer (rejected: gather too slow)
analysis/              scripts behind the statistics in docs/ADAPTIVE.md
distance_field/        GPU truncated-distance-field alternatives (all rejected, see docs)
experiments/           the scripts that produced the numbers in docs/RESULTS.md
docs/RESULTS.md        every sweep result and its conclusion
docs/ADAPTIVE.md       candidate ranking statistics, the pruning bug, the adaptive rule
docs/DISTANCE_FIELD.md GPU distance field: what worked, what did not
```

## The chain

```
record ──► scan decode ──► range filter ──► voxel ─┐
                                                   ├─► solve() ──► pose ──► refiner? ──► pose
initial pose ──► along-track dir ─────────────────┘        ▲
                                                           │
reference library ──► MapSource (triple cache) ────────────┘
```

Eight stages, all in `pipeline.localize`:

1. **reference triple** — three other-date references nearest the initial position
2. **map assembly** — the triple's merged cloud, built once, cropped per frame
3. **scan preparation** — decode, range filter, 0.3 m voxel downsample
4. **direction** — along-track unit vector from the initial pose's x axis
5. **coarse grid** — 17 along-track × 3 lateral, 1500 query points
6. **candidate NMS** — best 6 cells, ≥1.5 m apart along-track
7. **fine grid** — 5 × 9 per candidate, 4000 query points, gradient bound
8. **refine** — optional external local refiner

Minimal use:

```python
import fast_lib as F, pipeline as P

maps = F.MapSource(refs, positions)          # refs: rows with scan_path + world_from_body
cfg  = P.Config()
pose, diag = P.localize(record, initial_pose, maps, cfg)
# diag: s, d, cost, evals, triple, map_points, scan_points
```

Over many frames: `python src/run_frames.py --frames ... --refs ... --poses ... --out ...`.

## How it works

Given the current scan `q`, an initial pose `T0 = (R0, t0)`, and a local
reference cloud `M` built from three other-date reference scans:

1. **Anchor.** Scan the lateral offset `d ∈ [-1, 1]` in 0.25 m steps at
   `s = 0`, keep the best. Cheap and only used to centre the coarse grid.
2. **Coarse grid.** 17 along-track positions `s ∈ [-8, 8]` at 1 m × 3 lateral
   values `d_home + {-0.5, 0, 0.5}`, each scored with **1500** query points.
   Keep the best 6 cells, requiring them to be ≥1.5 m apart along-track so the
   six represent six distinct hypotheses rather than one peak.
3. **Fine grid.** For each candidate, a local 5 × 9 grid (along-track ±0.5 m at
   0.25 m, lateral ±0.4 m at 0.1 m) scored with **4000** query points.
   Candidates are processed cheapest-coarse-first; a **gradient lower bound**
   skips a candidate whose region provably cannot beat the incumbent.
4. Optionally hand the winner to an external local refiner (CoFAR).

The score is the mean over query points of `min(nearest-neighbour distance,
1.0 m)`.

### Why this shape

- **Lateral and along-track are handled asymmetrically.** Along-track is
  enumerated and compared; lateral is optimised within each hypothesis. They are
  never solved jointly — a joint 2-D argmax collapses onto the lateral zero
  (established separately: 58.1 % of frames pick lateral 0, gradient/sign
  agreement 0.073). Along-track identification improves from 46.2 % to 62.7 %
  once the lateral offset is known, which is why the lateral anchor exists at
  all.
- **The fine stage is not merely precision.** Replacing it with a single
  centre re-score costs 3.8 % MPE and 21 % median error even *after* the local
  refiner runs. Its real job is to re-rank candidates with 4× the query points;
  1500 points are enough to shortlist but not to decide.
- **Along-track step 1 m is a plateau.** Coarser steps force a wider fine
  window, and the fine stage costs ~2.7× more per evaluation, so total work
  grows while accuracy does not improve (see RESULTS).

## Lossless accelerations (verified bit-identical)

| change | effect |
|---|---|
| packed-key voxel downsampling | 6× faster than `np.unique(axis=0)`, identical output |
| reference-triple map cache | map build 205 ms → 2.8 ms on a hit |
| `distance_upper_bound=1.0` on the tree query | ~20 % end-to-end, identical cost |
| gradient lower bound in the fine stage | 270 → ~110 fine evaluations, identical pose |

Full-set check: MPE 1.5693, median 0.3528, 83 % within 1 m — unchanged by all
four, on all 1502 frames.

## Measured behaviour (1502 hard frames, NCLT dev dates)

| deploy error | frames | before | after (search only) | after (+ refiner) |
|---|---|---|---|---|
| 3–5 m | 49 | 3.56 m | 0.49 m | 0.23 m |
| 5–8 m | 24 | 6.02 m | 0.46 m | 0.33 m |
| > 8 m | 12 | 10.30 m | 10.81 m | 10.81 m |

All of the value is in the 1–8 m range. Beyond 8 m the cost surface has a
strong spurious minimum and this module makes things slightly worse; those
frames need global retrieval instead.

## Running it

The scripts under `experiments/` were run against this layout on the research
server:

```
/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/
    scored.npz        # keys, reference poses (GT), deployed baseline poses
    selection.json    # reference and query scan lists with paths
```

They issue shell heredocs; `ssh_exec.py` streams them to the host. Adapt the
paths before running elsewhere.

## Sampling defect (fixed) — read this before quoting any number

Every figure below was produced with `scan[:N]` as the point subset. Because the
voxelised scan is still in acquisition (azimuthal) order, that keeps a
~70-degree wedge rather than a sample: a 4000-point budget covered 7 of 36
azimuth cells, against 35 of 36 for the whole scan. Replacing it with an evenly
strided subset is free (0.00 ms) and improves MPE from 1.5638 to **1.5166**,
median from 0.3513 to **0.3020**, at 1.41 instead of 1.64 expanded windows. See
[docs/SAMPLING.md](docs/SAMPLING.md); the older numbers in RESULTS.md and
ADAPTIVE.md are pessimistic and were computed on the wedge.

## Adaptive expansion (recommended)

Scoring the six candidate centres at the full point budget, then expanding a
window only while its centre is within 0.10 m of the best centre, reproduces the
six-window result exactly: **MPE 1.5638, median 0.3513, 83 % within 1 m** — with
1.64 windows and 79 exact evaluations instead of 6 windows and 270 evaluations.
End to end that is ~146 ms (or ~116 ms with the GPU coarse stage), against 162 ms
before, at zero accuracy cost.

It also fixes a correctness problem: the previous gradient pruning used a bound
that is not valid. A sound Lipschitz bound would prune nothing at all on this
data, because the largest candidate gap is far below the window radius. The
adaptive rule is an explicit heuristic rather than a claimed bound. See
[docs/ADAPTIVE.md](docs/ADAPTIVE.md).

## GPU distance field (rejected)

A truncated distance field over the local map, sampled on the GPU, makes the
scoring 10–18× faster (coarse 27 ms → 1.5 ms, fine 234 ms → 24 ms). But every
way of using it to *reduce the number of exact evaluations* lost accuracy: the
field error (0.05–0.08 m) is larger than the cost differences between
neighbouring poses, so the field's ranking cannot replace the exact comparison.
The only change that holds accuracy is moving the **coarse stage** to the GPU,
which takes the chain from ~162 ms to ~132 ms. Details, including the axis-order
bug that made the first field silently useless, are in
[docs/DISTANCE_FIELD.md](docs/DISTANCE_FIELD.md).

## Caveats

- Hyper-parameters (ranges, steps, point counts) were selected **on the same
  1502 frames** used for evaluation. That is tuning on the test set; treat the
  absolute numbers accordingly and re-validate on unseen dates.
- The reference map is posed with ground-truth reference poses (offline
  assumption) and does not include reference-registration error.
- The along-track direction is taken from the incoming pose's own forward axis.
  It is single-frame and training-free, but inherits that pose's orientation
  error (median 1.3°).
- The 1502-frame set is not contiguous; frame ordering matters only for the
  map cache hit rate, never for the result.
