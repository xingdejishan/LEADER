# Sweep results

All numbers are on the 1502-frame hard set (NCLT dev dates), evaluated with the
project's unified scorer. Columns `1–3 / 3–5 / 5–8` are the fraction of frames
within 1 m, split by the **deployed** error bucket. The last three buckets are
the only ones where the module has value.

Machine is shared; wall times vary by up to 2× with load. Evaluation counts are
load-independent and are the more reliable cost measure.

> Note: several early tables undercounted the coarse grid in their "evals"
> column (they counted only the anchor and the fine stage). Tables are labelled
> where that applies. Relative comparisons are unaffected.

---

## 1. Coarse parameters, one at a time

Baseline here: lateral anchor over ±2 m, 1500 query points, along-track ±10 m at
1 m, 3 lateral values, top-6, fine ±0.5/±0.4.

| config | MPE | median | ≤1 m | 1–3 | 3–5 | 5–8 |
|---|---|---|---|---|---|---|
| baseline | 1.5693 | 0.3528 | 83 % | 87 % | 57 % | 58 % |
| **lateral anchor range ±1.0** | **1.5618** | 0.3513 | 83 % | 86 % | **65 %** | 58 % |
| lateral anchor step 0.5 | 1.5673 | 0.3593 | 83 % | 86 % | 55 % | 58 % |
| coarse query points 750 | 1.6439 | 0.3889 | 80 % | 82 % | 51 % | 46 % |
| coarse query points 500 | 1.6832 | 0.4114 | 78 % | 79 % | 53 % | 58 % |
| along-track range ±8 | 1.5713 | 0.3528 | 83 % | 87 % | 57 % | 58 % |
| lateral count 1 (no triple) | 1.6018 | 0.3744 | 82 % | 80 % | 53 % | 58 % |
| lateral count 5 | **1.5612** | **0.3500** | 83 % | 87 % | 61 % | 58 % |

**Conclusions**

- The anchor range can be halved to ±1 m: *better* and cheaper. Beyond ±1 m the
  extra range only adds noise.
- **1500 coarse query points cannot be reduced.** 750 points costs 0.075 MPE and
  destroys the 5–8 m bucket (58 % → 46 %). Point count is the one parameter that
  looks redundant and is not.
- Along-track ±8 instead of ±10 is neutral (57/58 % buckets identical). Nothing
  beyond 8 m is recoverable anyway.
- 3 lateral values are necessary (1 value → 1.6018). 5 values is slightly better
  but the 3–5 m gain (+4 points on 49 frames) is inside the sampling noise.

## 2. Lateral anchor step and along-track step

| config | MPE | median | ≤1 m | 3–5 | 5–8 |
|---|---|---|---|---|---|
| anchor step 0.25 (base) | 1.5693 | 0.3528 | 83 % | 57 % | 58 % |
| anchor step 0.5 | 1.5673 | 0.3593 | 83 % | 55 % | 58 % |
| anchor step 1.0 | 1.5772 | 0.3610 | 83 % | 55 % | 58 % |
| anchor step 1.5 | 1.5925 | 0.3667 | 83 % | 49 % | 54 % |
| anchor step 2.0 | 1.5846 | 0.3674 | 82 % | 53 % | 58 % |
| along-track 2.0 m / fine ±1.0 | 1.5624 | 0.3506 | 83 % | 59 % | **62 %** |
| along-track 2.5 m / fine ±1.25 | 1.5721 | 0.3528 | 83 % | 59 % | 62 % |
| along-track 3.0 m / fine ±1.5 | 1.5704 | 0.3613 | 83 % | 59 % | 62 % |
| along-track 4.0 m / fine ±2.0 | 1.5743 | 0.3645 | 83 % | 59 % | 62 % |

**Conclusions**

- Anchor step can go to 0.5 m, no further. Saving is small because the anchor is
  only 17 of ~230 evaluations.
- A coarser along-track grid *improves* the 5–8 m bucket (58 % → 62 %) but only
  because the fine window must grow to cover the cell — and that multiplies the
  fine stage. Evaluations go from 221 to 359. The step itself is not the win.
- **Denser does not help either** (see §3), so 1 m is a plateau in both
  directions.

## 3. Coarse density vs fine window

Along-track range fixed at ±8 m; fine window = half the coarse step.

| config | MPE | median | ≤1 m | evals | 1–3 | 3–5 | 5–8 |
|---|---|---|---|---|---|---|---|
| step 1.0, fine ±0.5 (current) | 1.5713 | 0.3528 | 83 % | 221 | 87 % | 57 % | 58 % |
| step 0.5, fine ±0.25 | 1.5735 | 0.3603 | 83 % | 214 | 88 % | 59 % | 54 % |
| **step 0.25, *no fine stage*** | **1.7514** | 0.4543 | 76 % | 212 | 65 % | 35 % | 33 % |
| step 0.25, fine ±0.25 | 1.5729 | 0.3603 | 83 % | 310 | 88 % | 59 % | 54 % |

**Conclusion — the important one.** A 0.25 m coarse grid *with the same 1500
points* is as good as a 1 m grid at resolving position, yet it collapses to
1.7514 when the fine stage is removed. Same grid resolution, same evaluation
count, only the query-point budget differs (1500 vs 4000).

So the fine stage is not "more resolution". It is **4× the points used to
decide**. The cost surface at 1500 points is too noisy to rank candidates; it is
fine for shortlisting. This explains why every attempt to cut the query-point
count failed.

## 4. How many candidates need re-scoring

| top-k | MPE | median | ≤1 m | evals | 3–5 | 5–8 |
|---|---|---|---|---|---|---|
| 6 (current) | 1.5638 | 0.3513 | 83 % | 214 | 65 % | 58 % |
| 4 | 1.5653 | 0.3526 | 83 % | 190 | 63 % | 50 % |
| 3 | 1.5819 | 0.3552 | 83 % | 172 | 57 % | 42 % |
| 2 | 1.6157 | 0.3674 | 82 % | 144 | 45 % | 38 % |
| none (coarse only) | 1.8084 | 0.5455 | 77 % | 60 | 37 % | 29 % |

Six is the knee. Below four the 5–8 m bucket falls apart.

## 5. Is the fine stage replaceable by the local refiner?

Both arms followed by the project's geometric refiner (CoFAR).

| arm | evals | before refiner | after refiner | median | ≤1 m | 3–5 | 5–8 |
|---|---|---|---|---|---|---|---|
| coarse + centre re-score only | 66 | 1.6909 | 1.5679 | 0.3629 | 83 % | 57 % | 71 % |
| coarse + fine grid | 214 | 1.5638 | 1.5098 | 0.2855 | 85 % | 67 % | 58 % |

The refiner recovers most of the cheap arm's deficit (1.6909 → 1.5679), which
confirms that the fine stage's *precision* work is duplicated by the refiner.
What it does not recover is the *selection* difference: 3.8 % MPE and 21 %
median. If latency dominates, the cheap arm costs ⅓ of the search work for those
3.8 %.

Variants of the cheap arm:

| arm | MPE | median | evals | 3–5 | 5–8 |
|---|---|---|---|---|---|
| full fine grid | 1.5638 | 0.3513 | 214 | 65 % | 58 % |
| centre + 4 cross points | 1.6009 | 0.4348 | 90 | 57 % | 62 % |
| centre only | 1.6909 | 0.5263 | 66 | 51 % | 71 % |

## 6. Ablations

| config | MPE | median | ≤1 m | 1–3 | 3–5 | 5–8 |
|---|---|---|---|---|---|---|
| baseline | 1.5693 | 0.353 | 83 % | 87 % | 57 % | 58 % |
| skip anchor solve | 1.5722 | 0.368 | 83 % | 82 % | 57 % | 57 % |
| along-track collapsed to ±0.5 m | 1.7737 | 0.450 | 76 % | 50 % | **0 %** | **0 %** |
| both collapsed | 1.7725 | 0.451 | 77 % | 46 % | 0 % | 0 % |

Collapsing the along-track range zeroes out exactly the frames the module
exists for. The anchor solve, by contrast, is nearly pure overhead (0.2 % MPE
for 10 % of the evaluations) — it is kept only because it makes the coarse
grid better conditioned on frames where the lateral offset is large.

## 7. Latency

Single process, 4-thread tree queries, warm cache, contiguous segment.

| stage | before | after |
|---|---|---|
| map build + scan read | 205–211 ms | **21 ms** |
| coarse | 31 ms | 31 ms |
| fine (with gradient bound) | 222 ms (270 evals) | ~110 ms (108 evals) |

Lossless accelerations, each verified bit-identical on all 1502 frames:

| change | gain |
|---|---|
| packed-key voxel downsampling | 6.2× vs `np.unique(axis=0)`, identical output |
| reference-triple map cache | 205 ms → 2.8 ms on hit |
| `distance_upper_bound` on the tree query | ~1.5–3× on the query stage |
| gradient lower bound | 270 → ~110 fine evaluations |

End-to-end measured 332 ms at machine load 39; the load-normalised estimate is
~160 ms. The machine is shared and the readings could not be taken below load
20.

## 8. Approaches that did **not** work

Recorded so they are not retried.

| idea | outcome |
|---|---|
| 1 m voxel-hash nearest-neighbour backend | 9.3× **slower** than the k-d tree at 4000 points (touches ~92 candidate points per query point); 200× slower in a naive implementation |
| branch and bound over the whole (s, d) domain | 573 evals vs 429 for the fixed grid, and MPE 1.5710 vs 1.5612 — the Lipschitz bound is too loose relative to the cost spread |
| progressive window widening (±1 → ±2 → ±6 → ±10) | early stop almost never fires: the coarse minimum sits on the window edge for most frames, so all four levels run (432 coarse evals vs 189) |
| lateral range halved to ±0.5 (single value) | 1.6018 |
| coarse query points 500–750 | 1.68–1.64 |
| `Q@R.T` layout / batching the 3321 poses into one query | 1.03×, negligible |
| fixing the along-track direction from the reference-library trajectory instead of the pose heading | 1.6352 vs 1.5612 |
| cross-frame direction estimate (neighbouring deployed poses) | 1.5910 — worse than the single-frame pose heading (1.5612); the dependency is unnecessary |
