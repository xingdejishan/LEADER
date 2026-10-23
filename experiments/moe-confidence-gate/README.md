# MoE relocaliser with a confidence gate

Two pieces of code and one measured result.

`algorithms/` is the mixture-of-experts localiser. One objective, three decisions:
a static LiDAR map, a robust truncated cost

    C(T) = mean over the scan of min( dist(T x, M), 1.0 m )

a coarse 17 x 3 along-track/lateral surface built once per frame, an adaptive fine
stage, an ambiguity ratio, and an optional visual hypothesis that replaces the search
pose only when it is cheaper by more than a margin. `CFS` is the search, `MoE` is the
whole chain.

`confidence/` answers the question the search cannot answer about itself: given that
the search moved the pose, was that move right? It re-measures the same cost gain on
disjoint evidence — eight azimuth sectors of the scan and each reference cloud
separately — and turns the result into a score. Low score: accept. High score: the gain
did not reproduce across independent views of the scene, so the move is probably the map
bias rather than the truth.

## Result on the full NCLT test split (104,728 frames)

Leave one test date out; the gate never sees the date it is scored on.

| policy | MPE | MOE | ok(1 m, 2 deg) | >1 m | >5 m | frames made worse |
|---|---|---|---|---|---|---|
| leader front end, no correction | 0.18755 | 1.404661 | 80.20% | 685 | 290 | — |
| accept every correction (baseline) | 0.18433 | 1.401758 | 80.59% | 366 | 284 | 4051 |
| **confidence gate** | **0.18041** | **1.401758** | **80.60%** | **353** | **284** | **1501** |

3.9 mm lower mean position error, 63% fewer frames made worse, identical orientation,
13 fewer frames above 1 m. The gate decides on 7,719 frames (7.4% of the split), keeps
55% of the corrections, and costs 27 ms per decided frame on top of a 150 ms search.

Ranking quality: AUC 0.766 leave-one-date-out, against 0.545 for everything the
pipeline already computes. Shuffled-label controls return 0.497 / 0.518 / 0.468.
Sector and reference consistency alone reach 0.725.

## Why the second modality is worth its cost

On the 1502 hard frames, with a frozen visual candidate per frame:

| | MPE | >5 m | >10 m |
|---|---|---|---|
| geometry alone | 1.5056 | 19 | 7 |
| vision alone, always its best candidate | 23.3105 | 372 | 306 |
| arbitration by the shared cost | 0.8640 | 9 | 3 |
| oracle, whichever is closer | 0.7904 | 4 | 3 |

Neither modality stands alone; the pair, arbitrated, lands within 0.07 m of the oracle
and resolves 97% of the frames where the two disagree by more than a metre. That is the
argument for conditional invocation, and the confidence score is how the condition is
decided. Details in `docs/vision-in-search-20261023.md`.

## Layout

    algorithms/          CFS (search) and MoE (whole chain); numpy + scipy only
    confidence/features.py   the cross-view consistency features
    confidence/gate.py       linear score and the accept rule
    confidence/coefficients.json   fitted mean, deviation, weights, bias, policy
    docs/                the two measurement notes

## Use

```python
from algorithms import MoE, CFS
from confidence.features import features
from confidence.gate import Gate

moe = MoE(refs, positions)                 # refs: [{'scan_path', 'world_from_body'}]
pose, diag = moe.localize(scan, T0)        # search, same cost, rotation preserved

f = features(diag['tree'], diag['map_points'], scan, T0, pose,
             diag['along_track'], diag['lateral'],
             surface=diag['surface'], s=diag['s'], d=diag['d'], refs=diag['ref_clouds'])
accept, score = Gate().accept(f)
pose = pose if accept else T0
```

`features()` needs the tree, the map and the surface the search already built; it
recomputes none of them.

## Provenance

- `confidence/features.py` is an extraction of the code that produced the numbers. It
  was re-run against the stored extraction on four frames spanning three dates and all
  39 features matched exactly (0 mismatches); the reproduction script is not shipped
  because it needs the internal data layout.
- `confidence/coefficients.json` is fitted on every triggered frame of the test split,
  which gives an in-sample AUC of 0.802. The end-to-end table above uses
  leave-one-date-out scores (AUC 0.766), so those numbers are the honest ones and the
  shipped coefficients are slightly optimistic on their own training split.
- The gate policy (`score < 0.3`, or any correction larger than 1.5 m) was chosen on the
  full split; all four dates improve individually under it, the largest single
  regression being one additional frame above 1 m on 2012-02-19.

## Negative result worth knowing

Putting vision inside the along-track search does not work. Restricting the search to a
window around the visual estimate is monotonically harmful (1.5056 to 1.7257 as the
window tightens from 8 m to 0.5 m); adding the visual estimate as an extra basin and
letting the shared cost choose is worth only 5 mm. The visual candidate sits a median
0.10 m along-track from the incoming pose, with errors of the same order as the
correction being made, so it cannot constrain a search that is already minimising the
right functional. See `docs/vision-in-search-20261023.md`.
