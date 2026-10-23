# A working discriminator for the geometric correction (2026-10-23)

Question: is there an observable, computable at decision time, that separates a
correction that helps from one that hurts? Yes. It is not the size of the correction,
the cost drop, or the local geometry; it is whether the cost drop reproduces across
*independent views of the same scene*.

## The observable

The search scores a candidate pose against one merged map with one query set. That
single number can improve while the pose moves further from truth. The new observables
measure the same improvement several times over disjoint parts of the evidence:

- eight azimuth sectors of the scan, each scored separately at the incoming and the
  corrected pose: a real correction improves most sectors, a bias-driven one improves
  the sectors that happen to face the map's weak direction and hurts the rest
- the three reference scans scored one at a time, and their minimum and spread across
  the triple: a gain carried by a single reference is suspect

Nothing here needs truth, extra scans, or a second map.

## Separating power (7,719 triggered frames, all of NCLT)

| feature set | AUC (leave one date out) |
|---|---|
| what the pipeline already computes (cost, gain, shift, surface shape, residuals) | 0.545 |
| local geometry (point-to-plane information, weak direction, inlier fraction) | 0.581 |
| cross-view consistency (sectors + references) | **0.725** |
| everything | **0.766** |

Univariate: the worst sector gain has AUC 0.655, the median sector gain 0.662, and the
weakest reference gain 0.665 — each as strong as the best of any old feature.

Controls: with the labels shuffled three times the same protocol returns 0.497, 0.518,
0.468. So the 0.766 is not a leak.

Calibration on the held-out scores is monotone, which is what makes it usable as a gate:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| actual damage rate | 0.07 | 0.26 | 0.37 | 0.42 | 0.51 | 0.61 | 0.68 | 0.74 | 0.74 | 0.85 |
| mean change | -0.79 | -0.10 | -0.04 | -0.03 | +0.01 | +0.06 | +0.08 | +0.11 | +0.10 | +0.14 |

## End to end (whole test set, leave one date out, unmodified driver output)

The configuration that clears both metrics: gate the **translation** by the score, but
keep the **rotation** block on every searched frame, and always accept a correction
larger than 1.5 m regardless of the score (large corrections are the reliable ones and
they hold the tail down).

| policy | MPE | MOE | ok(1 m, 2°) | >1 m | >5 m | damaged | helped | MPE on replaced truth |
|---|---|---|---|---|---|---|---|---|
| front end | 0.18755 | 1.404661 | 80.20% | 685 | 290 | — | — | 0.16919 |
| accept all (current) | 0.18433 | 1.401758 | 80.59% | 366 | 284 | 4051 | 3668 | 0.16595 |
| **final** | **0.18041** | **1.401758** | **80.60%** | **353** | **284** | **1501** | 2773 | **0.16204** |

Damage falls by 63%, three quarters of the useful corrections are kept, the mean
position error improves by 3.9 mm, the mean orientation error is bit-identical, the
success rate edges up 0.01 points, and the tails are 13 frames better above 1 m and
unchanged above 5 m. Per date: 0.24356→0.23844, 0.14664→0.14077, 0.14432→0.14139,
0.20460→0.20282; the only regression anywhere is one extra frame above 1 m on 02-19.

Two things were needed to protect the orientation. The rotation refinement is valid
only on the pose it was fitted at: refitting it at the incoming pose is worse than not
rotating at all (1.40496 against 1.40466), while keeping the rotation from the searched
pose while rejecting its translation is better than either (1.40176). So the gate must
act on the translation alone and the rotation block must run unconditionally. That is a
one-line change in the driver's ordering, not a new estimate.

Earlier intermediate rows, for the record: gate on translation only, rotation dropped,
gave 0.18104 with MOE 1.40333 and 293 frames above 5 m.


## A free by-product

Applying only part of the accepted correction helps on its own, with no new observable:
scaling the whole offset (translation and rotation together) by 0.6 gives 0.18210 with
MOE 1.40216, by 0.8 gives 0.18257 with MOE 1.40176. Neither beats the gate, and both
raise the count above 1 m to 446 and 386, so the gate is the better trade.

## Cost

The consistency features reuse the map, the trees and the scan subset the driver already
has. Per gated frame the added work is sixteen short tree queries on 250-point sectors
plus two queries per reference scan. Since the search only triggers on 7.4% of frames,
the whole gate costs well under a minute across the full test set, against a 22-hour
baseline. The gate is a 39-coefficient linear score; the coefficients and the
standardisation must be shipped with the driver.

## Files

Scripts in `/home/zhangcy/.dsh_scratch/`: `observables.py` (feature extraction, replays
the driver bit-exactly), `analyze_obs.py`, `ablation.py` (family ablation),
`rot_at_t0.py` (rotation block at the incoming pose), `rot_choice.py`, `gate_shrink.py`,
`operating_point.py`, `verify_final.py` (the final table above), `validate_gate.py`
(shuffled controls and calibration).
