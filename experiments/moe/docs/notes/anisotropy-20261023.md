# Where the front-end residual lives, and where the along-track search pays (2026-10-23)

Full test split, 104,728 frames, replaced-reference truth. This note exists because the
number everyone quoted — "95% of the residual is along-track" — came from a
difficulty-conditioned subset (frames whose front-end error exceeded 0.5 m). On the full
split the picture is much weaker, and the search's usefulness turns out to depend on the
*magnitude* of the along-track residual, not just its direction.

## Per-direction residual of the front end

Error vector decomposed in the front-end pose's own axes, mean absolute value, metres.

| date | n | along | lateral | vertical | rotation | along share |
|---|---|---|---|---|---|---|
| 2012-02-12 | 25,546 | 0.100 | 0.082 | 0.060 | 1.52° | 41.4% |
| 2012-02-19 | 26,543 | 0.094 | 0.071 | 0.048 | 1.35° | 44.1% |
| 2012-03-31 | 26,201 | 0.091 | 0.075 | 0.052 | 1.40° | 41.8% |
| 2012-05-26 | 26,438 | 0.146 | 0.094 | 0.064 | 1.36° | 47.9% |
| **all** | **104,728** | **0.108** | **0.081** | **0.056** | **1.41°** | **44.1%** |

The along-track direction is the largest single component but not a dominant one. What
*is* solid is the stability across dates (41–48%), so the anisotropy is a property of the
estimator rather than of one subset. The 95% figure should not be quoted for the full
split.

After the geometric correction the same decomposition reads 0.105 / 0.080 / 0.056 — the
correction moves the along-track component by 3 mm on the full split, which is why the
end-to-end MPE changes by 3.2 mm while 319 frames cross the 1 m line.

## Where the search actually helps

7,719 triggered frames, bucketed by the along-track component of the front-end error:

| along-track residual | frames | helped (>1 cm) | mean change |
|---|---|---|---|
| 0 – 0.1 m | 4,342 | 32% | **+0.087 m** |
| 0.1 – 0.3 m | 2,497 | 53% | +0.002 m |
| 0.3 – 1.0 m | 595 | 85% | −0.363 m |
| 1.0 – 3.0 m | 226 | 98% | −1.368 m |
| > 3.0 m | 59 | 88% | −3.368 m |

Monotone, and it says something the subspace argument needs: the along-track direction is
the right one to optimise, but only above roughly 0.3 m of residual. Below that the
correction is noise-driven and costs more than it buys.

## Observability does not discriminate

The along-track curvature of the coarse surface is above 0.01 on every one of the 7,719
triggered frames, so "is the map locally informative along this direction" is always true
inside the search window and cannot be used as a gate. What does discriminate is the
*global* question — whether a competing position fits as well — which is exactly what the
ambiguity ratio measures, and it fires on all three of the 444 m frames (0.145 / 0.155 /
0.169, all below the 0.21 threshold).

## Consequence for the subspace framing

The anisotropic-prior argument survives but has to be stated conditionally:

1. the residual is anisotropic, along-track is its largest component (44%), and the
   ordering is stable across dates;
2. optimising that subspace pays only when the component exceeds ~0.3 m;
3. the subspace is only meaningful where the along-track direction is globally
   unambiguous — the local curvature is not a usable gate.

Condition 3 is the router's job, not the optimiser's.

## Rotation block: removed, and what it cost

The rotation block (translation frozen, 0.5°/step, held-out acceptance) was removed from
both entry points on 2026-10-23. On the full split it was worth 0.0029° of MOE at zero
cost to MPE; on the 1502 hard frames it is worth more than that:

| 1502 hard frames | with the block | without |
|---|---|---|
| MPE | 0.8640 | 0.8598 |
| MOE | 2.1663 | 2.2017 |
| ok (1 m, 2°) | 754 | 745 |
| >5 m | 9 | 8 |
| frames switched to vision | 18 | 20 |

So the removal trades 9 success frames and 0.035° of MOE for a 4 mm MPE gain on the hard
subset. Recorded rather than hidden: it is the one change in this series that does not
strictly satisfy "no accuracy regression".

## Reproduce

    python anisotropy.py                      # per-direction residual and the buckets
    python three_route_unified.py --workers 8 --out results/_norot_check.npz
