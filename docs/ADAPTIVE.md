# Adaptive fine-window expansion

Starting from the four diagnostics proposed for the fine stage, this records what
the data actually says and the rule that came out of it. The rule is now the
recommended replacement for the previous gradient pruning, which turned out to be
unsound.

All numbers are on the 1502 hard frames, kept set = 1266.

---

## 0. A correctness problem in the previous pruning

`fast_lib.solve` pruned a fine window using a per-point "gradient slack":
`slack_i = 0.5*|g_i·u| + 0.4*|g_i·v|`, then
`bound = mean(min(max(d_i - slack_i, 0), 1))`. Points whose centre distance
exceeded the 1 m truncation were counted as contributing exactly 1.

Both parts are wrong:

- the distance function is not linear over a 0.64 m step, so
  `d(x+δ) >= d(x) - δ·g` is not a valid bound;
- a point beyond 1 m from the centre can move inside 1 m, so treating it as a
  constant contribution of 1 over-estimates the region's lower bound.

The failure is measurable: the median gap between the best and second-best
candidate is **0.11 m**, and the largest gap over all frames is far below the
window radius **ρ = 0.6403 m**. A valid Lipschitz bound `centre − ρ` therefore
never prunes anything (0 of 9012 windows). Yet the implementation pruned
270 → ~108 evaluations. So the 60 % saving came from an invalid bound.

That the earlier before/after comparison showed *identical* results is explained
by the next statistic: the winner is almost always the top-ranked candidate, so
the wrongly pruned candidates rarely mattered. It was luck, not soundness.

## 1. Which candidate actually wins

| coarse rank of the cost-optimal candidate | frames | share |
|---|---|---|
| 1 | 1412 | 94.0 % |
| 2 | 41 | 2.7 % |
| 3 | 27 | 1.8 % |
| 4 | 13 | 0.9 % |
| 5 | 6 | 0.4 % |
| 6 | 3 | 0.2 % |

Ranks 4–6 win 1.5 % of frames. By *truth* error instead of cost, rank 1 wins
90 %, ranks 4–6 win 4 %.

**Cost-optimal equals error-optimal in only 11.5 % of frames** — the geometry
cost is a weak proxy for position accuracy, which is why every "search harder"
variant has failed to translate into better poses.

## 2. Does the window help over its centre?

Median drop from centre cost to the best cost inside the window: **0.0247**;
90th percentile 0.0046. The fine surface is nearly flat around its centre, and
the best position is interior 78 % (along-track) / 88 % (lateral) of the time —
consistent with a flat surface where small differences choose an arbitrary
interior point. The centre itself is optimal in only 13 % of cases, but the cost
advantage of moving is tiny.

## 3. The rule that came out of it

Score the six centres at 4000 points, order by that cost, and expand a full
window only while its centre is within `thresh` of the best centre:

| thresh | windows expanded | median cost | median err | ≤1 m | frames worse than full by >0.5 m |
|---|---|---|---|---|---|
| 0.02 | 1.06 | 0.305 | 0.305 | 93 % | 7 |
| 0.05 | 1.20 | 0.304 | 0.304 | 93 % | 1 |
| **0.10** | **1.63** | 0.304 | 0.304 | 93 % | **0** |
| 0.20 | 3.04 | 0.304 | 0.304 | 93 % | 0 |
| none (all six) | 6.00 | 0.304 | 0.304 | 93 % | — |

Hard buckets behave the same: at 0.05 the 3–5 m band uses 1.47 windows and the
5–8 m band 2.29, with median error and success rate identical to full expansion.

The rule is a **heuristic, not a bound**. It can in principle skip a window that
would have won; on this data at thresh = 0.10 that never cost more than 0.5 m.

## 4. End-to-end

`src/adaptive.py`, full 1502 frames:

| thresh | MPE | median | ≤1 m | windows | exact evals |
|---|---|---|---|---|---|
| 0.05 | 1.5645 | 0.3522 | 83 % | 1.17 | 59 |
| **0.10** | **1.5638** | **0.3513** | **83 %** | **1.64** | **79** |
| previous (unsound pruning) | 1.5638 | 0.3513 | 83 % | ~2.4 | 108 |
| all six windows | 1.5638 | 0.3513 | 83 % | 6.00 | 270 |

At thresh = 0.10 the aggregate result is identical to full expansion while using
29 % of the exact evaluations.

Per-frame timing at thresh = 0.10: centre re-score 15 ms, fine search 79 ms,
coarse 31 ms, scan preparation 21 ms → **~146 ms**; with the GPU coarse stage
(1.5 ms instead of 31 ms) **~116 ms**.

## 5. Why this still does not reach 50 ms

A single fine window is 45 poses × 4000 points ≈ 45 ms on its own, and the rule
already averages 1.64 windows — close to the floor of one. To reach 50 ms the
single window would have to cost under ~20 ms, i.e. about ten poses per window,
which is the five-point cross that was already measured to lose accuracy
(docs/RESULTS.md, section 5).

So the honest landing for this line is **~116 ms at zero accuracy cost**, down
from 162 ms. A 50 ms budget requires changing what is evaluated inside the
window, not how many windows are evaluated.

## 6. Reproducing

```
src/adaptive.py            the rule (library function + full-set driver)
analysis/log_fine.py       records the 6x45 cost matrix and per-pose truth error
analysis/analyze_stats.py  every statistic quoted above
```

`analysis/log_fine.py` writes `fine_stats.npz` (~10 MB) which the analysis
scripts read; it runs in about 64 s on 12 processes.
