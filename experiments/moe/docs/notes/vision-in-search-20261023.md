# Vision inside the along-track search (2026-10-23)

Two ways to put vision into the branch itself were implemented and measured on the 1502
hard frames, the only set with a frozen visual candidate for every frame. Both reuse the
project's own search code; the shared truncated cost, the coarse/fine grid and the
rotation block are untouched. Scored against the replaced-reference truth, the same
convention as every other number on this subset.

| run | MPE | MOE | ok | >5 m | >10 m |
|---|---|---|---|---|---|
| current: vision as an alternative hypothesis (arbitration) | 0.8640 | 2.1663 | 754 | 9 | 3 |
| plain search, no vision | 1.5056 | 2.4839 | 746 | 19 | 7 |
| vision as a search window, +-4 m | 1.5230 | 2.4833 | 739 | 20 | 7 |
| vision as a search window, +-2 m | 1.5936 | 2.4875 | 715 | 20 | 7 |
| vision as a search window, +-1 m | 1.6708 | 2.4943 | 674 | 25 | 7 |
| vision as a search window, +-0.5 m | 1.7257 | 2.4967 | 642 | 31 | 7 |
| vision as an extra basin, cost decides | **1.5002** | 2.4860 | 744 | 19 | 7 |
| control: extra basin, centre jittered by +-3 m | 1.5055 | 2.4845 | 746 | 19 | 7 |
| control: extra basin, centre jittered by +-6 m | 1.5055 | 2.4839 | 746 | 19 | 7 |

## What each design does

**Search window.** The visual candidate's along-track coordinate s_vis restricts where the
grid is allowed to look: |s - s_vis| <= W. A visual prior exists on 1095 of 1502 frames
(72.9%, the rest are further than 8 m or 2 m laterally from the incoming pose). This is
strictly harmful and degrades monotonically as the window tightens. At +-1 m only 236
frames change, and 184 of them get worse.

The reason is visible in the distribution: the visual candidate sits a median 0.10 m
along-track from the incoming pose, and its own error is the same order as the
correction the search is trying to make. Restricting the search to that neighbourhood
removes the basin the search needs; the median gap between the visual and the chosen
along-track is 0.56 m but the 90th percentile is 19.5 m.

**Extra basin.** The full search still runs over +-8 m; in addition a 11 x 9 fine grid at
0.1 m is scored around the visual estimate, and the lower-cost basin wins. This can only
match or improve on the plain search. It improves the mean by 5.4 mm and takes the visual
basin on 454 frames.

The jitter controls show the gain is real visual information and not just the finer
sampling: displacing the basin centre by +-3 m drops the win rate from 454 to 28 and the
metric returns to the plain-search value. So the visual estimate is approximately right
and the fine grid polishes it.

## What this means

Vision inside the search is worth about five millimetres. Vision as an independent
hypothesis is worth 641 millimetres on the same frames. The difference is not an
implementation detail: the search is already finding the minimum of its cost, so a
finer or better-centred grid can only shave it. The large gains appear exactly on the
frames where the cost's minimum is the wrong pose, and no amount of search-side help
fixes that, because the search is minimising the quantity that is wrong.

For the write-up this is the clean argument for the architecture: the geometric expert
searches, the visual expert proposes an independent pose from a different sensor, and a
shared cost with a margin decides. The consistency observable of 2026-10-22 then says
when the geometric answer can be trusted without asking vision at all.

## Appendix: what each modality is worth on its own, and what it costs

| on the 1502 hard frames | MPE | MOE | ok | >5 m | >10 m |
|---|---|---|---|---|---|
| geometry alone | 1.5056 | 2.4839 | 746 | 19 | 7 |
| vision alone, always taking its best candidate | 23.3105 | 12.9626 | 342 | 372 | 306 |
| arbitration (vision accepted only when it beats the geometric hypothesis by 0.10) | 0.8640 | 2.1663 | 754 | 9 | 3 |
| oracle: whichever of the two is closer | 0.7904 | 2.1937 | 793 | 4 | 3 |

Vision is the better hypothesis on 375 frames (25.0%), yet used unconditionally it is
unusable. The two disagree by more than a metre on 637 frames; the arbiter picks the
better hypothesis on 97% of those. That is the argument for conditional use: the second
modality is a hypothesis generator with a score attached, not a replacement estimator.

Cost per gated frame, single-thread medians:

| stage | median | p90 |
|---|---|---|
| map crop | 26.5 ms | 39.6 |
| scan decode | 8.5 ms | 12.2 |
| along-track search (coarse + fine) | 56.1 ms | 79.4 |
| rotation block | 58.4 ms | 109.2 |
| sector + reference consistency features | 26.9 ms | 35.6 |
| normals and point-to-plane features | 104.3 ms | 175.9 |
| total with every new feature | 281 ms | — |

Gating costs 27 ms per gated frame with the consistency family, or 131 ms if the normal
estimates are included. On 7,719 gated frames that is 3.5 minutes and 17 minutes over the
whole test set, 0.3% and 1.3% of the 22-hour run. The visual branch costs 3.28 s per call
(median, p90 5.92 s), so the expensive part is entirely a question of how often it is
asked: 430 calls (the current ambiguity trigger, scaled from 5.6%) is 0.4 hours, 2,785
calls (every frame the confidence score leaves uncertain) is 2.5 hours.


- `vis_along_track.py --windows 0.5,1,2,4,inf` builds the window runs.
- `vis_basin.py --tag basin`, `--jitter 3 --tag jitter3`, `--jitter 6 --tag jitter6`.
- `score_visat.py`, `score_basin.py` score both families under the replaced-reference
  truth; the plain-search row reproduces the published 1.5056 / 2.4839 / 746 / 19 / 7.

No project file was modified. The next step, if the extra basin is wanted in production,
is to add it to the driver's search as an option and re-run the full set, since the
frozen visual table only covers the 1502.
