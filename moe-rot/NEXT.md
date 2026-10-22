# Next step: what supervision could generalise?

Do not start another A/B yet. The closest properly powered experiment already exists and it
came back negative; repeating it would only re-measure the same null.

## Prior result: it was already run on V7, and the gain did not transfer

2026-10-10, source report `/home/zhangcy/projects/finetune-degradation-20261010-v1/FINAL_REPORT.md`
(kept in the project tree, not copied here). Three arms fine-tuned from V7 epoch-9:

    Original   no fine-tuning
    TRR-FT     the original registration loss only
    Rot-FT     the original loss plus the rotation term

10 epochs x 300 updates, 1200 training frames, 1200 held-out frames, per-frame paired,
date-stratified bootstrap.

| split | metric | TRR-FT | Rot-FT | Rot - TRR | 95% CI | significant |
|---|---|---|---|---|---|---|
| train | MOE (deg) | 0.9440 | 0.8473 | -0.0967 | [-0.1036, -0.0901] | yes, -10.2 % |
| dev | MOE (deg) | 1.2885 | 1.2869 | -0.0017 | [-0.0071, +0.0039] | no, -0.1 % |
| dev | MPE (m) | 0.1378 | 0.1385 | +0.0007 | [+0.0003, +0.0011] | yes, +0.5 % worse |

Transfer ratio about 1 %. Over the whole trajectory the two arms diverge monotonically on
the training frames (4.0 -> 9.9 percentage points of MOE) and stay within 1.2 points on the
held-out dates, with no over-fitting shape: both arms improve monotonically on both splits.

That report also explains part of the mechanism: on V7 the weighted Kabsch **degenerates
exactly into a rotation pose loss** - both sides are rigid transforms of the same input
points, so the solution is `R_p^T R_t` for any weights and neither the point set nor the
weights enter. In that setting the term constrains the pose, not the correspondence field.

What it settles: adding a rotation term when fine-tuning a *pose-output* model does not
improve held-out MOE.

What it does not settle: constraining the *MMRegressor's per-point field* (`coh-rot/`, trained
from the official L0 base), where the Kabsch is not degenerate because the predicted points
are learned per-point regressions rather than a rigid transform of the input. That A/B was
measured on the training-side field rotation only; `runs/` and `eval/` are empty.

## What to do instead, with the artifacts that already exist

The V7 report's own third experiment says the learned increment is date-specific
(significant on the fitted frames and on same-day unfitted frames, zero on held-out dates)
and that the required correction is "almost not determined by the place". Those two
statements together are the real open question: if the needed rotation correction is not a
function of the place, what is it a function of?

Probe, in this order, all on existing per-frame dumps, no training:

1. Regress the needed rotation correction (per frame, in the same frame convention as the
   dumps) on single-frame observables that could plausibly carry it: the spread and
   anisotropy of surface normals, the number of distinct planes, the range distribution, the
   angular rate during the sweep, how many references come from one date, the residual of
   the current field fit. Report R^2 and the out-of-date generalisation of the fit.
2. If nothing predicts it with a useful R^2, stop. That is an answer: the residual rotation
   error carries no single-frame signature, so no trained constraint on the fitted
   distribution will transfer, and the honest move is to re-scope the claim (MPE and the
   tail counts) rather than to train again.
3. Only if something does predict it should that observable become a training target - and
   then the A/B in the previous version of this file is the right protocol, with the
   held-out dates as the only criterion.

Constraint that must not be relaxed: single frame. Every candidate observable above is
computed from one frame, because the deployment path has to answer per frame.
