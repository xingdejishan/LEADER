# Next experiment: does a training-time rotation constraint reach the deployed pose?

The open question is not whether the coherence loss reduces the rotation of the predicted
coordinate field - it does, by ~25 % on the training dates. The open question is whether
that reduction survives the pose solver (consensus sampling plus weighted Procrustes on a
confidence-selected subset) and shows up as deployed MOE on unseen dates.

## Protocol

Base: the same checkpoint, the same training dates and budget, the same solver, the same
evaluation, for every arm. Deviating on any of these makes the comparison unreadable.

    arm A   TRR only                              (control)
    arm B   TRR + lambda * L_coh-rot              (lambda ~ 0.26, i.e. ~15 % gradient share)
    arm C   TRR + lambda * L_coh-rot, weights frozen from the base checkpoint
            (separates "the field was corrected" from "the confidence moved")

Evaluate on the official dev split (300 frames per date, four dates that are not in the
training set) and report, per arm:

    deployed MOE and MPE (median and mean), plus the fraction within 1 m / 2 deg
    the coherent rotation of the predicted field
    the skip rate of the loss (angle gate and degeneracy) - if it silently skips most
    frames the run means nothing

## Decision rule

Proceed to a full training run only if

    deployed MOE falls significantly (paired bootstrap, p < 0.05) on all four dev dates,
    and MPE does not regress by more than 2 %,
    and arm C shows the same direction as arm B (so the effect is the coordinate field,
    not the confidence distribution).

If MOE does not move, the residual 1.4 deg is not in the field rotation, and the whole
"train the rotation" line should be dropped rather than tuned.

## Before the run

    re-calibrate lambda on the *base checkpoint being trained from* (the 1.75 gradient
    ratio above was measured on the official L0 base, not on a fine-tuned one)
    check whether `research/coh-rot-mini` on this repository already contains this
    experiment, and if so, start from its conclusions instead of repeating it
