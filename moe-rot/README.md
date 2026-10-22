# MOE / rotation experiments for LEADER

Two things live here: the frame-level coherent-rotation loss with its A/B, and the
diagnostic harness that explains why MPE keeps improving while MOE does not.

## `coh-rot/` - the coherent-rotation loss and its A/B

`L_coh-rot` is an extra training term for the LEADER decoder, dropped in next to the
official per-point TRR loss. For each frame it de-centres the predicted coordinate field
and the ground-truth field, solves the weighted Kabsch rotation between them, and
penalises that residual rotation. Whole-frame translation cancels in the de-centring; a
rigidly consistent but wrongly oriented field does not. The gradient goes through a
first-order linearised rotation, so the SVD is never differentiated.

    coh_rot.py            the loss (forward + analytic backward)
    train_cohrot_ab.py    the A/B driver: arm A = TRR, arm B = TRR + lambda * L_coh-rot
    leader_nclt.py        NCLT frame store (preprocessing identical to the canonical loader)
    calibrate_lambda.py   gradient-share calibration (four frames - superseded)
    recal_cohrot.py       re-calibration on 256 frames across the four training dates
    eval_cohrot.py        per-frame coherent-rotation statistics
    paired_bootstrap.py   significance test for an A/B
    train_diagnostics.py  the diagnostic pass behind train-diagnostics.json

### What the numbers say

    train-diagnostics.json   arm A (TRR only)   field rotation mean 1.358 deg, median 0.906
                             arm B (+ coh-rot)  field rotation mean 1.019 deg, median 0.658
                             cost of arm B: point error 2.568 -> 2.681 m (+4%), TRR 0.792 -> 0.800

    recal_cohrot.py output (256 frames, official L0 base, all four training dates):
        field rotation mean 2.36 deg, median 1.61, p90 5.18; 40 % of frames above 2 deg
        per-date means 2.24 - 2.54, i.e. the term is systematic rather than accidental
        gradient ratio |g_trr| / |g_coh| = 1.75 on three batches
        lambda for a 10 / 15 / 30 % gradient share = 0.175 / 0.262 / 0.525
        the originally used lambda = 0.1 gives only 5.4 %

Read those as: there is a real, systematic, frame-level rotation residual in the
predicted field, and the training objective as shipped does not constrain it directly.
The 25 % reduction above is measured **on the training distribution, on the coordinate
field** - it says nothing yet about the deployed pose.

### Two caveats that matter for attribution

**The reliability weights are not frozen.** The A/B driver computes them every step as
`weights_from_scores(prediction[:, 3].detach())`, i.e. from the *current* model's
confidence, detached only for that backward pass. (The docstring in `coh_rot.py` claims
they are fixed for the whole run; that is the design intent, not what the A/B does.)
A coherent rotation shows up in any subset of points, so re-weighting cannot fully hide
it - but the effective point set does move during training, so "arm B improved" mixes
two effects: coordinate-field correction and confidence re-distribution. The clean
control is an arm with the weights frozen from the base checkpoint.

**`lambda = 0.262` is a candidate, not an optimum.** It comes from three gradient
batches; a larger share may buy more rotation consistency and cost more point accuracy
and MPE.

### Not self-contained

`eval_cohrot.py`, `paired_bootstrap.py` and `summarize.py` expect the A/B's `policy/`
files and the server data layout (`/data2/zhangcy/...`, the LEADER repo at
`/home/zhangcy/projects/LEADER-magic-909c81d`). Those are not in this branch, so the
evaluation cannot be reproduced from a fresh clone; only the loss, the frame store and
the calibration are self-contained.

## `diagnostics/` - why MPE keeps improving while MOE does not

Offline CPU scripts, read-only against frozen artefacts. Numbers are in
`diagnostics/RESULTS.md`.

    diag_moe.py            MPE/MOE on the full 104,728-frame set; heading vs tilt split
    diag_time.py           timestamp offset as the cause
    check_truth_noise2.py  whether the truth attitude is itself noisy
    expA_rot_landscape.py  cost landscape in rotation; ordering test against the truth pose
    expF_plane_cost.py     the same with a point-to-plane cost
    expI_level.py          gravity / levelling correction
    expK_place.py          are the errors reproducible at the same place on another day
    expM_tiltfilter.py     temporal filtering (rejected)
    expN2/expO/expP/expQ/expR/expS/expT/expU2/expV   single-frame image rotation:
                           rotation-only Wahba from the project's own 2D-3D bindings,
                           fusion sweeps, matcher variants, heading-only, gating
    expW/expW2_normals.py  whether surface normals carry the rotation signal (the
                           instrument failed on the 0.3 m voxel map: 16 % planar points)
    audit_history.py       which of these routes had already been tried in the project
    probe_schema.py, dbg_*.py, check_*.py, fix_cache.py, scan_shared_cache.py  support

Headline numbers (full set, LEADER V7 front end as baseline):

    MPE 0.1876 m   MOE 1.4047 deg   median MOE 1.238
    80 % of frames are already at 0.11 m and carry three quarters of the total MOE
    the deployed geometric cost ranks the truth pose *worse* than the front-end pose
    (0.3499 vs 0.3818; truth better in only 44/180 frames)
    optimising that cost one axis at a time and composing the three axes makes MOE
    worse (1.40 -> 1.99) - a joint 3-dof optimisation was not tested

## Running

The diagnostic scripts expect the project tree at
`/home/zhangcy/projects/e2e-moe-20261020-v1` and the frozen artefacts under
`/data2/zhangcy/...` as referenced in each file header. CPU only. The coh-rot scripts
need a GPU and the LEADER repo.


## Prior evidence - read before proposing another rotation loss

The frame-level rotation term has already been tried in the fine-tuning setting, on this
machine, and it did not transfer: on the V7 baseline the training frames improved by 10.2 %
in MOE while the held-out dates moved by 0.1 % with a confidence interval straddling zero
(transfer ratio about 1 %), and held-out MPE got 0.5 % worse. See `NEXT.md` for the table and
for what it does and does not settle. The short version: on V7 the weighted Kabsch degenerates
into a pose loss, so that experiment constrains the pose; whether constraining MMRegressor's
per-point field helps is still open, but the 1 % transfer ratio is a strong prior against it.
