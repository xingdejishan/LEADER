# MOE / rotation experiments for LEADER

Two things live here.

## `coh-rot/` - the frame-level coherent-rotation loss and its A/B

`L_coh-rot` is an extra training term for the LEADER decoder, dropped in next to the
official per-point TRR loss.  For each frame it de-centres the predicted coordinate
field and the ground-truth field with the frozen model's reliability weights, solves the
weighted Kabsch rotation between them, and penalises that residual rotation.  Whole-frame
translation cancels in the de-centring; a rigidly consistent but wrongly oriented field
does not.  The gradient goes through a first-order linearised rotation, so the SVD is
never differentiated.

Files:

    coh_rot.py            the loss (forward + analytic backward)
    train_cohrot_ab.py    the A/B driver: arm A = TRR, arm B = TRR + lambda * L_coh-rot
    leader_nclt.py        NCLT frame store (preprocessing identical to the canonical loader)
    calibrate_lambda.py   gradient-share calibration
    eval_cohrot.py        per-frame coherent-rotation statistics
    paired_bootstrap.py   significance test for an A/B
    summarize.py          result summarisation
    train_diagnostics.py  the diagnostic pass that produced train-diagnostics.json
    recal_cohrot.py       this session's re-calibration on 256 frames (see below)

Status of the evidence:

    train-diagnostics.json   arm A (TRR only)   field rotation mean 1.358 deg, median 0.906
                             arm B (+ coh-rot)  field rotation mean 1.019 deg, median 0.658
                             cost of arm B: point error 2.568 -> 2.681 m (+4%), TRR 0.792 -> 0.800
    calibration.json         the original calibration used four frames only
    recal_cohrot.py output   re-calibration on 256 frames across all four training dates

Re-calibration result (256 frames, official L0 base, four training dates):

    coherent rotation of the predicted field: mean 2.36 deg, median 1.61, p90 5.18
    40 % of frames above 2 deg; per-date means 2.24 - 2.54 (stable)
    gradient ratio |g_trr| / |g_coh| = 1.75
    lambda for a 10 / 15 / 30 % gradient share = 0.175 / 0.262 / 0.525
    the previously used lambda = 0.1 gives only 5.4 %

So the open item is not the idea but the coefficient and the evaluation: run the A/B at
lambda around 0.26 and score the *deployed* pose on the official dev split, not the
training-set field rotation.

## `diagnostics/` - why MPE keeps improving while MOE does not

Offline CPU scripts, all read-only against the frozen artefacts.  Full numbers are in
`diagnostics/RESULTS.md`.

    diag_moe.py        quantifies MPE/MOE on the full 104,728-frame set and splits the
                       rotation error into heading and tilt
    diag_time.py       tests a timestamp offset as the cause
    check_truth_noise2.py  tests whether the truth attitude is itself noisy
    expA_rot_landscape.py  cost landscape in rotation; ordering test against the truth pose
    expF_plane_cost.py     same with a point-to-plane cost
    expI_level.py          tests a gravity / levelling correction
    expK_place.py          are the errors reproducible at the same place on another day
    expM_tiltfilter.py     temporal filtering (rejected: the truth itself moves too fast)
    expN2/expO/expP/expQ/expR/expS/expT/expU2/expV  the single-frame image-rotation route:
                       rotation-only Wahba solve from the project's own 2D-3D bindings,
                       fusion sweeps, matcher variants, heading-only and gated variants
    expW/expW2_normals.py  whether surface normals carry the rotation signal (instrument
                       failed on the 0.3 m voxel map: 16 % planar, 30 deg disagreement)
    audit_history.py       which of these routes had already been tried in the project
    probe_schema.py, dbg_*.py, check_*.py, fix_cache.py, scan_shared_cache.py  support

Headline numbers (full set, LEADER V7 front end as the baseline):

    MPE 0.1876 m   MOE 1.4047 deg   median MOE 1.238
    80 % of frames are already at 0.11 m and carry three quarters of the total MOE
    the deployed geometric cost ranks the truth pose *worse* than the front-end pose
    (0.3499 vs 0.3818; truth better in only 44/180 frames), and optimising that cost
    makes MOE worse (1.40 -> 1.99)

## Running

The scripts expect the project tree at `/home/zhangcy/projects/e2e-moe-20261020-v1`
and the frozen artefacts under `/data2/zhangcy/...` as referenced in each file header.
They are CPU-only unless noted; the coh-rot scripts need a GPU and the LEADER repo.
