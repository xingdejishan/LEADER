# MOE candidate selection: what was tried and what it cost

Two documents and the scripts behind them.

- `MoE-experiment-ledger.md` — every hypothesis, method and negative result measured on
  the 1502 hard frames, each with the number that decided it.
- `candidate-coverage-research.md` — the cross-domain mechanism survey (18 papers,
  6 directions), the de-duplication against earlier work, and the one surviving design.

## What the data says, in one paragraph

The rotation error of the deployed pose is a *distribution* problem, not an estimation
problem: the median rotation error is pinned at 1.81–1.86 deg under every configuration
tried, while the mean moves. The only lever that produced real movement is replacing a
whole pose by a visual candidate on the frames where the second source is decisively
better, and that gain equals (fraction of frames replaced) x (their mean excess over the
floor). Everything else — yaw sweeps, point-to-plane Gauss-Newton, plane and vanishing
point structure, pole landmarks, robust rotation averaging, free-space consistency,
cost arbitration — was measured and came back at or below the noise.

## The one positive result in mechanism space

Cross-candidate consensus — counting how many other candidates from *disjoint reference
sets* agree with a candidate in pose — separates frames that hold a usable candidate
from frames that do not, with **AUC 1.000** on the 55 dead frames (41 frames with no
usable candidate all score 0; the 14 that have one all score >= 1).

Used as a frame-level gate it does not move the metric by much, because only those 14
frames can be rescued: end to end it gives MPE 0.9628 / MOE 2.0885 against 0.9706 /
2.1065 for the shipped rule. Used to *pick* the candidate instead of the score it fails
(MPE 1.83–3.17), and used with no map score at all it fails badly.

## Scripts

All read-only against the frozen artifacts; each states its inputs at the top. They
expect the absolute paths of the original machine (`/data2/zhangcy/...`), which is where
the frozen tables live; edit `MOE`/`MATCH`/`VERIFY` at the top of each script to run
elsewhere.

| script | question |
|---|---|
| `rot_decomp.py`, `rot_worth.py` | axis decomposition of the rotation error |
| `bias_test.py`, `structure.py`, `yawrate.py` | is the floor a bias, or does it correlate with anything observable |
| `yaw_ident.py`, `yaw_reach.py` | is the true yaw identifiable, and is it reachable from the deployed pose |
| `obj_compare.py` | does any objective rank yaw better than the prior already does |
| `gn_refine4.py` | point-to-plane Gauss-Newton, 4-DOF |
| `plane_consistency.py`, `vp_v2.py`, `pole_v3.py`, `pole_repeat.py` | plane, vanishing point and pole-landmark routes |
| `feature_auc.py`, `profile*.py` | per-candidate feature AUCs and the true oracle |
| `consensus_*.py`, `cons_gate.py`, `cons_only.py`, `live_cost.py` | cross-candidate consensus as a gate and as a selector |
| `gbm_rule.py`, `gbm_hybrid.py`, `gbm_ctx.py` | learned selector over the existing features |
| `fs_compute.py`, `fs_eval.py` | relative free-space consistency |
| `points_needed.py`, `two_stage.py`, `hybrid_arb.py` | how few query points the LiDAR score needs, and the two-stage arbitration |
| `dead55.py`, `gated84.py`, `gated_split.py`, `diversity.py` | anatomy of the 84 gated frames and their 55 unreachable ones |

## Not finished

The one mechanism that is designed but not measured: **multiple pose hypotheses from
coherent subsets of the aggregated 2D-3D correspondences** (HLoc-style, but keeping
several hypotheses instead of one). The aggregated channel already exists in
`match_prov.py` (`channel == 'full'`) and was the best arm of the ablation, but it emits
one PnP per frame, so only 13% of all candidates come from it. Producing several
hypotheses from different coherent subsets is untested, and the frozen artifacts do not
store the correspondences, so the experiment needs one re-run of the matcher with a dump.
