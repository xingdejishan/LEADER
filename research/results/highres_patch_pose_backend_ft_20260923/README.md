# High-resolution patch head fine-tuning with final-pose supervision

## Protocol

This run continues from the frozen high-resolution patch head (`fine_tune_from_sha256` is recorded in `training_report.json`). It trains only that head; LEADER, RoMa, map geometry, calibration, correspondence selection, LiDAR information prior, visual precision matrices, and deployed pose bounds remain fixed. Each training step uses all valid camera correspondences from one frame and produces one shared six-degree-of-freedom pose.

The forward pose is computed by the existing SciPy L-BFGS-B backend. Its objective is precision-weighted quadratic visual reprojection plus the quadratic LiDAR information-matrix prior and the existing negative-depth penalty, with per-axis bounds of ±0.1 m and ±1 degree. The actual objective has no robust-loss term, so none was added. Backpropagation uses an active-set KKT implicit derivative of this bounded objective.

Training used 47 manifest train frames (13,628 correspondences, 13,604 visible pixel labels), 8 fixed epochs, AdamW at 1e-4, weight decay 1e-4, gradient clipping at 1, and pixel-loss weight 0.1. Pose and pixel loss scales were computed from train data only. The final epoch was kept without validation-based stopping or checkpoint selection. Validation GT was not read during training or inference; it was read only by the separate evaluator after the run was frozen.

## Results

All pose metrics use all 32 frames from the repeatedly used 2012-02-18 development sequence. This is not an independent-sequence test.

| Method | Mean MPE (m) | Mean MOE (deg) | MPE P90 (m) | MOE P90 (deg) |
|---|---:|---:|---:|---:|
| LiDAR two-stage full-pool baseline | 0.089916 | 0.902451 | 0.133623 | 1.408789 |
| `peak_then_pose` | 0.087139 | 0.881202 | 0.129616 | 1.392802 |
| Fine-tuned high-resolution head | 0.087839 | 0.889047 | 0.133247 | 1.391690 |

Against `peak_then_pose`, the fine-tuned head changes mean MPE by +0.700 mm and mean MOE by +0.007846 degrees. The contiguous four-frame block-bootstrap descriptive 95% intervals cross zero for both metrics: [−1.331, +2.539] mm and [−0.00259, +0.01559] degrees. The result therefore does not establish a pose improvement over the peak path.

Pixel endpoint error did improve: mean 7.506 px to 5.351 px over 5,757 GT-visible correspondences (4,913 improved, 844 worsened). This pixel gain did not establish a pose gain. On the training set, final mean MPE/MOE were 0.076986 m / 0.717041 degrees. Forward objective parity against the deployed solver was at most 9.95e-13; the mean/max projected KKT residual used for implicit differentiation was 0.02998 / 0.05315, so the derivative is local to the returned approximate optimum.

## Artifacts

- `highres_patch_pose_head.pt`: final epoch weights only; intermediate epoch checkpoints and input caches are not included.
- `training_report.json`: data hashes, fixed protocol, per-epoch/per-frame training records, and final checkpoint hash.
- `validation_run.json`: frozen GT-free inference output, per-frame solver records, and input hashes.
- `evaluation.json`: separate post-freeze GT metrics and paired comparisons.

The input manifest, image data, LiDAR caches, and materialized feature/match caches are not included. Their hashes are recorded in the reports.
