# Frozen pixel head with learned full 2D covariance

## Protocol

This is the distinct extension identified by the pre-run audit: freeze the final high-resolution pixel head (SHA-256 `ad058d2bd7dbfd346691c35355467888e2b9410a2948c0bda8c8294b0b109c1d`), predict a full SPD 2×2 visual precision per refined correspondence, pass those matrices into the deployed bounded LiDAR+Camera pose solver, and backpropagate final pose loss through its active-set KKT system. The older LSCR experiment jointly trained pixel offsets and diagonal sigmas, but did not train full covariance through the deployed pose backend; that overlapping experiment was not repeated.

Training used the manifest's 47 train frames (13,628 correspondences; 13,604 visible pixel labels), AdamW for 8 fixed epochs, and no development-label checkpoint selection. The frozen pixel coordinates, LEADER, RoMa, map, calibration, point selection, LiDAR information, solver bounds, and backend objective were kept fixed. The objective combines normalized final translation/rotation error and a train-normalized visible-point Gaussian NLL. The final checkpoint is the fixed eighth epoch.

A structural issue required a fixed regularization: 1,733 training peak-precision matrices had an eigenvalue below `1e-4 px^-2`, so they were not positive definite and could not define a full Gaussian covariance. The covariance parameterization floors eigenvalues at `1e-4 px^-2`, then predicts a bounded full relative precision matrix, including an off-diagonal term. This slightly regularizes those original measurements and is recorded as part of the method.

Training uses the same SciPy L-BFGS-B forward objective as inference. The implicit derivative targets the returned approximate stationary point. An initial attempt stopped at epoch 4 after one finite solve returned a non-success status; it produced no final checkpoint and was not used. The completed run restarted from the fixed initial weights and accepted finite non-success results only under a projected-KKT bound of `0.1`, set before that completed run. Three of 376 training solver calls reported a finite non-success status; all passed that bound. The maximum projected KKT residual was `0.0887` over all epochs and `0.0713` in the final epoch. Maximum Torch/SciPy objective discrepancy was `4.0e-12`. All 32 frozen development inference solves reported success.

## Results

The 32 frames are from the repeatedly used 2012-02-18 development route, not an independent test sequence. GT was read only by the separate evaluator after predictions were frozen. Metrics use the same pose-error calculation as the preceding high-resolution experiment.

| Method | Mean MPE (m) | Mean MOE (deg) |
|---|---:|---:|
| LiDAR two-stage full-pool baseline | 0.089916 | 0.902451 |
| `peak_then_pose` | 0.087139 | 0.881202 |
| Frozen pixel head, original precision | 0.087839 | 0.889047 |
| Frozen pixel head, learned full precision | 0.087172 | 0.884819 |

Against the frozen pixel head with original precision, learned covariance changes mean MPE/MOE by **−0.667 mm / −0.004229°**. The four-frame-block descriptive 95% intervals cross zero: **[−1.632, +0.346] mm** and **[−0.009342, +0.002094]°**.

Against `peak_then_pose`, the learned-precision result changes mean MPE/MOE by **+0.033 mm / +0.003617°**. Both intervals cross zero: **[−1.086, +1.131] mm** and **[−0.001384, +0.008248]°**. Thus this run does not establish a pose improvement over `peak_then_pose`.

Pixel endpoint error stayed fixed at **5.351 px**, as intended. On 5,757 GT-visible correspondences, mean Gaussian NLL (without the constant) decreased from **4.466** with regularized original precision to **4.399** with learned precision. However, the learned model was overconfident on this development set: 90% ellipse coverage was **78.9%** (95% coverage was **85.2%**), below nominal. Lower NLL alone does not establish calibrated uncertainty or better localization.

## Artifacts

- `precision_covariance_head.pt`: fixed final-epoch covariance weights.
- `training_report.json`: train-only provenance, hashes, losses, and per-frame/per-epoch solver diagnostics.
- `validation_run.json`: frozen 32-frame predictions, new precision/covariance matrices, and no GT fields.
- `evaluation.json`: separate post-freeze pose and uncertainty evaluation.
- `SHA256SUMS.txt`: file-integrity hashes.

The raw images, LiDAR, materialized match caches, and temporary resume state are not included.
