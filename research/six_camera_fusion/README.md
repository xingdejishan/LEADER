# Controlled six-camera LEADER fusion

Branch: `research/leader-six-camera-controlled`. This uses the existing 96-frame six-camera data with the same 64 training / 32 development split. It does not substitute the larger 905/301/148 data, for which this experiment has no six-view feature preparation.

| Trained arm | Visual input | Aggregation | Updated parameters |
|---|---|---|---|
| LiDAR control | None | None | Original decoder `pred_out` |
| Cam5 | Cam5 only | Single view | Same `pred_out` + common residual gate |
| Six-camera mean | All valid cameras | Equal weights | Same `pred_out` + same residual gate |
| Six-camera query | All valid cameras | Geometric feature queries image keys and camera-frame bearings | Same `pred_out` + same residual gate + attention parameters |

The untrained original LEADER is also evaluated as a reference, not a fifth training experiment. Cam5 vs mean controls the architecture while changing available cameras. Mean vs query changes aggregation and additional parameter capacity, so it is not a parameter-count-matched proof of attention's benefit. Both have identical visible voxel unions and candidate points.

All arms share the original LEADER checkpoint, cached voxel features/local coordinates, DeDoDe-B encoder, PCA256->128, labels, frame/point schedules, optimizer, final-step selection and two-stage solver. Training uses three paired seeds (2089, 2090, 2091), 600 steps per arm/seed, Adam 1e-4, up to 1,024 points per step and 10% whole-frame visual dropout. The encoder, PCA and decoder before `pred_out` remain frozen. PCA was fitted in the earlier 907-image training set, shared across arms, and not refitted here. No Gaussian prior, covisibility loss, ASQB, extra coordinate head or new correspondence is added.

Queries select among per-camera descriptors for an existing voxel. Camera-frame bearings use its raw representative and calibration, without GT/world pose. Invalid views are masked before normalization/pooling. Common gates start identically within each seed with zero residual output. Independent per-step NumPy schedules make sampling invariant to extra parameters/RNG consumption. All voxel rows remain in the coordinate loss and solver, including those lacking images.

The historical cache stores the mean descriptor and per-view masks, not individual descriptors. Extraction materializes all six descriptors, checks exact mask/coverage parity and compares the regenerated mean before training. Targets: 67,710 total voxels, 14,728 Cam5-visible, 45,133 visible in the union, 10,125 supported by multiple cameras. This does not constitute manual association ground truth.

Every pose uses upstream SC2 top50% and original `full_pool_refine` at 1.2m/0.6m, with a fixed solver seed per frame across all methods/training seeds. The frozen baseline must reproduce the earlier same-frame two-stage errors. Query missing-image/shuffled-correspondence diagnostics need no extra training. Shuffling preserves each camera's valid mask and descriptor distribution. Missing images return the adapted decoder's LiDAR-only output, not necessarily the original pretrained pose.

```bash
bash research/six_camera_fusion/launch.sh all
```

Stages: `check`, `prepare`, `extract`, `train`, `evaluate`, `report`; outputs: `/home/zhang/leader-six-camera-controlled`. Image extraction uses `rscore-l`; training/solving uses `egonn118`. Reports retain every seed and give mean±sample standard deviation, never the best seed. Repeating the same 32 frames does not create 96 independent evaluation samples.

This previously touched development date participated in LEADER pretraining. The preregistered pass condition requires both mean translation and rotation errors to beat the matched-budget LiDAR control; success counts and missing/shuffled images are reported separately. Local results do not establish cross-date generalization.
