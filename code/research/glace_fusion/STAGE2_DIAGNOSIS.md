# GLACE stage 1 archive and stage 2 diagnostic

Stage 1 ended at the user's request on 2026-09-12. The selected checkpoint is the evaluated **80,000-step head**, not the last in-memory training state (84,201 steps when stopped).

- Remote archive: `/root/rivermind-data/glace_nclt_rgb_large_20260912/stage1_80k/`
- Head SHA256: `9d56719ba7f96bd442686f2f870a5e89f513d52d61b9b41bd9bfba5abaf0b507`
- `head.pt` and adjacent `config.json` must remain together.
- `manifest.json` identifies the checkpoint and the original paired evaluation.
- This is a head checkpoint; optimizer/scheduler state was not saved.

The original 80k test64 comparison used all output grid points, matching 35k. Training GT q10 improved from 5.02% to 12.67%; test GT q10 improved from 1.68% to 2.48%. Half-credit pairwise accuracy was 50.65% for 0.2–0.5m, 52.99% for 0.5–1m, and 55.56% for 0.5–1°. This is one test date and does not establish improved LEADER candidate selection.

## Confirmed issue and bounded change

Official NCLT U2D maps leave invalid image borders. The previous loader initialized `image_mask` to ones, so those positions could enter the reprojection training buffer. The base 480-pixel-high grid contains approximately 37% invalid FOV points. This is not a claim that their entire encoder receptive fields contain only black pixels, nor a measured fraction of the augmented training buffer.

`valid_region.py` builds a validity map by remapping a white source image with the same official maps and resizing it to the stored image dimensions. It never uses RGB brightness to classify pixels. A conservative bilinear resize and rotation of this mask follows local image augmentation; image intensities, K, poses and RGB global features are unchanged.

`patch_valid_region.py` patches a **new vendor copy**. The dataset optionally reads `scene/train/valid_mask.npy`. The training buffer samples the mask at the actual GLACE pixel centers `8*(x+0.5), 8*(y+0.5)` instead of resizing the mask to the output shape, which otherwise shifts the boundary. Output centers outside the image are invalid. Images without valid output centers fail explicitly.

`mask_preflight.py` checks 18 actual-loader cases: three images, three resolutions and augmentation enabled/disabled. The masks must agree with output-grid indexing; images, K, poses and global features must remain identical with and without the mask.

`evaluate_scene --valid-mask MASK.npy --pose-backend none` evaluates only valid correspondence. The manifest records the mask hash and each record gives the correspondence count. Omitting the flag preserves the original evaluation population. Do not compare masked and all-grid results as if they used the same denominator. Masked pose estimation is not implemented by this option.

## Local mask ablation

Run root: `/root/rivermind-data/glace_nclt_stage2_local_mask_20260912`

Two heads train from scratch on the same 256 images in a radius-40m region, with camera orientation within 30 degrees of the anchor. The region is chosen deterministically by train/test pose coverage, without inspecting model scores. The 64 nearby images from the existing 2012-02-12 test date remain excluded from training. They are a development probe, not an untouched final holdout. Pose proximity does not certify actual co-visibility.

Both arms retain RGB R2Former features, local height 480, feature diffusion 0.1, K=50, three head blocks, mlp_ratio=2, soft clamp 50 and seed 2089. Each uses 5,000 iterations, batch 8,192 and 1,024 buffered samples per image. Both use the corrected output-center mask sampling and conservative augmentation boundary; only the physical FOV mask differs between the two arms.

This short local schedule is a diagnostic, not a budget-matched comparison with the full-scene 80k model. The latter is also evaluated on the exact same local images and valid pixels for context. Local results cannot be attributed solely to the mask when compared with the global model, because training data scope and budget differ.

```bash
python -m research.glace_fusion.local_mask_ablation \
  --run-root /root/rivermind-data/glace_nclt_stage2_local_mask_20260912 \
  --source-run /root/rivermind-data/glace_nclt_rgb_large_20260912 \
  --test-scene /root/rivermind-data/glace_nclt_rgb_eval_20260912/scene \
  --calibration /root/rivermind-data/datasets/NCLT/calibration/map_cache \
  --deit-checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth
```

The run directory cannot already exist. `state.json`, `unmasked/train.log`, `masked/train.log` and ultimately `results.json` identify execution progress and outcomes. No fusion architecture is changed.

The separate cross-frame LiDAR audit found median point-to-plane ICP corrections of 8.8cm / 0.74°. This remains a consistency concern, not proof of GT error: motion distortion, occlusion, sampling and ICP degeneracy were not eliminated. The GT trajectory and calibration are therefore unchanged in this experiment.

## Depth inflation diagnostic and sparse LiDAR arm

The mask ablation did not establish a substantial cross-date ranking benefit. The unmasked/masked local test q10 values were 14.00% / 13.72%; translation 0.5–1m ranking was 52.08% / 53.13%, and rotation 0.5–1° was 60.20% / 60.37%. Training fit improved substantially relative to the full-scene head, but the short local experiment changed scene scope and budget as well as the supervision distribution.

`audit_coordinate_depth.py` compares frozen camera predictions with exact-timestamp `velodyne_sync` scans at the corresponding camera pixels. It uses the official body-to-camera calibration, a nearest-pixel depth buffer, three LiDAR neighbors within 3px and a maximum 1.2 depth ratio to reject discontinuities. This is sparse depth consistency evidence; it cannot eliminate every occlusion, scan-motion or calibration error. The large observed discrepancy persists even among 10px reprojection inliers and is consistent with fitting image rays at excessive depth.

A third local arm retains the masked arm's images, seed, head, features and 5k budget. It adds sparse training-only LiDAR supervision:

- Only the selected training images can create `scene/train/lidar_world/*.npy`; exact scan timestamps are required. Missing scans produce no auxiliary labels.
- World points are projected through each augmented GT camera pose and K during buffer creation, so augmentation and pixel centers remain aligned.
- The median supported LiDAR depth defines a target on the sampled camera ray; unstable depth neighborhoods are excluded.
- A Smooth L1 camera-frame 3D residual (beta 1m, weight 1) is added to the existing reprojection objective, summed over supported points and divided by the full batch size.
- Evaluation remains RGB-only. Test LiDAR is used exclusively for the separate depth audit, never for optimization or inference.

This is **GLACE with sparse LiDAR training supervision**, not a pure RGB-and-pose reproduction of official GLACE. The experiment is designed to test whether constraining depth recovers translation-sensitive evidence before considering any full-scene training.

```bash
python -m research.glace_fusion.train_lidar_auxiliary \
  --experiment /root/rivermind-data/glace_nclt_stage2_local_mask_20260912 \
  --dataset-root /root/rivermind-data/datasets/NCLT \
  --deit-checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth

python -m research.glace_fusion.summarize_mask_ablation \
  --run-root /root/rivermind-data/glace_nclt_stage2_local_mask_20260912

python -m research.glace_fusion.audit_coordinate_depth \
  --run-root /root/rivermind-data/glace_nclt_stage2_local_mask_20260912 \
  --dataset-root /root/rivermind-data/datasets/NCLT
```

The summarizer checks that frame sets, GT, K and sampled pixels match exactly across all arms and reports paired time-block bootstrap intervals for the mask and LiDAR effects. A single local region and seed cannot establish whole-NCLT effectiveness.

## Completed local results

All three local heads completed 5,000 iterations. Evaluation uses exactly the same 64 cross-date local test images, RGB cached features, FP32 head, GT/K and valid-FOV pixel coordinates. These images differ from the earlier global test64 probe; the old global percentages must not be substituted into this table.

| Local test metric | Unmasked local | Masked local | Masked + sparse LiDAR |
|---|---:|---:|---:|
| GT q10 | 14.00% | 13.72% | 16.11% |
| Translation 0.2–0.5m ranking | 50.82% | 51.78% | 76.78% |
| Translation 0.5–1m ranking | 52.08% | 53.13% | 77.13% |
| Rotation 0.5–1° ranking | 60.20% | 60.37% | 63.93% |

These local arms have no ties in the displayed intervals, so strict and half-credit ranking accuracy coincide. For sparse LiDAR versus masked local, paired time-block bootstrap 95% intervals for the accuracy changes are +19.44 to +30.56 percentage points, +21.31 to +26.30 points, and +0.52 to +6.47 points respectively. These intervals describe this one region, not seed or whole-dataset uncertainty.

On the same sparse LiDAR support, the test per-frame depth-median aggregate is 7.25m for LiDAR, 114.71m for stage1 80k, 174.51m for masked local, and 10.16m for masked + LiDAR. The median per-frame absolute relative depth error falls from 17.95 to 0.368 between the two masked arms. Even among 10px reprojection inliers, the corresponding relative depth error falls from 12.39 to 0.122. This supports depth inflation as a significant translation-ranking failure mechanism in this region; it does not establish it as the only full-scene failure cause.

Training GT q10 rises from 46.82% to 55.66%; test q10 remains only 16.11%. The model is still a local candidate-evidence experiment. No full-scene auxiliary training, official DSAC* evaluation, real LEADER candidate evaluation or fusion change has been performed in stage2.

Validation includes 54 regression tests, 18 real-loader mask checks, completed real-image training/evaluation for all arms, identical evaluation geometry/pixel assertions, and sparse-depth audits on 62 training / 64 test frames. Two evaluated training images lacked exact-timestamp scans and were excluded only from the depth audit.
