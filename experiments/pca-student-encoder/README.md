# Image-only student of the SAM feature frontend

`RGB -> 64x64x64 code -> 4-bit storage -> fixed PCA expansion -> frozen V7`

The point of this experiment is not to shrink SAM's output but to remove SAM from the inference
path entirely: at scoring time only the image goes in, and the frozen V7 appearance field
receives a 256-dimensional map reconstructed from a 64-dimensional code.

## Method

The student predicts **coordinates in the frozen PCA-64 subspace of the SAM token map**, not
arbitrary 256-d features.  Two consequences matter:

- reconstruction to 256 dimensions is the fixed PCA expansion, so there is no learned adapter
  and the code lives in exactly the space that gets quantised;
- the storage format and the regression target are the same object.

Network: MobileNetV3-Small backbone, stride-16 stage with a light FPN over stride 16 and 32
(width 128), a depthwise 3x3 + 1x1 head to 64 channels.  Input is preprocessed the way SAM was -
longest side 1024, SAM mean/std, zero pad to 1024 square, patch 16 -> 64x64 grid - so the token
grid is aligned with the teacher.

Three supervision terms, all against frozen teachers:

| term | target | weight |
|---|---|---|
| code | the PCA-64 coordinates of the cached SAM tokens, per-channel standardised (L1 + cosine) | 1.0 |
| descriptor | the 32-d unit descriptor V7 actually reads, sampled at the projected LiDAR points (cosine) | 2.0 |
| energy / margin / rotation | V7 appearance energy over sampled candidate poses (found neutral, see ablations) | 0 by default |

Training data: four NCLT training dates disjoint from the four evaluation dates, 4000 frames per
date, 3000 steps, AdamW 1e-3, about 20 minutes on one RTX 4090.

## Result on the 2502-frame NCLT pool (V7 frontend only)

| variant | storage/frame | MPE (m) | MOE (deg) | success (<=1 m & <=2 deg) |
|---|---:|---:|---:|---:|
| SAM ViT-L FP16 (baseline) | 2 MiB | 1.3204 | 1.9875 | 1482 |
| SAM ViT-L per-channel INT8 | 1 MiB | 1.3206 | 1.9938 | 1483 |
| PCA-64 INT8 (closed form) | 256 KiB | 1.3230 | 2.0467 | 1464 |
| PCA-64 4-bit (closed form) | 128 KiB | 1.3207 | 2.0468 | 1461 |
| **this student, 8-bit code** | 256 KiB | 1.3285 | 2.0255 | 1463 |
| **this student, 4-bit code** | 128 KiB | 1.3301 | 2.0265 | 1458 |

Against the SAM baseline the 4-bit configuration loses 70 frames and gains 46 (net -24), with no
new frame over 10 m.  Against the 8-bit version of the same encoder, 4-bit costs 5 frames.

## Ablations (all at 4-bit storage, same schedule)

| variant | success | what it shows |
|---|---:|---|
| full | 1458 | - |
| without the descriptor term | 1383 | the descriptor target is the single largest contributor, -75 frames |
| learned 256-d adapter instead of the fixed PCA expansion | 1457 | the subspace choice is neutral for accuracy, it simplifies the pipeline |
| plus boundary/along-track decoy margins | 1457 | the energy already separates poses just past the 1 m / 2 deg line |
| plus a rotation-direction term (1-4 deg about each axis) | 1457 | rotation is not limited by this feature; see diagnostics |

## Diagnostics that bound the approach

These were run to decide whether the feature is worth pushing further; each is a standalone
script in `diagnostics/`.

- **Sparsity of the consumer**: V7 samples the token map only at the pixels where the frame's
  LiDAR points project, and that pixel set is fixed for the whole refinement.  On 160 hard frames
  the 2x2 bilinear neighbourhood of those pixels covers **11.4%** of the valid 64x64 grid
  (median 10.7%, p90 14.9%).  A query-aligned store would therefore keep ~11% of the tokens.
  The end-to-end sparse evaluation did not complete in this environment, so the storage figure is
  a coverage measurement, not a verified accuracy result.
- **Rotation**: from the LEADER initial pose to V7's refined pose, position error drops 16.1% but
  rotation only 9.6% on the hard pool, and the rotation update the refiner applies has a median
  cosine of **-0.33** against the correction that was needed (-1 would be perfect), with 89 of 200
  frames ending up worse.  Training the feature specifically for rotation changed this statistic
  from -0.330 to -0.365: no effect.  Rotation is an identifiability/optimiser problem here, not a
  feature problem.
- **Compression sensitivity**: replacing SAM with a 64-d 4-bit code shifts the rotation error by
  0.05-0.07 deg on hard frames, i.e. the same order as the 75-frame descriptor ablation is not.
  The features carry far more information than the metric consumes.
- **Position-only decoding**: a quadratic in image position explains only 34.9% of the code's
  variance, and the code is full rank (64).  Generating descriptors from a global code plus query
  position is therefore not viable; the map is image content, not a smooth function of position.
- **Retrieval**: a pooled descriptor from this encoder reaches recall@5 (25 m radius) of 0.515
  against 0.799 for a frozen DINOv2 CLS baseline on the same 1502/4752 benchmark.  A compact image
  encoder is not a substitute for a retrieval backbone.

## Reproduction

```
# train (four training dates, disjoint from the four evaluation dates)
python student_pca.py --out runs/student --steps 3000 --gpu 0 --per-date 4000 --lr 1e-3 \
    --w-code 1.0 --w-desc 2.0

# evaluate on the 2502-frame pool, three shards
python eval_pca_student.py predict --output runs/eval --gpu 0 --shard 0 --shards 3 \
    --encoder runs/student/encoder.pt --width 128
python eval_pca_student.py score --output runs/eval
```

Dependencies: the frozen V7 stack (`e2e_online.boot_online`), the cached SAM features and the
`e2e-moe` project, the PCA basis fitted on the training dates, and the protocol truth used only
by the scoring process.

## Limits

- V7 frontend only; the MoE routing replay and the visual arbitration were not re-run for this
  encoder.
- Single training run, single seed, no hyper-parameter search.
- NCLT only; Oxford is a different camera and single-channel imagery and was not tested.
- The storage claim (128 KiB) matches 4-bit PCA-64; the gain of this variant is architectural -
  SAM is not needed at inference - not a smaller footprint.
