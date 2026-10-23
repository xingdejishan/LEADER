# PCA-space student encoder: RGB -> 64x64x64 four-bit code, 128 KiB per frame

Goal: a feature extractor of our own that reaches the 4-bit PCA-64 operating point
(128 KiB/frame, 1461 successful frames on the 2502-frame pool) without running SAM at
inference time.

## Design

The student predicts **coordinates in the frozen PCA-64 subspace** rather than arbitrary
256-d features. Consequences:

- reconstruction to 256 dimensions is the fixed PCA expansion, so there is no learned adapter
  and no trainable projection — only the image-to-code map is learned;
- the code lives in exactly the space that the storage format quantises, so 4-bit behaviour
  carries over from the PCA experiments.

Network: MobileNetV3-Small backbone, 16x-stride stage plus a light FPN (stride-16 and
stride-32 lateral fusion, width 128), 3x3 depthwise + 1x1 head to 64 channels. Input is
preprocessed exactly as SAM was (resize longest side to 1024, SAM mean/std, zero pad to
1024 square, patch 16 -> 64x64 grid) so the token grid is aligned.

Three supervision signals, all against frozen teachers:

| signal | target |
|---|---|
| code | the PCA-64 coordinates of the cached SAM tokens, per-channel standardised (L1 + cosine) |
| descriptor | what V7 actually consumes: the 32-d normalised descriptor from the frozen observation module sampled at the projected points (cosine) |
| energy | V7's appearance energy over 11 candidate poses per frame (value + ranking) |

The descriptor term is the main change against the previous attempt, which tried to match the
full 256-d token space: the descriptor is 32 numbers at the ~2048 points the energy actually
reads, so it is both smaller and better aligned.

Training: four training dates disjoint from the four evaluation dates, 4000 frames per date,
3000 steps, AdamW lr 1e-3, ~17 minutes on one 4090.

## Result (2502-frame NCLT pool, V7 frontend)

| variant | storage/frame | MPE (m) | MOE (deg) | successful (<=1 m & <=2 deg) |
|---|---:|---:|---:|---:|
| SAM ViT-L FP16 (baseline) | 2 MiB | 1.3204 | 1.9875 | 1482 |
| PCA-64 INT8 (closed form) | 256 KiB | 1.3230 | 2.0467 | 1464 |
| **PCA-64 4-bit (target)** | **128 KiB** | **1.3207** | **2.0468** | **1461** |
| previous encoder, 8-bit code | 256 KiB | 1.3253 | 2.1354 | 1410 |
| **this encoder, 8-bit code** | **256 KiB** | **1.3285** | **2.0255** | **1463** |
| **this encoder, 4-bit code** | **128 KiB** | **1.3301** | **2.0265** | **1458** |

Against the SAM FP16 baseline, the 4-bit configuration loses 70 frames and gains 46
(net -24), no new >10 m frame, maximum extra rotation 11.9 deg on a frame that was already
failing. Against the 8-bit version of the same encoder, 4-bit costs 5 frames, consistent with
the bit-width sweep on the closed-form basis.

**Reading:** predicting the PCA coordinates from the image alone reproduces 1458 of the 1461
successful frames that the oracle projection achieves, at the same 128 KiB/frame, with SAM
removed from the inference path. Rotation error is slightly better than the oracle
(2.0265 vs 2.0468 deg), position slightly worse (1.3301 vs 1.3207 m).

## Limits

- V7 frontend only; the MoE routing replay and visual arbitration were not re-run.
- Single training run, one seed, no hyper-parameter search; NCLT only, Oxford untested.
- The storage saving is identical to 4-bit PCA-64; the gain here is architectural (no SAM at
  inference), not a smaller footprint.
