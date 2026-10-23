# PCA-64 SAM code: bit-width sweep on the 2502-frame NCLT pool

Question: after 256-d FP16 -> 256-d INT8 -> 64-d INT8, how many bits per coefficient can the
frozen V7 tolerate before accuracy actually moves?

Method: the cached SAM token map is projected onto the PCA-64 basis fitted on the four
training dates, quantised with a per-channel symmetric scale at B bits, dequantised and
projected back, then fed to the frozen V7.  Five variants (8, 6, 5, 4, 3 bits) share the same
scan, initial pose and image on every frame.  Deltas are paired against the 8-bit code; the
8-bit row reproduces the earlier PCA-64 result exactly (1464 successful frames), which is the
cross-check that this sweep uses the same basis and pipeline.

| bits | bytes/frame | MPE (m) | MOE (deg) | success <=1m & <=2deg | >10m | net success vs 8-bit |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 262,144 | 1.3230 | 2.0466 | 1464 | 8 | - |
| 6 | 196,608 | 1.3226 | 2.0426 | 1466 | 8 | +2 |
| 5 | 163,840 | 1.3244 | 2.0490 | 1461 | 8 | -3 |
| 4 | 131,072 | 1.3207 | 2.0468 | 1461 | 8 | -3 |
| 3 | 98,304 | 1.3251 | 2.0715 | 1436 | 8 | -28 |

Paired against the 8-bit code:

| bits | dMPE (m) | dMOE (deg) | success lost | success gained | new >10m |
|---:|---:|---:|---:|---:|---:|
| 6 | -0.00037 | -0.0041 | 13 | 15 | 0 |
| 5 | +0.00137 | +0.0024 | 14 | 11 | 0 |
| 4 | -0.00233 | +0.0002 | 27 | 24 | 0 |
| 3 | +0.00210 | +0.0249 | 58 | 30 | 0 |

Reading: down to 4 bits (128 KiB/frame, one sixteenth of the SAM FP16 cache) the aggregate
error and success count do not move; the per-frame churn at 4 bits is 27 lost and 24 gained,
so it is "no net loss", not "identical output".  At 3 bits the loss becomes systematic
(58 lost against 30 gained, rotation +0.025 deg).  No variant creates a new >10 m frame.

Scope: V7 frontend only, no MoE routing replay, no visual candidate arbitration.  Storage
figures exclude the per-channel scales (64 floats per frame, negligible).
