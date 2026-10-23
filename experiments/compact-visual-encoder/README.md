# Compact visual encoder: RGB -> 64x64x64, distilled against frozen V7

Replaces the cached SAM ViT-L token map (64x64x256, 2 MiB/frame in FP16) with a 64-channel
code (64x64x64, 256 KiB/frame in INT8) produced directly from the RGB image by a
MobileNetV3-Small + light FPN encoder and a 1x1 adapter back to 256 channels, so the frozen
V7 appearance field and the whole MoE chain stay untouched.

Training uses two signals against frozen teachers: relational distillation of SAM token
similarity structure, and pose-response distillation of V7's appearance energy over sampled
candidate poses. End-to-end backprop through V7 is not possible: the refiner is a closed-form
constrained solve and returns poses without gradients.

Result on the 2502-frame NCLT pool (see `docs/REPORT.md` for the full tables): the encoder
reaches 1410 successful frames against 1482 for SAM FP16 and 1464 for a PCA-64 INT8 basis.
Position error matches the baseline, rotation error does not. The INT8 storage step is
essentially free (0.0002 m / 0.0039 deg against the same encoder in FP16).

## Layout

| path | content |
|---|---|
| `src/encoder.py` | student network, SAM-aligned preprocessing, INT8 code quantisation |
| `src/train_compact.py` | distillation training loop on the four training dates |
| `src/eval_compact.py` | paired 2502-frame evaluation, three variants, shardable |
| `src/make_table.py` | builds the comparison tables from the scoring reports |
| `results/run1_report.json` | evaluation of the 600-step checkpoint |
| `results/run2_report.json` | evaluation of the 4000-step checkpoint |
| `docs/REPORT.md` | method, protocol, results, limitations, next steps |

## Protocol notes

- Evaluation pool: 1502 hard frames plus 1000 ordinary frames on the four test dates.
- Training dates are disjoint from the four test dates; scoring truth is read only by the
  scoring process, never by inference.
- All variants share the same scan, initial pose and image; only the 256-dimensional features
  fed to V7 change.
- Evaluation covers the V7 frontend only, not the MoE routing replay.
