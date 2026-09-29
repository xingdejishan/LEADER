# LEADER MaGiC runner order

The four development dates are used once per epoch for checkpoint selection. Every result from them is development/validation output, not an independent test score. The predictor reads public manifests and scans only; only the evaluator reads `evaluator_only/test_gt`.

Run the steps from the frozen branch and environment after the real-frame GPU smoke, batch-size freeze, and resource review. The train and development SAM cache manifests must already exist and pass the adapter's frozen hash checks.

## 1. Freeze the same-code L0 reference

Use the same project revision, LEADER checkpoint, development manifests, and development SAM cache that the MaGiC run will use. Do not edit the predictor, evaluator, backend, data adapter, or runtime code between this step and training.

```text
python -m tools.predict_magic_nclt \
  --data-root <NCLT-root> \
  --manifest-root <public-dev-manifest-root> \
  --sam-cache-manifest <dev-SAM-cache-manifest.json> \
  --sam-cache-sha256 <frozen-dev-cache-sha256> \
  --checkpoint <official_l0.pt> \
  --predictions <new-run>/l0/predictions.jsonl \
  --freeze <new-run>/l0/prediction_freeze.json \
  --seed 20 --device cuda

python -m tools.evaluate_magic_nclt_dev \
  --predictions <new-run>/l0/predictions.jsonl \
  --freeze <new-run>/l0/prediction_freeze.json \
  --checkpoint <official_l0.pt> \
  --predictor <project>/tools/predict_magic_nclt.py \
  --manifest-root <public-dev-manifest-root> \
  --dev-gt-root <NCLT-root>/evaluator_only/test_gt \
  --role l0_reference_development_validation \
  --report <new-run>/l0/evaluation.json
```

The baseline report must bind all four development manifest hashes, the same SAM cache manifest, the official L0 checkpoint, and the current runtime source hashes. The trainer rejects a mismatch.

## 2. Train and select

Use a new, private run directory. Freeze the physical batch size from the real-frame smoke; accumulation is one and `drop_last` is false. Each of 40 complete epochs trains on all 1,200 training frames, then runs the GT-free predictor and separate evaluator over all 1,200 development frames. The selection order is failure count first, then `MPE/MPE_L0 + MOE/MOE_L0`, with ties kept at the earlier epoch.

```text
python -m tools.train_leader_magic \
  --data-root <NCLT-root> \
  --train-manifest-root <public-train-manifest-root> \
  --dev-manifest-root <public-dev-manifest-root> \
  --train-sam-cache-manifest <train-SAM-cache-manifest.json> \
  --dev-sam-cache-manifest <dev-SAM-cache-manifest.json> \
  --dev-sam-cache-sha256 <frozen-dev-cache-sha256> \
  --dev-gt-root <NCLT-root>/evaluator_only/test_gt \
  --base-checkpoint <official_l0.pt> \
  --l0-report <new-run>/l0/evaluation.json \
  --out <new-run>/training \
  --batch-size <smoke-frozen-physical-batch> \
  --seed 20 --device cuda
```

To continue an interrupted run, pass the exact same arguments and append `--resume`. Resume requires byte-for-byte identical frozen settings, revalidates the L0 report hash, and recovers `best.pt` against the committed `latest.pt`. If epoch 40 is committed but selection metadata was not written, `--resume` only verifies the 40 history rows and writes the same selection record; it does not train or reevaluate.

Only `latest.pt` and `best.pt` are retained as full checkpoints. Each candidate is hashed before validation; if selected, it is atomically promoted to `best.pt` and the hashes must match. The old best is held as `best.previous.pt` until the new `latest.pt` commits. The temporary candidate is removed before that commit; history and selection metadata preserve its SHA, model-state SHA, validation outputs, and whether its bytes are retained as the best checkpoint.

## Public snapshot asset requirements

All filesystem paths in the commands are placeholders. Supply your existing NCLT manifests, scans, Cam5 images, isolated development GT, official_l0.pt and frozen SAM cache manifests. The manifest adapter intentionally verifies the original frozen frame/SAM hashes: arbitrary resampled frames or regenerated metadata are not interchangeable with this protocol. The snapshot includes preparation helpers but not the original data and weights.

Keep the published source unchanged while generating the L0 reference and training; runtime source hashes bind the predictor/evaluator and checkpoint evidence. The new convenience entry point delegates to the unchanged runner. Existing private-run checkpoints are not a claim that a new public-source run has already happened.
