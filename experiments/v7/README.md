# V7: LWV-trained FAR with the original V6 inference solver

V7 is the user-selected **LWV-trained new weights + original V6 self-verifying solver**. The LEADER/SAM frontend remains frozen in this release. LWV is used during training; online inference uses the original cubic candidate volume. Training and inference intentionally use different solvers. This release preserves that distinction and does not substitute the later repaired LWV controller.

`runtime/adaptive_solver.py` and `runtime/selfverify.py` are byte-identical to the registered scientific sources. The inference solver SHA256 is `7a0712eaf3a979623cf751d58baa318316c46337f0f9899c89ad895ab5d13f07`; the training solver SHA256 is `f65cc6c54de4497581be4b2fe872dfd26c7d38614c6dc11cedf19adbddc22b7b`. Model/frontend sources and the hash encoder are inherited from `../v5/source` and `../v5/runtime`. `source_provenance.json` distinguishes exact copies, source-path adapters, and the extracted parallel inference class.

## Algorithm and execution

- Inference: two geometric warm updates followed by at most 12 self-verifying proposals. Spacing starts at 0.4 m, remains within 0.025–0.8 m, and follows the original acceptance, rejection and stopping rules. The solver uses a 125-candidate cubic volume and rebuilds correspondence evidence at the current pose.
- Training: original adaptive LWV query shape, matching CE, final scan Huber with delta 0.2 and coefficient 1, plus shape loss with coefficient 0.1. The registered V7 recipe is not the separate Huber-weight sweep.
- Training acceleration: seven independent CUDA-stream replicas, complete mixed-coordinate/parameter backward CUDA kernels, analytic pose derivatives, template-moment inverse reuse, scheduled per-frame forward/backward and gradient reduction. Effective batch is 25; the final batch has 14 frames. This is one trained network with temporary gradient replicas.
- Inference acceleration: four independent replicas execute the **unchanged original solver** with one-thread input prefetch. A discarded serial refinement initializes CUDA/linalg before parallel work. Every evaluated checkpoint receives a fixed four-date 32-frame exact comparison; failure falls back to serial execution. Rejected cooperative field batching is not an active inference mode.

The registered small-set checkpoint comes from 10 epochs / 480 Adam updates on 1,200 training frames. Its SHA256 is `d541c41f02968ea1e9ccf8cce59a4d3e07e2e9e626ce0bd23c10166f8798fc7d`, body SHA256 `0ecb4dd67eb8eb97e3468f2012c618a9d0ffddc200f1a2bb48c7fa0739858071`. The full continuation uses a fresh Adam optimizer with LR 0.001 and StepLR gamma 0.9, five full epochs, and evaluations after epochs 3 and 5. There are 107,339 Train / 104,732 Test frames and 4,294 updates per epoch. Epoch 3→5 retains optimizer, scheduler, RNG and frame order; Test is not used to choose weights or tune settings.

## Verified results and limits

The registered **reused 1,200-frame development** result is MPE **0.1189443441 m**, MOE **1.4115913623°**, joint success (<1 m and <2°) **960/1200**. It is not a full-data or independent blind-test result. The full-data 3+2 experiment was still ongoing at publication; this release does not certify final full-Test MPE/MOE.

On the same frozen full-epoch-1 weights and fixed 1,200 four-date frames:

| Cached inference measurement | Serial | Parallel4 |
|---|---:|---:|
| Observation + warm + refinement compute | 121.98793 s | 65.72942 s |
| Input loading + compute sum | 162.32712 s | 97.32773 s |
| Maximum pose difference | — | 0 |
| Maximum continuous trace difference | — | 0 |
| Discrete decisions | Reference | Identical |

Compute throughput improved 1.85591×. The load+compute sum improved 1.66784×, with sequential cache reuse affecting load timing. These measurements exclude frontend cache creation, process/model startup, parity guards, saving and control overhead; they are not raw-data end-to-end measurements or single-frame latency. Two independent cold processes additionally passed exact serial comparisons on 256 continuation frames each. The first production attempt without serial initialization failed; the initialization fix and failed/rejected variants are documented rather than treated as accuracy results.

The inherited environment was Linux, Python 3.8, PyTorch 1.12 / CUDA 11.6, MinkowskiEngine, NumPy, SciPy, Pillow and the existing LEADER dependencies. The science and execution modules were validated on the original deployment. The public prepared-frame/path adapters have syntax, dependency, source-provenance and AST checks; they have **not been rerun on a fresh GPU installation**. Continuous multi-epoch bitwise training equivalence is not claimed.

## Prepared-frame interface

Data, model checkpoints, SAM embeddings and compiled platform binaries are external assets. The public adapter accepts serialized CPU frame dictionaries produced from the existing frozen frontend/geometry loaders. It does not regenerate LEADER, SC2-PCR or SAM outputs. Preserve the canonical original frame order, sample selection, center translation, image scaling and tensor dtypes. Use the existing V5 loaders / deployment preparation to construct these dictionaries; the seven-tensor geometry-only cache by itself is insufficient.

Each frame is a trusted `.pt` dictionary with `points`, `pixels`, `visible_ids`, `structure`, `initial`, `source`, `target`, `rgb`, `sam`, `hw`, and `key`. Train additionally contains `truth` and `raw`, using the original centered training labels. Test must have `input_sha256` and must omit `truth`, `raw`, and `perturbation`. Training perturbations are generated with the unchanged seed-20 sequence inside the training loop.

A separate JSON manifest for each split contains:

```json
{
  "split": "train",
  "frozen_LEADER_sha256": "49cf3085b507e2f67a0f250394cadd980b0f5da18eb00dcfe37a6c766d3f4d7a",
  "center_t": [0.0, 0.0, 0.0],
  "records": [
    {"key": "DATE/UTIME", "file": "frames/frame.pt", "sha256": "FRAME_FILE_SHA256"}
  ]
}
```

Replace the illustrated center with the exact frozen LEADER checkpoint center. Train has 107,339 records; Test uses `"split": "test"` and 104,732 records. The date split is fixed in `protocol.json`. Each frame file is checked against its manifest SHA before loading. The manifests describe serialized inputs for this public adapter, rather than claiming the same byte format as the original deployment cache.

## Run

Use an existing compatible environment and select one suitable GPU explicitly with `CUDA_VISIBLE_DEVICES`. The original deployment chose idle GPUs 0–3 first and 4–7 only when all 0–3 were unavailable. Runtime protection retains a 23,500 MiB physical GPU cap, 12 GiB RSS cap, 32 GiB minimum system available RAM and 8 GiB output-disk reserve. No other process is terminated by these entrypoints.

```bash
python experiments/v5/compile_cuda.py --arch 86
python experiments/v7/compile_cuda.py --arch 86
python experiments/v7/configure.py \
  --train-manifest /path/to/prepared/train.json \
  --test-manifest /path/to/prepared/test.json \
  --initial-checkpoint /path/to/registered-v7-small.pt \
  --evaluator-gt /path/to/evaluator-only \
  --output /path/to/v7-output

CUDA_VISIBLE_DEVICES=0 python experiments/v7/train_v7.py --epoch 1
CUDA_VISIBLE_DEVICES=0 python experiments/v7/train_v7.py --epoch 2
CUDA_VISIBLE_DEVICES=0 python experiments/v7/train_v7.py --epoch 3
CUDA_VISIBLE_DEVICES=0 python experiments/v7/predict_v7.py --checkpoint /path/to/v7-output/epoch3.pt
# Score the printed evaluation directory in a separate CPU process.
CUDA_VISIBLE_DEVICES=-1 python experiments/v7/score_v7.py \
  --predictions /path/to/v7-output/evaluation-CHECKPOINT_HASH_PREFIX \
  --gt /path/to/evaluator-only/poses.npz
CUDA_VISIBLE_DEVICES=0 python experiments/v7/train_v7.py --epoch 4
CUDA_VISIBLE_DEVICES=0 python experiments/v7/train_v7.py --epoch 5
CUDA_VISIBLE_DEVICES=0 python experiments/v7/predict_v7.py --checkpoint /path/to/v7-output/epoch5.pt
```

Re-running the same interrupted `--epoch` resumes atomic slots with full Adam/scheduler/RNG/order state; source and plan hashes must match. Generated configurations, checkpoints and output are ignored by Git. Private deployment checkpoints use the private plan SHA and are not automatically resumed as public-adapter runs. `predict_v7.py --serial` selects the reference execution explicitly. The scorer's GT NPZ contains `frame_keys` and world-frame `poses`; prediction chunks must be completely frozen before scoring.

## Evaluation policy

All new scores use the exact `evaluation/nclt_reference_policy.py`, policy JSON and SHA-verified replacement reference. The 255 flagged frames in the specified 2012-02-12 UTIME interval use fixed-SE3-aligned official 100 Hz odometry; all remaining frames retain original reference. Missing replacement reference fails rather than falling back to GT. Reports save protocol, policy SHA and replacement counts. This convention is **not the unmodified official GT benchmark**. Training labels are unaffected. V5 comparisons must be rescored under the same convention.

The repository upload does not stop, modify or restart the original running full-data experiment or re-enable the disabled chat monitor.
