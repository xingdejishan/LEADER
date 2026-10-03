# V5: accelerated Co-Volume FAR

V5 is the validated accelerated execution of V2, with the same learned model and localization procedure. It retains one LEADER/SC2-PCR initialization, two Tukey geometry warm-start updates (1.2 m and 0.6 m), 125 candidates, and three correlation/update rounds at each of the 0.4 m and 0.1 m scales. Each round rebuilds the volume at the updated pose. The output is the final pose; there is no second SC2-PCR backend.

The acceleration consists of a fused CUDA hash-grid encoder with first-order gradients, batched per-frame Gauss–Newton systems, a 4096-point query chunk, and static geometry caching with one-thread prefetch. Geometric scales, bounded updates, robust weights, CE and full-scan Huber supervision are retained. Geometry caches do not contain learned field scores or scores reused across different poses. `source/` contains the exact 97-file frozen model/LEADER runtime; `source_provenance.json` identifies the original hashes and the deployment-only adaptations.

## Training protocol and current status

Stage 1 trains the V5 backend from scratch while freezing LEADER and using cached SAM features. The fixed protocol is seed 20, 50 epochs, physical batch 25, Adam with learning rate 0.001, StepLR gamma 0.9 per epoch, CE plus full-scan Huber, and gradient clipping at 10. Every fifth epoch evaluates **LEADER+V5**, without separately evaluating LEADER. The full prepared set has 107,339 training frames and 104,732 development frames; three LiDAR frames have no exact Cam5 image. Raw point range is 1–100 m, with at most 2,048 selected visible points.

At publication, the actual scratch training had started; completion of all 50 epochs and its final accuracy are not certified. The requested later LEADER fine-tuning stage is pending and is not implemented by this entrypoint. Development data were used previously; they are not an independent untouched test set.

## Verified evidence

The accelerator was validated separately using an existing V2 epoch-40 checkpoint:

| Full development set, same checkpoint | Original execution | Accelerated execution |
| --- | ---: | ---: |
| Frames | 104,732 | 104,732 |
| Mean position error (m) | 0.2993054215 | 0.2993054223 |
| Mean rotation error (deg) | 1.6749538026 | 1.6749538328 |
| Joint success: position < 1 m and rotation < 2 deg | 74,352 | 74,352 |

A separate complete continuation epoch processed all 107,339 training frames in **5,570.998 s (1.5475 h)** with batch 8; startup took another 113.173 s. This was continuation from V2 e40, not the new scratch run. The scratch batch-25 prefix suggested approximately 1.45 h per epoch; that number is an extrapolation, not a completed-epoch measurement. Further accuracy gains after continuation training belong to the extra training budget, not to the accelerator. First-order numerical agreement and full localization accuracy passed; bitwise equality and higher-order autodiff are not claimed.

The appearance-off ablation in `results/appearance-ablation.json` uses the old 1,200-frame development subset and its SLERP scoring protocol. With the same e40 checkpoint, initialization and 3+3 procedure, turning off appearance changed MPE 0.13245→0.20103 m and MOE 1.34295→1.66100 deg. Learned geometry and uniform-volume constraints were retained; this is not an independently trained pure-geometry model and is not comparable directly with full-trajectory nearest-GT numbers.

## Run on prepared assets

The validated environment was Linux, Python 3.8, PyTorch 1.12 with CUDA 11.6, MinkowskiEngine, NumPy, SciPy, Pillow, OpenCV and an NVIDIA compute-8.6 GPU. Use an existing compatible LEADER environment. The CUDA kernel is supplied as source and must be built for the target GPU; platform binaries are not part of this branch. Portable entrypoints have syntax/provenance checks; the deployment adapters have not been rerun on a new GPU environment.

Required external assets are the frozen official LEADER checkpoint, the exact prepared full-trajectory catalog/manifests, Cam5 images, LiDAR scans, SAM caches, training labels and raw SC2 initial caches. Training truth and development evaluator-only truth are separate. Asset hashes in `protocol.json` are fixed to the validated run; this release does not automatically download or regenerate the full dataset/SAM/initial caches.

```bash
python experiments/v5/compile_cuda.py --arch 86
python experiments/v5/configure.py \
  --data /path/to/prepared-full-trajectory \
  --leader /path/to/official_l0.pt \
  --geometry-cache /path/to/geometry-cache-root \
  --output /path/to/v5-output \
  --raw-gt /path/to/raw-groundtruth/evaluator_only

# Skip this step if the verified static geometry cache already exists.
CUDA_VISIBLE_DEVICES=-1 python experiments/v5/build_geometry_cache.py
CUDA_VISIBLE_DEVICES=0 python experiments/v5/train_v5.py
# Resume the same local frozen plan and output:
CUDA_VISIBLE_DEVICES=0 python experiments/v5/train_v5.py --resume
```

Training saves atomic alternating resume slots with optimizer, scheduler, RNG and frame order, plus a separate checkpoint per completed epoch. Evaluation predicts with development GT access blocked, freezes all predictions, then invokes the CPU scorer. Standalone completed-epoch evaluation can be run as:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/v5/predict_v5.py --epoch 5
CUDA_VISIBLE_DEVICES=-1 python experiments/v5/score_v5.py --epoch 5
```

The inherited per-process memory, system RAM and disk checks remain in the portable entrypoints. Generated local paths/plans, caches, weights and outputs are ignored by Git. The currently running private deployment is unchanged by this publication.
