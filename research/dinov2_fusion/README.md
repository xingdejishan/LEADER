# Frozen DINOv2 features fused into LEADER

Branch: `research/dinov2-leader-feature-fusion`, based on `research/rscore-l-local` at `f84b5f1`.

This is feature-level fusion before LEADER's existing MMRegressor. LEADER predicts world coordinates for local LiDAR voxels; it is not a scan-to-map descriptor matcher. DINOv2 patch features are sampled at projected real LiDAR representatives, compressed by training-only PCA, and injected into the frozen 512D geometric features through an 83,393-parameter residual gate. The original coordinate/confidence head and SC2 pose solver consume the result. No R-SCoRe coordinate predictions, Node2Vec or separate visual PnP enter this experiment.

DINOv2 ViT-B/14 uses the [official pretrained backbone](https://github.com/facebookresearch/dinov2). The backbone stays frozen. Its downloaded checkpoint is 346,378,731 bytes and has SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. The source archive and input hashes are recorded in `results/protocol.json`. This borrows the visual-to-LiDAR feature projection idea from [VFM-Registration](https://github.com/vniclas/VFM-Registration); it is not a reproduction of that paper, its surround-view configuration, or its descriptor-matching backend.

## Local diagnostic

The existing audited local data contains 64 training and 32 held-out development frames from 2012-02-18, with a 20-frame interval between their sampling pools. This date is part of LEADER pretraining and these development frames were previously used. This is not the original 907/303/148 split, a blind evaluation, or a full NCLT result. The original 907-frame bundle does not contain the raw training scans needed to reproduce this front end.

Inputs are the existing `/home/zhang/leader-image-gate` manifest, frozen encoder outputs and audited projection mapping. A real scan point within each output voxel supplies the projection location; localization continues to use the unchanged voxel center. Visibility uses the existing raw-scan zbuffer, 4-pixel cells, 0.5m depth tolerance and undistortion mask. GT is used only for training targets and error reporting. The projection mapping and feature rows are checked for exact correspondence.

Images are resized to 630x476 and normalized with ImageNet mean/std. The normalized 768D patch tokens form a 45x34 grid. Pixel-center-aware bilinear sampling preserves the image-resize coordinate transform. PCA fits 9,099 visible descriptors from the 64 training frames, reduces to 128D and retains 91.68% variance. No validation features fit PCA.

Both aligned and shuffled controls use seed 2089, the same gate initialization and frame/point schedule, 600 Adam steps at 1e-4 and up to 1,024 voxels per step. DINOv2, LEADER encoder and decoder remain frozen. Shuffling cyclically permutes visual descriptors among visible voxels while preserving the validity mask. Only the fusion gate is trained.

Evaluation uses the original upstream SC2 Matcher, confidence top-50% and fixed seed for every arm. **It does not use the v1 two-stage backend underlying the previous 148-frame comparison.** Results therefore must not be numerically compared with that earlier baseline. Missing images exactly recover the frozen LEADER predictions and poses.

## Running

Large weights and runtime artifacts are outside Git in `/home/zhang/dinov2-leader`. Extract the official source archive into `dinov2-main` there and place the backbone in `weights/dinov2_vitb14_pretrain.pth`. `prepare` freezes source, input and checkpoint hashes before training; changing the protocol requires a fresh run directory. The default local paths are explicit in `run.py`.

```bash
bash research/dinov2_fusion/launch.sh check
bash research/dinov2_fusion/launch.sh all
```

Individual stages are `prepare`, `extract`, `train`, and `evaluate`. Feature extraction uses the existing `rscore-l` environment; training/evaluation use the existing `egonn118` environment to reproduce the cached LEADER decoder and SC2 numerics. The three focused checks cover patch-center sampling/resizing, exact missing-image fallback even after gate changes, and gradient flow through the frozen decoder.

Results are in [results/report.md](results/report.md). The local configuration did not improve average pose accuracy, so it should remain an experimental branch rather than replacing the existing baseline.
