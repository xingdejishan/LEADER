# LEADER query-level feature fusion

Branch: `research/leader-query-gaussian-fusion`, based on `research/dinov2-leader-feature-fusion` at `bcb5ede`.

Each existing LEADER voxel supplies its 512D geometric feature as a query. A 5x5 image neighborhood, spaced one DINOv2 patch (14 pixels) apart around its audited raw-point projection, supplies keys and values. Attention operates in 64D; a zero-initialized gated residual updates the original geometric feature. The existing MMRegressor then predicts world coordinates and confidence for the same local voxels. There are no newly generated 3D points, no independent visual coordinate head, and no use of GT/world pose to form attention queries or image projections.

The Gaussian arm differs only by adding the fixed image-plane log prior `-0.5 * (dx^2 + dy^2)` to attention logits, with offsets in patch units (sigma 14 pixels). This is a soft association prior, not a 3D Gaussian map, Gaussian splatting, or calibrated positional covariance. The non-Gaussian arm sees exactly the same neighborhood and learned relative-position encoding. The general soft-association idea is motivated by [TransFusion](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers.html); this experiment is a localization adaptation, not a reproduction of its object detector.

## Frozen experiment

- Reuse the previously audited 64 training / 32 development frames on 2012-02-18. This is not the 907/303/148 experiment. The date participated in LEADER pretraining and has already been used for development.
- Freeze official DINOv2 ViT-B/14, training-only PCA768->128, LEADER encoder and all decoder layers except `pred_out`.
- Compare original frozen LEADER, matched-budget `pred_out` fine-tuning, query fusion plus `pred_out`, and Gaussian-query fusion plus `pred_out`.
- Each trained arm gets the same 600 frame/point steps, Adam 1e-4, seed 2089 and up to 1,024 voxels per step. Visual arms have 10% whole-frame modality dropout. Original coordinates, targets, point counts and valid center projections remain unchanged.
- Neighbor tokens obey raster bounds and undistortion masks. The center uses the audited raw representative and raw-scan visibility. Neighbor pixels have no assumed depth; soft association can still include another object, which is a limitation of this first version.
- All arms use original SC2 confidence-top50% initialization, followed by the unmodified `full_pool_refine` from `glace-local/code/tools/full_pool_robust_v1.py`, thresholds 1.2m then 0.6m. Its source hash is frozen, and its output is checked against an existing v1 two-stage cache before evaluation. GT is not used by either solver.
- Missing-image diagnostic arms use their own fine-tuned decoder. Exact feature fallback does not imply a return to the original pretrained pose once the decoder has been adapted.
- Pass criterion is frozen before training: a visual arm must improve both MPE and MOE against both original two-stage and same-budget LiDAR fine-tuning, without losing 1m/2deg successes. No validation-based hyperparameter sweep or checkpoint selection.

## Run

```bash
bash research/query_fusion/launch.sh all
```

Stages: `check`, `prepare`, `extract`, `train`, `evaluate`, `report`. The default output is `/home/zhang/leader-query-fusion`. Input paths and hashes are recorded in `results/protocol.json`; local DINOv2 weights/PCA come from `/home/zhang/dinov2-leader`, and geometric features/projection maps from `/home/zhang/leader-image-gate`.

`extract` uses the existing `rscore-l` environment; training and the original two-stage solver use `egonn118`. The tests cover Gaussian attention ordering, completely missing visual input, zero-initialized identity with trainable gradients, and invalid-neighbor masking. Results and per-frame initial/final errors are in `results/`.
