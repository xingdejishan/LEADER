# Query × backbone factorial experiment

Run `bash research/query_backbone_factorial/launch.sh all` from WSL. Runtime artifacts remain in `/home/zhang/leader-query-backbone-factorial`; reports are copied into `results/` after evaluation.

The four cells are DeDoDe B / DINOv2 ViT-B/14 crossed with projected-center / 5×5-neighborhood attention, using all six cameras. Each camera first pools its image tokens with a LiDAR query, then the same query/key layers pool camera features with a camera bearing embedding. The center condition selects token 12 from the exact same feature cache. Parameter counts and initialization match across all four cells; neighborhood computation is greater.

Both frozen backbones receive the same 448×616 RGB raster and normalization. Neighborhood locations have 14-pixel spacing in this common raster. Backbone receptive fields are not identical. Each backbone gets its own 128D PCA fitted on the same visible center indices from the 64 training frames and all six cameras. Development images never enter PCA fitting. The comparison therefore measures the pretrained backbone plus its fitted PCA, rather than isolating semantic knowledge.

All five trained arms, including the pure LiDAR control, adapt the same pretrained decoder's final head for 600 Adam steps at 1e-4. Three paired seeds use the same frame order, voxel subsampling, and 10% whole-frame image dropout. Final checkpoints are used without development-set selection. The original pretrained LEADER is evaluated separately.

Evaluation uses the same 32 previously used development frames, original SC2 top-50% estimation, and unmodified two-stage full-pool refinement with 1.2/0.6m thresholds. Every model uses the same solver seed per frame. Missing-image and shuffled-association diagnostics run for all four visual arms. Report per-seed MPE/MOE and paired factorial differences, not a best seed; repeated seeds do not create independent test frames.

Projection uses audited raw-scan representatives and calibrated visibility; localization keeps original voxel coordinates. Patch neighbors are context candidates, not new 3D correspondences. No world ground-truth pose enters attention, projection, or feature extraction. Ground truth is used only for the existing regression training loss and evaluation.

The protocol records data, weight, and implementation hashes before extraction. The checks cover identical initialization, equal parameter participation, center-only isolation, single-token equivalence, and exact missing-image fallback even with invalid feature values.

Extraction initially stopped on a host-disk I/O error before any training. `recover_cache.py` losslessly recompresses this experiment's new caches, verifies exact array equality, and removes only incomplete new files for re-extraction. The original protocol is retained as `protocol_before_storage_fix.json`; training uses a freshly hashed protocol after the storage-only changes. On this host, WSL memory/swap limits were temporarily reduced for recovery; the original `.wslconfig` was restored after the run and verified against its backup. Other running WSL tasks were not restarted.

The completed result and interpretation are in `results/INTERPRETATION.md`; all 15 training runs and all paired evaluations finished. `audit.py` verifies identical backbone masks, identical center/neighborhood voxel coverage, finite features, and training-only PCA indices.
