# Pre-voxel six-view + NULL fusion (Track A, boundary-validation round)

## Integrated LEADER smoke entry

`leader_model.py` wraps the repository's original `models.model_mink.LEADER`.
The original encoder, decoder, and pose-side code remain frozen. The trainable
path is `ViewAdapter + ViewWeighting + NULL + VisualProjection`; the visual
residual is added after the original first LiDAR stem block. `smoke_dataset.py`
uses the same raw-point order and ME representative `index` for LiDAR and
visual features. The polar angular scale is `voxel_size * horizontal_res`,
the quantization values are parsed from the active `run_mink.py` defaults, and
the TRR loss is loaded from that same file. `VisualProjection` is zero-initialized
so the initial multimodal function is the frozen LEADER function; the normal
backbone path is still used when its residual is zero. `train_smoke.py` runs the
fixed 64/32 manifest split, performs a real backward step, checks all validation
frames, runs a strict pre-training B0 equivalence check, and writes a report.
The Transformer view weights are renormalized conditional on using Camera;
NULL/reliability gates the projected residual once.

Example in the bundled Ubuntu environment:

```bash
python research/prevoxel_multiview/train_smoke.py \
  --manifest /home/zhang/leader-image-gate-multicamera/manifest.json \
  --checkpoint /mnt/c/Users/zhang/Documents/ChatGPT/LEADER/research/image_gate_checkpoint \
  --output research/prevoxel_multiview/results/smoke \
  --steps 2
```

The quantization defaults and TRR implementation are read from the original
`run_mink.py`; the visual configuration intentionally has no independent
voxel-size value. The smoke validation is a loss-only check over all 32 frames;
it does not claim pose-solver localization metrics. The strict B0 check runs
the same comparison on CPU because this MinkowskiEngine CUDA build is
numerically nondeterministic across separate sparse-UNet calls; the check
itself requires zero difference for coordinates, LiDAR features, encoded
features, prediction, target, and TRR loss.

## Frozen voxel residual probe

`residual_probe.py` is deliberately separate from the trainable fusion entry.
It consumes the already aligned frozen-LEADER output caches: `target -
prediction` is the residual, while `all_features/*.npz:image` is the fixed
representative-voxel DeDoDe/PCA128 descriptor. Valid views were averaged when
those caches were created; the untrained Transformer is not used. The script
validates the six-view masks, trains the same small probe on real and
within-view-count stratified shuffled features, then reports all voxels and
0/1/2/3-view strata on the 32-frame validation split.

```bash
python research/prevoxel_multiview/residual_probe.py \
  --manifest /home/zhang/leader-image-gate-multicamera/manifest.json \
  --checkpoint /mnt/c/Users/zhang/Documents/ChatGPT/LEADER/research/image_gate_checkpoint \
  --visual-cache /home/zhang/leader-image-gate-multicamera/all_features \
  --lidar-cache /home/zhang/leader-image-gate/lidar \
  --output research/prevoxel_multiview/results/residual_probe \
  --steps 600 --seeds 2089,2090,2091
```

The decisive fields are `results[*].real_beats_shuffled` and
`results[*].real.with_view` versus `results[*].shuffled.with_view`, together
with the validation `relative_reduction` and the per-view-count entries. A
positive real-versus-shuffled separation is evidence that fixed image evidence
can predict part of the frozen LEADER residual; it is not a localization result.
The output also records the checkpoint SHA-256 and the 0/1/2/3-view voxel
histograms for train and validation. `summary.all_seeds_real_beats_baseline_with_view`
is the strict version of the proposed test: it must be true before claiming a
stable residual-prediction gain; real-versus-shuffled alone can be a small
relative separation without lowering the frozen baseline error.

预注册: `2026-09-18_prevoxel_sixview_null` (kill_test 级, 仅代码 + sanity, 不启动训练).

规格来源: 用户 2026-09-18 粘贴的实现规格书 (raw-point 六视图投影 → 可见性/质量 →
ViewAdapter → DeepChoice 式 Transformer 加权 + NULL → 点级视觉特征 → 原 voxelizer)。

## 文件

| 文件 | 内容 |
|---|---|
| `config.json` | 全部可调参数 (禁止 hard-code) |
| `projector.py` | NCLT uint16 bin 解析 (lb3 系) → body → 6 相机投影, 复用官方 ssc 标定链 |
| `visibility.py` | 硬过滤 (z>0 / 图内 / 黑边) + sparse z-buffer 遮挡检测 (τ_occ(z)=max(0.5, 0.03z)) |
| `image_feature.py` | 冻结 DeDoDe descriptor-B + PCA128, grid_sample 双线性采样 |
| `quality.py` | range / border distance / depth margin+evidence / sharpness / contrast / saturation / cross-view consistency |
| `fusion.py` | ViewAdapter (共享) + NULL token Transformer (d=64, 2层, 4头) + 点级融合 v_i=Σα_ij·z_ij, r_i=1−α_NULL |
| `dataset_hook.py` | NCLT_mink.__getitem__ 的 pre-voxel 注入: scan(lb3)→visual features→与原 feats/coords 对齐拼接 |
| `sanity_check.py` | Check1 force-NULL==baseline (数值等价) / Check4 invalid α=0 / Check5 无视图→α_NULL=1,v=0 / 覆盖统计 |

## LEADER 接入点 (实测确认)

- `LEADER/data/NCLTVelodyne_datagenerator_mink.py` `__getitem__` L153-162:
  raw point 的最后位置在 `ME.utils.sparse_quantize` 之前。scan 为 lb3 传感器系
  (get_velo: uint16×0.005−100),投影链 = inv(ssc(x_lb3_c)) · inv(ssc(x_body_lb3)),
  与 `research/projection_audit_assets/project_vel_to_cam.py` 官方脚本逐位一致
  (camera_to_body 与 manifest 最大偏差 2.2e-16, 已实测)。
- sparse_quantize 是 barrier: 量化会合并/重排 point index。当前 ME 构建对每个
  quantized row 选择一个代表 raw point；因此视觉特征严格使用同一个 `index` gather，
  输出 [M, 64] 与原 [M, 3] features 同源。它**不**会汇聚同一 voxel 内其他 raw point
  的视觉信息。
- `LEADER/models/model_mink.py` RPGE(in_channels=3→68): 新增通道按规格书 §12
  W_V=0 初始化 + W_L 复制原权重 (见 fusion.py `LEADERFirstLayerExtension`)。

## 与已判负家族的差异 (预注册 novelty, 勿混述)

- 融合在 **raw point 级** (不是 voxel 特征级 / 512D 接口);
- 每视图先独立编码再 Transformer 加权 (**不是 raw DeDoDe 直接平均**);
- 显式 **NULL** 退出 (α_NULL→1 ⇒ v_i→0 ⇒ 等价 LiDAR-only);
- 禁止修改 xyz / voxel index / 标定 / solver; GT 只进训练 loss。

## 数据路径 (本地实测)

- 图像 Cam0-4: WSL `Ubuntu` `/home/zhang/leader-image-gate-multicamera/Cam{0..4}/` (96/96)
- Cam5 原始缓存: `C:/Users/zhang/Documents/ChatGPT/LEADER/glace-local/data/validation_scene/train/rgb/` (96/96)
- 扫描: `glace-local/data/scans/2012-02-18/velodyne_sync/` (301 bins)
- 标定: `research/projection_audit_assets/{K_cam*.csv,x_lb3_c*.csv}` + Cam0-4 `K.txt`
- 注意: 存图为 **808×616 半分辨率** (K 已匹配), 不是 1616×1232; 原始 tiff 16bit 在
  官方包里, jpg 为 8bit 转存。

## 运行 (WSL Ubuntu, egonn118=训练环境 / bufferx=DeDoDe 环境)

```bash
wsl -d Ubuntu -- bash -c 'cd /mnt/c/Users/zhang/Documents/ChatGPT/LEADER/research/prevoxel_multiview && \
  /home/zhang/miniconda3/envs/egonn118/bin/python sanity_check.py --frames 8'
```
