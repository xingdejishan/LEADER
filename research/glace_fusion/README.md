# GLACE 独立 Camera 分支 + LEADER 融合接线

从 `v1-two-stage` (a90b142) 分出。实现的核心结构变更：

```
LEADER: 3D_local → 3D_world        (SC2-PCR, 不改动原定位路径)
GLACE:  2D_pixel → 3D_world        (scene coordinates, 独立分支)
                ↓ 两者在固定好的 pose + correspondence fusion 层汇合
   (T_L, c_L, T_C, c_C) → SELECT / LERP+SLERP / FALLBACK
```

运行时仅需 `lidar_camera_fusion.py`（NumPy/SciPy/OpenCV），其余为数据接线与评测工具。

## 目录

| 文件 | 作用 |
| --- | --- |
| `lidar_camera_fusion.py` | 固定融合模块（自 camera-reliability 复制），新增可选 `extra_hypotheses` 参数：外部候选（如 SC2-PCR seedwise 池）与内部采样候选一起进入统一的评分/聚类/联合优化/验收流程；默认 `None` 时行为与原版完全一致 |
| `glace_adapter.py` | GLACE adapter。`infer()` 返回 `GLACEOutput(T_WC, T_WB, uv, xyz_world, K, inlier_count, inlier_mask, ...)`：**不丢弃** `scene_coordinates_B3HW`，每个 8×8 cell 中心即一个 camera correspondence（`u = OUTPUT_SUBSAMPLE*(x+0.5)`，与 vendor `get_pixel_grid` 一致）。位姿求解 v1 用 OpenCV PnP-RANSAC+LM（不改 DSAC* C++）；`T_WB = T_WC @ inv(T_BC)` |
| `packet.py` | 融合 packet 构造。`lidar_pool_from_export` 恢复 `p_B = T_corr⁻¹·c_local`、`P_W = c_pred + center_t`；支持率 `q_L/q_C`（见下）；`IsotonicCalibrator`（PAVA，验证集标定 `P(success|q)`）；`make_fusion_evidence` 产出融合模块的 `FusionEvidence`；`diverse_poses` 做 pose-distance 互异候选挑选 |
| `make_glace_scene.py` | 设计点 6：以 LEADER 世界系生成 GLACE 训练/测试场景。对每个同步对写 `T_WC_GT = T_WB_GT @ T_BC` 到 `poses/<ts>.txt`、缩放后的 K 到 `calibration/<ts>.txt`、图像到 `rgb/`（vendor CamLocDataset 布局），并保存 timestamp pair 与时间差 |
| `run_fusion_eval.py` | 联合评测：LEADER 导出池 + GLACE adapter → `localize()` → LEADER/GLACE/融合三路误差报告 |
| `test_glace_fusion.py` | 几何/接口单元测试（合成数据，无需 GPU；SC2 测试需 torch） |

## LEADER 侧改动（核心路径不变）

- `run_mink.py`：测试循环中 top-50% 筛选**之前**保留全量池（`c_pred_all/u_pred_all/c_local_all`），原 top-50% → SC2-PCR 路径一字未动。`--export_fusion_pool DIR` 导出每帧 npz：全量池、`T_corr`、`center_t`、最终 `T_WB`、GT `T_WB_gt`、scan 时间戳、top-50% 索引，以及 `--export_seedwise`（默认 8）个 pose-distance 互异的 seedwise 假设（已转换回 raw body→world：`T_WB_seed = T_seed·T_corr`，`t += center_t`）。
- `models/sc2pcr.py`：`cal_seed_trans/SC2_PCR/estimator` 增加 `return_hypotheses=False` 关键字参数，为 `True` 时额外返回 `(final_trans, seedwise_trans, seedwise_fitness)`。默认行为与之前完全一致（正常 LEADER 路径仍只用 `final_trans`）。

### 坐标系约定（易错点）

- LEADER 训练目标是 `world - center_t`；`c_local` 位于 `T_corr`（地面水平化校正，raw body → leveled）之后的坐标系。
- 因此 packet 侧恢复：`p_i^B = T_corr⁻¹ · c_local_i`，`P_i^W = c_pred_i + center_t`，与 `T_WB = T_est @ T_corr`（`t += center_t`）作用于同一刚体变量 B。
- GLACE 位姿约定为 `T_WC`（camera→world，`sc = pose @ camera_point`）；折算 `T_WB = T_WC @ inv(T_BC)`，`T_BC` 为 camera→body 外参（融合模块约定；NCLT 标定链给出 body→camera，注意取逆）。

## 置信度（设计点 10）

`u_pred`（3D-3D 对应可靠性）与 `inlier_count`（相机 RANSAC 内点数）不可比。统一改为"最终 pose 在本模态完整 correspondence 池上的支持率"：

```
q_L = #{ ||T_L p_i − P_i||  < s_L } / N_L          (s_L 默认 0.3 m)
q_C = #{ ||π((T_L·E)⁻¹ P_j) − u_j|| < s_C } / N_C   (s_C 默认 4 px，负深度计外点、保留分母)
```

再经验证集标定的 `f(q)=P(success|q)`（isotonic，`IsotonicCalibrator`）映射到 [0,1]。未标定时为恒等映射 —— 此时快速路径的门限判断（`conf_use/conf_gap`）**尚未校准**，结果解读需谨慎；先在验证 split 上用 `records.json` 的 `q_*`/成功标签拟合，再冻结配置用于独立测试。

## 运行

```bash
# Stage A: LEADER 导出（不改变 LEADER 定位结果）
python run_mink.py --mode test --dataset NCLT --dataset_folder <NCLT> \
    --resume_model <ckpt> --export_fusion_pool /path/pool --export_seedwise 8

# Stage B: GLACE + 融合评测
python -m research.glace_fusion.run_fusion_eval \
    --pool_dir /path/pool --vendor_dir <ace-vendor> --glace_head <head.pt> \
    [--deit_checkpoint <CVPR23_DeitS_Rerank.pth>] \
    --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 \
    --out_dir /path/out [--use_extra_hypotheses] [--lidar_conf cal_L.json --camera_conf cal_C.json]

# 生成 LEADER 世界系的 GLACE 训练场景（设计点 6）
python -m research.glace_fusion.make_glace_scene --dataset_folder <NCLT> \
    --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 --out /path/glace_scene

# 单元测试
python -m unittest research.glace_fusion.test_glace_fusion -v
```

## 数据平台说明

设计文档以 Oxford 为例，但本服务器没有 Oxford velodyne/图像原始数据；全部已有相机基础设施（NCLT_camera_v1 六相机、标定链、训练过的 ACE/GLACE ROI heads、融合模块的测试数据）都在 NCLT 上，故接线与验证平台为 **NCLT**（`--camera Cam5`，2 Hz 分组同步）。代码对数据集保持通用：Oxford 只需按 `make_glace_scene.py` 的同步约定补一个数据适配器（stereo 左目 pinhole 流），架构零改动。

## 回退候选（设计点 9）

`H = {T_L, T_C} ∪ H_L^{SC2-PCR seedwise} ∪ H_C^{AP3P}`：

- `H_C` 与部分 `H_L` 由融合模块内部生成（uniform 3 点 SVD / 4 点 AP3P，交替采样）；
- SC2-PCR 内部本就为每个 seed 生成 `seedwise_trans` 并按 `seedwise_fitness` 评分，经 `--export_seedwise` 导出 pose-互异子集后，用 `--use_extra_hypotheses` 注入融合模块，与内部候选统一评分、联合优化（SciPy LM，分块 Cauchy 目标）、支持/歧义/可观测性验收。
