# GLACE 独立 Camera 分支 + LEADER 共享几何后端

从 `v1-two-stage` (a90b142) 分出。两版架构都在本分支上：

- **v2（当前主架构）**：双 SCR 前端 + 共享几何后端 —— 每一帧都经过
  `候选池 → 共同评分 → 联合细化 → 验收`，见 `joint_solver.py`。
- **v1（保留作对照）**：置信度门控的 SELECT / LERP+SLERP / 冲突回退，
  见 `lidar_camera_fusion.py`（原固定融合模块 + 可选 `extra_hypotheses`）。

```
LEADER: 3D_local → 3D_world        (SC2-PCR, 前端不改)
GLACE:  2D_pixel → 3D_world        (scene coordinates, 前端不改)
                ↓ 两路对应关系 + 两路位姿候选
   共同评分 S(T) → 联合细化（单刚体 T_WB）→ 验收
   JOINT / SINGLE_MODAL / AMBIGUOUS / DEGENERATE / REJECTED
```

运行时依赖：NumPy、SciPy、OpenCV（v2 后端）；`lidar_camera_fusion.py`（v1 对照）另需 SciPy。

## 文件

| 文件 | 作用 |
| --- | --- |
| `joint_solver.py` | **v2 共享位姿求解器**。`solve(problem, T_L, T_C, seedwise, mode=...)`：候选池 = {T_L, T_C} ∪ 全部有效 SC2-PCR seedwise（不按 fitness 预筛）∪ 区域多样性 P3P 候选（`cv2.solveP3P`，T_CW→`H=inv(T_CW)@inv(E)`，预算 256）；权重 `w^L_i`（TRR 有界变换）+ 均匀 `w^C_j=1/N_C`；`S(T)=½Σw·min(||r||²,1)`；支持集上加权 Cauchy 联合细化（SciPy LM 精确实现 Ceres 配方的分块目标，单刚体变量、无 T_L/T_C 先验、细化后回全池重评分并保留原候选）；验收含两路支持、歧义 margin、`JᵀWJ` 可观测性；输出 JOINT / SINGLE_MODAL（明确标记的降级）/ AMBIGUOUS / DEGENERATE / REJECTED。mode：`select`（分别评分后选择）/ `joint`（仅共同评分）/ `joint_refine`（完整） |
| `glace_adapter.py` | GLACE adapter。`infer()` 返回 `GLACEOutput(T_WC, T_WB, uv, xyz_world, K, inlier_count, inlier_mask, ...)`：**不丢弃** `scene_coordinates_B3HW`，每个 8×8 cell 中心即一个 camera correspondence（`u = OUTPUT_SUBSAMPLE*(x+0.5)`，与 vendor `get_pixel_grid` 一致；uv 是预处理后输入图像的像素坐标）。v1 位姿求解用 OpenCV PnP-RANSAC+LM（不改 DSAC* C++）；`T_WB = T_WC @ inv(T_BC)` |
| `packet.py` | 融合 packet。`lidar_pool_from_export` 恢复 `p_B = T_corr⁻¹·c_local`、`P_W = c_pred + center_t`；诊断用支持率 `q_L/q_C`；`IsotonicCalibrator`（PAVA）；`make_fusion_evidence`（v1 接口）；`diverse_poses` |
| `lidar_camera_fusion.py` | v1 固定融合模块（自 camera-reliability 复制），可选 `extra_hypotheses` 注入外部候选；默认行为与原版一致 |
| `make_glace_scene.py` | 以 LEADER 世界系生成 GLACE 训练/测试场景：`T_WC_GT = T_WB_GT @ T_BC` 写 `poses/<ts>.txt`，缩放 K 写 `calibration/`，图像 `rgb/`（vendor CamLocDataset 布局），保存 timestamp pair 与时间差 |
| `run_fusion_eval.py` | 联合评测 runner。`--backend joint|compare|fallback`；`--backend compare` 逐帧同时跑 select / joint / joint_refine 三模式，用于归因收益来源（多模态证据 vs 多候选 vs 联合细化） |
| `test_glace_fusion.py` / `test_joint_solver.py` | 几何/接口单元测试（合成数据；SC2 测试需 torch） |

## LEADER 侧改动（前端定位路径不变）

- `run_mink.py`：测试循环中 top-50% 筛选**之前**保留全量池（`c_pred_all/u_pred_all/c_local_all`），原 top-50% → SC2-PCR 路径一字未动。`--export_fusion_pool DIR` 导出每帧 npz：全量池、`T_corr`、`center_t`（读自 checkpoint）、最终 `T_WB`、GT `T_WB_gt`、scan 时间戳、top-50% 索引，以及**全部**有效 seedwise 假设（`--export_seedwise 0` 默认；已恢复坐标 `H^L = A·H_raw·Q`，`A=[[I,center_t],[0,1]]`，`Q=T_corr`）。
- `models/sc2pcr.py`：`cal_seed_trans/SC2_PCR/estimator` 增加 `return_hypotheses=False` 关键字参数，为 `True` 时额外返回 `(final_trans, seedwise_trans, seedwise_fitness)`。默认行为与之前完全一致（正常 LEADER 路径仍只用 `final_trans`）。

### 坐标系约定（易错点）

- LEADER 训练目标是 `world - center_t`；`c_local` 位于 `T_corr`（地面水平化校正，raw body → leveled）之后的坐标系。
- packet 侧恢复：`p_i^B = T_corr⁻¹ · c_local_i`，`P_i^W = c_pred_i + center_t`，与 `T_WB = T_est @ T_corr`（`t += center_t`）作用于同一刚体变量 B；seedwise 同样恢复。
- GLACE 位姿约定为 `T_WC`（camera→world，`sc = pose @ camera_point`）；折算 `T_WB = T_WC @ inv(T_BC)`，`T_BC` 为 camera→body 外参（NCLT 标定链给出 body→camera，注意取逆）。GLACE 内部已加回自己的坐标均值，**不能**再加 LEADER 的 `center_t`。
- `c_L/c_C`（支持率标定）在 v2 中仅作诊断输出，不再参与决策。

## 运行

```bash
# Stage A: LEADER 导出（不改变 LEADER 定位结果；center_t 来自 checkpoint）
python run_mink.py --mode test --dataset NCLT --dataset_folder <NCLT> \
    --resume_model <ckpt> --export_fusion_pool /path/pool --export_seedwise 0

# Stage B: GLACE + 共享求解器评测（默认 v2）
python -m research.glace_fusion.run_fusion_eval \
    --pool_dir /path/pool --vendor_dir <ace-vendor> --glace_head <head.pt> \
    [--deit_checkpoint <CVPR23_DeitS_Rerank.pth>] \
    --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 \
    --out_dir /path/out --backend compare

# v1 置信度门控对照
python -m research.glace_fusion.run_fusion_eval ... --backend fallback

# 生成 LEADER 世界系的 GLACE 训练场景
python -m research.glace_fusion.make_glace_scene --dataset_folder <NCLT> \
    --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 --out /path/glace_scene

# 单元测试
python -m unittest research.glace_fusion.test_glace_fusion -v
python -m unittest research.glace_fusion.test_joint_solver -v
```

## 验证与注意事项

- **归因对比**：`--backend compare` 固定前端、对应关系、候选池与预算，逐帧比较 `select`（分别评分后选择）、`joint`（共同评分）、`joint_refine`（共同评分+联合细化），用于分清收益来源。
- **门限未经真实验证**：`JointSolverConfig` 的默认门限（支持数量/比例、max_score、single-modal 更严门限、歧义 margin、可观测性谱）是接线起始值。调参必须用独立验证 split，**不要用 LEADER 的 `val_loader`（实为测试序列）调门限**；调完后 `JointSolverConfig.save()` 冻结。
- **SINGLE_MODAL** 是明确标记的降级输出（一路通过更严格的单路验收、另一路在全部候选上无支持），不得报告为两路共同确认；"候选池中无第二解"不等于全局唯一。
- **f_x=f_y**：GLACE 官方 DSAC* 接口要求单一焦距；若日后接 DSAC*，预处理需重映射为无畸变、f_x=f_y 的虚拟针孔图像并同步 K（OpenCV 求解路径无此限制）。
- 端到端计时必须包含 GLACE 全局特征提取，不能把缓存 `features.npy` 当免费输入。

## 数据平台说明

设计文档以 Oxford 为例，但本服务器没有 Oxford velodyne/图像原始数据；全部已有相机基础设施（NCLT_camera_v1 六相机、标定链、训练过的 ACE/GLACE ROI heads）都在 NCLT 上，故接线与验证平台为 **NCLT**（Cam5，2 Hz 分组同步，超时差门限的帧对不进入联合求解）。代码对数据集保持通用：Oxford 只需按 `make_glace_scene.py` 的同步约定补一个数据适配器（stereo 左目 pinhole 流），架构零改动。

