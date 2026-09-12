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
| `make_glace_scene.py` | 以 LEADER 世界系生成 GLACE 训练/测试场景：`T_WC_GT = T_WB(t_camera) @ T_BC`（真实曝光时间，平移线性插值、旋转 SLERP）写 `poses/<ts>.txt`，与实际保存图像尺寸匹配的 K 写 `calibration/`，由 vendor loader 唯一负责后续 resize，图像 `rgb/`（vendor CamLocDataset 布局），保存 timestamp pair 与时间差 |
| `run_fusion_eval.py` | 联合评测 runner。`--backend joint|compare|fallback`；`--backend compare` 逐帧同时跑 select / joint / joint_refine 三模式，用于归因收益来源（多模态证据 vs 多候选 vs 联合细化） |
| `test_glace_fusion.py` / `test_joint_solver.py` | 几何/接口单元测试（合成数据；SC2 测试需 torch） |

## LEADER 侧改动（前端定位路径不变）

- `run_mink.py`：测试循环中 top-50% 筛选**之前**保留全量池（`c_pred_all/u_pred_all/c_local_all`），原 top-50% → SC2-PCR 路径一字未动。`--export_fusion_pool DIR` 导出每帧 npz：全量池、`T_corr`、`center_t`（读自 checkpoint）、最终 `T_WB`、GT `T_WB_gt`、scan 时间戳、top-50% 索引，以及**全部**有效 seedwise 假设（`--export_seedwise 0` 默认；已恢复坐标 `H^L = A·H_raw·Q`，`A=[[I,center_t],[0,1]]`，`Q=T_corr`）。
- `models/sc2pcr.py`：`cal_seed_trans/SC2_PCR/estimator` 增加 `return_hypotheses=False` 关键字参数，为 `True` 时额外返回 `(final_trans, seedwise_trans, seedwise_fitness)`。默认行为与之前完全一致（正常 LEADER 路径仍只用 `final_trans`）。

### 坐标系约定（易错点）

- LEADER 训练目标是 `world - center_t`；`c_local` 位于 `T_corr`（地面水平化校正，raw body → leveled）之后的坐标系。
- packet 侧恢复：`p_i^B = T_corr⁻¹ · c_local_i`，`P_i^W = c_pred_i + center_t`，与 `T_WB = T_est @ T_corr`（`t += center_t`）作用于同一刚体变量 B；seedwise 同样恢复。
- GLACE 位姿约定为 `T_WC`（camera→world，`sc = pose @ camera_point`）；折算 `T_WB = T_WC @ inv(T_BC)`，`T_BC` 为 camera→body 外参（NCLT 链为 `T_BC = T_B_LB3 @ T_LB3_C`，此处不取逆）。GLACE 内部已加回自己的坐标均值，**不能**再加 LEADER 的 `center_t`。
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
    --dataset_folder <NCLT-parent> --image_resolution 616 \
    --out_dir /path/out --backend compare

# v1 置信度门控对照
python -m research.glace_fusion.run_fusion_eval ... --backend fallback

# 生成 LEADER 世界系的 GLACE 训练场景
python -m research.glace_fusion.make_glace_scene --dataset_folder <NCLT> \
    --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 --out /path/glace_scene

# 单元测试
python -m unittest research.glace_fusion.test_glace_fusion -v
python -m unittest research.glace_fusion.test_joint_solver research.glace_fusion.test_runner -v
```

## 验证与注意事项

- **归因对比**：`--backend compare` 固定前端、对应关系、候选池与预算，逐帧比较 `select`（分别评分后选择）、`joint`（共同评分）、`joint_refine`（共同评分+联合细化），用于分清收益来源。
- **门限未经真实验证**：`JointSolverConfig` 的默认门限（支持数量/比例、max_score、single-modal 更严门限、歧义 margin、可观测性谱）是接线起始值。调参必须用独立验证 split，**不要用 LEADER 的 `val_loader`（实为测试序列）调门限**；调完后 `JointSolverConfig.save()` 冻结。
- **SINGLE_MODAL** 是明确标记的降级输出（一路通过更严格的单路验收、另一路在全部候选上无支持），不得报告为两路共同确认；"候选池中无第二解"不等于全局唯一。
- **f_x=f_y**：GLACE 官方 DSAC* 接口要求单一焦距；若日后接 DSAC*，预处理需重映射为无畸变、f_x=f_y 的虚拟针孔图像并同步 K（OpenCV 求解路径无此限制）。
- 端到端计时必须包含 GLACE 全局特征提取，不能把缓存 `features.npy` 当免费输入。

## 数据平台说明

设计文档以 Oxford 为例，但本服务器没有 Oxford velodyne/图像原始数据；全部已有相机基础设施（NCLT_camera_v1 六相机、标定链、训练过的 ACE/GLACE ROI heads）都在 NCLT 上，故接线与验证平台为 **NCLT**（Cam5，2 Hz 分组同步，超时差门限的帧对不进入联合求解）。代码对数据集保持通用：Oxford 只需按 `make_glace_scene.py` 的同步约定补一个数据适配器（stereo 左目 pinhole 流），架构零改动。


## 正式 NCLT 实验的数据契约

- 固定 LEADER 划分：训练 `2012-01-22 / 2012-02-02 / 2012-02-18 / 2012-05-11`，测试 `2012-02-12 / 2012-02-19 / 2012-03-31 / 2012-05-26`。scene 生成器拒绝跨划分日期、训练/测试重叠和非空输出目录，防止旧文件混入；验证数据需从允许的训练序列中独立留出，不能在测试序列调门限。
- 默认要求请求的相机序列齐全；缺数据时直接报错。只有明确的局部实验才使用 `--allow_partial`，报告会列出缺失序列。该选项不会补齐数据。
- `rgb/` 保留源图像实际尺寸，`calibration/` 的 K 对应该尺寸：原始 1616×1232 图像写原始 K，已经保存为 808×616 的图像写半尺寸 K。不能给半尺寸图像写原始 K。scene 不再接受预设 `--image_size`；它读取文件尺寸和 metadata 原始尺寸。
- GLACE 训练设置 `--image_resolution 616`，runner 同样使用 `--image_resolution 616`，等比例缩放并使用与 CamLocDataset 一致的插值、灰度和标准化。DeiT 全局特征使用其独立的 480×640 输入尺寸；训练特征提取必须使用同一预处理，旧 head 不可未经验证直接混用。
- scene 标签由 NCLT 原始 GT trajectory 在 `original_image_timestamp` 插值；不再依赖 LiDAR loader 或最近 scan。runner 也使用真实曝光时间匹配，按相机时刻评估相机位姿、按 scan 时刻评估 LiDAR 和联合位姿；不外推 GT。
- `--dataset_folder` 指包含 `NCLT/` 的父目录。GT 仅用于标签和误差评估，不参与候选生成、评分、细化和验收。两路非同时采样的运动误差并不会因为插值 GT 自动消失：当前求解仍使用 `--max_sync_delta_s` 内近似同时观测，报告明确记录该近似；若需要消除它，必须引入独立于 GT 的运动估计或严格同步采集。
- 报告的 `leader_baseline / glace_baseline / select / joint / joint_refine` 使用同一有图像子集；`leader_all_input_export_gt` 单独保留全部输入 LiDAR 的原导出 GT 口径，不与插值 GT 子集混称同一指标。报告包含输入数、同步跳过数、GT 越界数、共同子集占比，拒绝帧计入定位成功率分母。
- 新增 runner 回归测试使用替代视觉输出，真实执行数据读取、几何求解和 JSON 汇总，覆盖 joint/compare/fallback；它不等于真实 LEADER+GLACE 网络端到端能力验证。


## 全训练集 GLACE head

`train_nclt_head.py` 仅使用四个训练日期的 Cam5 图像，保存源码快照、权重哈希、参数与阶段状态；不读取测试图像用于训练。ACE encoder 和 DeiT 固定，仅训练回归头。全局特征严格使用 adapter 的同一灰度三通道路径，480×640 与现有 DeiT checkpoint 的 1202 个位置 token 对应。

```bash
python -m research.glace_fusion.train_nclt_head \
  --out /path/new_run --vendor /path/glace_vendor \
  --deit_checkpoint /path/CVPR23_DeitS_Rerank.pth \
  --dataset_folder /root/rivermind-data/datasets \
  --camera_root /root/rivermind-data/datasets/NCLT_camera_v1
```

训练为 30000 次更新、batch 8192、每张图像 128 个局部特征样本、616 图像高度；特征缓冲覆盖一次全部有效训练图像。`state.json` 的 `complete` 表示权重已保存且训练图像推理检查通过，不表示 NCLT 测试集指标已验证。


## 2026-09-12 当前实验状态

以下是既有 K64 / 60k 灰度全局特征权重的诊断记录；该权重未通过定位验收。原训练入口的 `complete` 只表示训练和有限值检查结束；定位验收由 `validate_training_head.py` 单独执行。新 RGB 实验见下节，不能沿用旧灰度推理入口。

- 当前训练缓存和推理均使用灰度复制三通道的全局特征，**偏离官方 GLACE/R2Former 的 RGB 全局输入流程**。直接读取现有 `features.npy` 不会将其变成官方 RGB 特征；不得不经评估就把已有 head 的输入切换为 RGB。
- 修正外参后的 60000 次恢复实验配置在 `experiments/retrain_corrected.py`。这轮同时改变了多个参数，是失败恢复实验的记录，不是单因素归因或推荐训练配方。
- `camera_separability.py`：原验收的 64 张训练图像，固定对应点，比较 GT、原始 LEADER 和 GT+2m/5°。GT 在 64/64 张上胜过该固定扰动，不代表测试集排序有效。
- `camera_separability_cached.py`：直接按图像排序读取训练缓存，并复用上一轮的三个候选矩阵。平均评分基本不变；坐标逐点并非完全相同，仍有未定位的数值敏感性。
- `diagnostic_comparison.py`：2012-02-12 的固定 64 帧同帧比较。64 帧融合均拒绝并回退到 v1-two-stage，没有观察到相机增益。
- `pairwise_camera_ranking.py`：同一组未训练测试帧，分开构造平移/旋转、六个方向的候选。严格两两排序准确率为平移 45.77%、旋转 52.99%；平局半分对照为 50.69%、57.55%。细粒度排序接近随机，目前不支持把该 head 用于 LEADER 候选精排。
- 上述测试结果仅覆盖一个测试序列的 64 帧，**没有完成完整 NCLT 测试集评测**。`full_comparison.py` 是尚待完整端到端验证的入口，要求 head 先通过定位验收并要求测试图像齐全。

所有诊断脚本保留产生当前结果时的服务器绝对路径和非覆盖输出检查，需要现有工程、数据、依赖和权重，不能在空目录直接运行。它们不会在导入时启动训练或评测。`experiments/` 的历史训练/监控脚本是独立可执行记录，不应作为模块导入。

`experiments/summarize_*.py` 汇总对应运行目录中的记录，生成 JSON/CSV/图表；不重新生成候选或训练模型。数据、权重、凭据、日志及 Python 缓存不属于此次源码提交。

```bash
python -m unittest research.glace_fusion.test_glace_fusion research.glace_fusion.test_joint_solver research.glace_fusion.test_runner research.glace_fusion.test_pose_boundary research.glace_fusion.test_camera_separability research.glace_fusion.test_pairwise_camera_ranking
```

## NCLT RGB large-scale baseline（2026-09-12）

`retrain_rgb_baseline.py` 从头训练新 head，保留旧 K64 / 60k 权重及诊断结果。该实验同时恢复多项配置，是新基线，不是 Feature Diffusion 的单因素消融。

| 项目 | 新配置 |
| --- | --- |
| 全局输入 | 原始存储 RGB → 官方 R2Former 480×640 → 按文件名排序缓存 |
| 局部输入 | 官方灰度归一化，高度 480；resize 宽度使用官方 round |
| Feature Diffusion | 0.1 |
| 增强 | 旋转 ±15°、缩放 1/1.5–1.5，保留亮度／对比度增强 |
| Head | blocks=3、mlp_ratio=2、channels=768、decoder clusters=50 |
| 训练 | 100000 iterations，batch=40960，soft clamp=50 |
| 样本 | 四个原训练日期，共 43012 张，每张 1024 个样本 |
| 缓存 | CPU 内存中 44044288 个样本，约 47.58 GiB；逐图数量强制校验 |
| Seed | 2089，显式传入实际 Trainer；KMeans seed=0 |

这是单张 RTX 3090 上的适配版本，**不等同于官方 Aachen 八卡训练计算预算**：有效 batch 是 40960，而非 8×40960；缓存也从每 GPU 16M 改为 CPU 保存所有训练帧的样本。ACE encoder 和 R2Former 都冻结，只有 head 训练。

NCLT 标定主点不是严格的图像中心。恢复旋转增强时，图像与 mask 绕标定主点旋转，pose 保持官方右乘旋转规则；若 fx/fy 不相等则拒绝该训练路径。场景 K 对应存储图像尺寸，只有 loader 对 K 随缩放调整。所有几何更改及 CPU 缓存适配都保存在本轮 vendor 快照中，哈希记录于 config.json。

启动顺序为：标签与 split 检查 → RGB 全量特征 → loader/缓存/旋转一致性预检 → 完整 batch 的 20 轮 GPU 冒烟训练 → 正式训练。预检通过不表示定位或测试泛化通过；`state.json` 的 `ready_for_fusion` 不会自动置为 true。

```bash
python -m research.glace_fusion.retrain_rgb_baseline \
  --run-root /root/rivermind-data/glace_nclt_rgb_large_20260912 \
  --source-run /root/rivermind-data/glace_nclt_corrected_20260912 \
  --deit-checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth
```

入口拒绝覆盖已有目录；上述目录已用于本轮训练。检查当前运行：

```bash
cat /root/rivermind-data/glace_nclt_rgb_large_20260912/state.json
cat /root/rivermind-data/glace_nclt_rgb_large_20260912/glace_head.pt.progress.json
tail -f /root/rivermind-data/glace_nclt_rgb_large_20260912/train.log
```

本轮预检结果：RGB 缓存与在线路径差值 0；局部图像与官方 loader 差值 0；K 最大绝对差 4.08e-6；旋转投影检查通过；batch=40960 冒烟训练产生有效优化更新。46 个回归测试通过，其中新增缓存顺序/完整性、480 分辨率取整和 RGB head 拒绝旧灰度输入测试。

训练后的定位、细粒度两两排序、跨四个测试序列、真实 LEADER 候选、空间可见性以及数值敏感性仍需独立验收。DSAC* 尚未在本环境编译，不能把旧 OpenCV 4px 的定位结果表述为官方 DSAC* 结果。新 head 的 config 会使旧灰度 adapter 明确报错，避免静默使用错误的全局输入。

## RGB inference / evaluation contract

`GLACEAdapter.infer(gray480, K480, global_feature=feature256)` 显式接收全局特征，新 RGB head 禁止灰度 callback。上层 `InferenceSession.infer(rgb_path, K_stored)` 统一负责按训练配置缩放局部图像及 K，并从已校验的缓存或原始 RGB 路径取全局特征。不要将已经缩放过的 K 再传给 session。

`InferenceSession` 校验全局 backbone 和局部 encoder 的训练哈希、缓存哈希、文件顺序与实际图像路径。报告中的 head 哈希来自真正加载的同一份字节，避免训练期间检查点更新造成标识错配。权重需与本轮 `config.json` 放在同一目录；复制中间检查点时也要复制配置。

新 RGB head 默认采用 `fp32_head`：ACE encoder 仍使用 AMP，head 使用 FP32 且关闭 TF32；不会改变训练。`--coordinate-precision amp` 可显式恢复官方 AMP 对照，fusion 入口对应 `--coordinate_precision amp`。所有报告记录精度模式，不能混用两种模式比较。两张真实训练图、中间权重的缓存／在线一致性检查中，AMP 最大坐标差为 5.85m；FP32 head 降为 0.0652m，重复同一缓存的坐标、K 与 pixel grid 完全一致。这只是数值检查，不是整个数据集的误差上界或定位验收。

准备独立 test scene，不向训练场景加入测试图像：

```bash
python -m research.glace_fusion.make_glace_scene \
  --dataset_folder /root/rivermind-data/datasets \
  --camera_root /root/rivermind-data/datasets/NCLT_camera_v1 \
  --out /root/rivermind-data/glace_nclt_rgb_eval_20260912/scene \
  --train_dates --allow_partial

python -m research.glace_fusion.rgb_features \
  --scene /root/rivermind-data/glace_nclt_rgb_eval_20260912/scene \
  --vendor /root/rivermind-data/glace_nclt_rgb_large_20260912/vendor \
  --checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth
```

本次已准备 5117 张 test 图像的 RGB 缓存；现有 camera root 仅提供一个测试日期，meta 会明确记录其余缺失日期，不能称为完整四序列测试集。场景和特征输出拒绝覆盖，上述目录已存在。

新旧 head 共用验收入口：

```bash
python -m research.glace_fusion.validate_training_head \
  --run-root /root/rivermind-data/glace_nclt_rgb_large_20260912 \
  --head /root/rivermind-data/glace_nclt_rgb_large_20260912/glace_head.pt \
  --vendor /root/rivermind-data/glace_nclt_rgb_large_20260912/vendor \
  --deit-checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth \
  --output /root/rivermind-data/glace_nclt_rgb_eval_20260912/final_train64.json

python -m research.glace_fusion.pairwise_camera_ranking \
  --scene /root/rivermind-data/glace_nclt_rgb_eval_20260912/scene \
  --head /root/rivermind-data/glace_nclt_rgb_large_20260912/glace_head.pt \
  --vendor /root/rivermind-data/glace_nclt_rgb_large_20260912/vendor \
  --deit-checkpoint /root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth \
  --out /root/rivermind-data/glace_nclt_rgb_eval_20260912/final_test64 --limit 64
```

`pairwise_camera_ranking` 默认使用新通用 scene 入口；`--legacy` 才运行原硬编码灰度实验。`--image-stems stems.json` 可固定与旧报告完全相同的帧，文件内容是 stem 字符串数组。默认 `--pose-backend none` 只计算固定 correspondence 的 GT residual、10px inlier rate 和原截断平方分数的候选排序，不把 PnP 成功作为排序前提；`--online` 改从 RGB 路径提取全局特征。records 包含相邻误差区间计数、严格排序准确率、平局数量和半分对照。

`--pose-backend opencv` 使用新 RGB 默认 10px / 1000 iterations EPNP + LM；`--pose-backend dsacstar` 使用 DSAC* 10px / 3200 hypotheses，要求已安装官方扩展且 fx=fy；缺失扩展时直接报错，不回退伪装成 DSAC*。独立 pose backend 和坐标精度都写入报告。当前环境未安装 DSAC*，真实验证覆盖的是 OpenCV 和 correspondence 路径。

fusion runner 自动读取新权重的 480 配置，支持 `--feature_split <scene/test>`，省略则在线 RGB 提特征；显式指定不匹配的 `--image_resolution 616` 会报错。可用 `--pose_backend none` 在不依赖相机独立 PnP 的情况下运行候选证据流程。融合求解公式没有修改。

本次检查包含：47 个回归测试、真实中间权重的两图缓存／在线对照、64 张训练图的验收 runner、两张 held-out 图像的缓存 → scene coordinates → OpenCV / GT residual / pairwise report。中间权重的结果不代表 100k 最终模型质量，不用于选择检查点或提前停止训练。
