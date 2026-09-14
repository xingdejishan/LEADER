# 多模态融合推理交接

主 baseline 是原始纯 LEADER；多模态对照是修复投影后的①。②③④均未超过原始 LEADER，不能作为已有效的新 baseline。

本页与全部代码、报告位于 [`research/leader-image-gate`](https://github.com/xingdejishan/LEADER/tree/research/leader-image-gate) 分支。权重和96帧复现输入/缓存位于 [配套 Release](https://github.com/xingdejishan/LEADER/releases/tag/multimodal-handoff-2026-09-14)，文件清单和 SHA-256 见 [assets_manifest.json](assets_manifest.json)。不需要另一个 agent 访问原电脑就能阅读全部推理依据。

## 先阅读

1. [冻结权重诊断：为何投影修复后最终定位仍变差](results/fusion_diagnostic/REPORT.md)
2. [四条线完整指标](results/COMPARISON_1_2_3_4.md)
3. [① 修复投影后的门控训练](results/raw_retrain/REPORT.md)
4. [② LP-GRF 文档版、蒸馏和两阶段微调](results/lpgrf_retrain/REPORT.md)
5. [③ Node2Vec 和 coarse/refinement 上下文](results/context_retrain/REPORT.md)
6. [④ 三维监督上下文](results/geometry_retrain/REPORT.md)
7. [投影审计与图像叠加](results/projection_audit/REPORT.md)
8. [原设计文档](docs/LEADER+R-SCoRe.md)及[研究约束](docs/LEADER_RESEARCH_GUIDELINES.md)

辅助背景：[此前 GLACE 三维监督结果](docs/stage2_results.md)、[其最终定位验证](docs/stage3_full_region_report.md)。它们不是本轮四条线的结果，不能混用。

## 可以直接交给另一个 agent 的任务

请分析 LEADER 多模态融合的改进方向，先做机制推理和实验设计，不直接增加模块或开始训练。

主 baseline：本地保存的 upstream/main（4a1bde8）对应的原始纯 LEADER 结构及官方预训练权重，保留 RPGE、MMRegressor、TRR 训练损失与原 Matcher 求姿态流程。本分支从 research/rscore-l-local 的 f84b5f1 分出；核心 LEADER 文件保持原结构，实验新增在 research/image_gate。

多模态对照：投影修复后的①。DeDoDe/PCA 128D 图像特征，通过 encoder 输出 voxel 关联的原始 Cartesian 代表点投影与双线性采样，与 LEADER 512D 特征作门控残差融合。定位使用的 coarse voxel 坐标及其 GT 目标不变。

同一32帧本地开发集平均位置误差：

| 方法 | 误差m |
|---|---:|
| 原始LEADER | 0.1298 |
| ① 简单门控 | 0.1347 |
| ② 文档版门控、蒸馏及两阶段微调 | 0.1495 |
| ③ Node2Vec＋coarse/refinement上下文 | 0.1322 |
| ④ 三维监督场景上下文 | 0.1343 |

投影问题已修复：总覆盖率约22%，几何可见点的最终保留率约64%，分母不同。不要把 cylindrical voxel 坐标当 Cartesian XYZ，也不要为了增加比例调阈值。文档中原来的 coarse voxel 中心投影已经被真实表面代表点替代。

诊断结果：

- ①将全部图像有效点的平均场景坐标误差从1.473m降至1.244m；但原LEADER选中的高可靠点从0.295m恶化至0.301m，收益主要集中在未入选低可靠点。
- 门控未稳定区分有益和有害修正；训练集高可靠点改善，验证集恶化，存在泛化不足迹象。
- 固定权重只采用融合后的坐标，最终位置误差0.1342m；保留原坐标、只采用融合后的可靠度，为0.1288m。这是事后诊断干预，尚不是独立验证的新方法。
- ④明显减轻视觉深度/坐标错误，却没有改善最终定位。不能把中间指标当作最终方法收益。

请重点推理：如何利用视觉互补信息并保护准确的LiDAR对应；比较视觉调整可靠度、受约束坐标修正等方案，明确失败机制、可证伪预测、必要对照和最小验证实验。

约束：当前数据已接触，不是盲测或完整NCLT；②更新过回归头，其缺图结果不等于原始LEADER；重复编码与旧LiDAR缓存存在未解释差异，本次比较统一复用同一缓存，不能混用重编码结果；不预设图像无用，也不预设换融合策略就一定有效。

## 代码入口

- [`fusion.py`](fusion.py)：几何投影、图像采样、①门控。
- [`projection_audit.py`](projection_audit.py)、[`calibration_audit.py`](calibration_audit.py)：代表点映射及标定审计。
- [`run.py`](run.py)、[`retrain_raw.py`](retrain_raw.py)：原版接口、本地数据划分、①训练与评估。
- [`context_experiment.py`](context_experiment.py)：③、④上下文融合；④使用 `--scene-variant geometry`。
- [`lpgrf.py`](lpgrf.py)、[`lpgrf_experiment.py`](lpgrf_experiment.py)：②可微LoFTR末级、精确文档门控、蒸馏和两阶段训练。
- [`diagnose_fusion.py`](diagnose_fusion.py)：冻结权重逐点误差、门控、可靠度、常量/错配图像及姿态诊断。
- [`../rscore_l/losses.py`](../rscore_l/losses.py)：既有 coarse/final 三维监督。

## 大文件与复现布局

代码、JSON、报告和投影图直接在 Git 中；`.pt` 权重在 Release，不使用 Git LFS。附件包括原始LEADER权重、所有本轮训练权重、③④预训练场景模型、Node2Vec、PCA、检索输入、LoFTR/DeDoDe局部模型，以及本次96帧缓存和配对图像/扫描。它们足够复查本轮结果，未包含全部NCLT数据或重新从头训练视觉场景模型的907帧原始数据。

下载并验证：

```bash
python research/image_gate/download_handoff.py --output /path/to/handoff --extract
```

解压目录：

- `repo/`：放回本分支根目录的结果权重。
- `workspace/`：对应原工作区根目录中的 `glace-local`、`rscore-assets`、`research/image_gate_checkpoint`。
- `wsl/`：对应原 `/home/zhang/` 中的实验运行目录。

原代码使用原工作区路径和两个本地Python环境；其他机器复跑时需适配这些路径。不要把 `manifest.json` 内的本机绝对路径误认为GitHub可访问的路径。所有对应输入均可按附件的相对目录恢复。

缓存复用关系：`leader-image-gate-raw/lidar`、`leader-image-gate-context/lidar`、`leader-image-gate-geometry/lidar`、`leader-image-gate-lpgrf/lidar`指向`leader-image-gate/lidar`；`leader-image-gate-raw/visual`指向`leader-image-gate/visual_raw`。这些重复目录不另打包。

本轮训练/评估使用 PyTorch 2.0.1+cu118、MinkowskiEngine 0.5.4 环境；DeDoDe和场景头提取使用 rscore-l 的PyTorch 2.5.1环境，其冻结依赖表随附件提供。只做推理分析无需安装这些环境。
