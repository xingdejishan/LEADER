# REPORT — prevoxel_sixview_null（代码实现轮 v2：可训练性修复后，未训练）

run_id: `2026-09-18_prevoxel_sixview_null` ｜ Track A ｜ 预算: kill_test ｜ 决策: **keep（代码交付 + sanity 全 PASS）**

> v2 (2026-09-18)：按外部 review 的 4+1 条批评完成修复（原 v1 报告存档于 git/账本）。
> v1 的 Check 1/4/5 只验证了前向结构；v2 补齐了**梯度链**与**量化映射等价**两项正确性验证。

## 外部 review 的批评与修复（全部完成）

| # | 批评 | 修复 | 验证 |
|---|---|---|---|
| 1 | **梯度断裂**：ViewAdapter+Transformer 在 `@torch.no_grad()` 内，loss 只能到 W_V，"学哪个 view 可信"不成立 | 数据/模型分离：`dataset_hook.py` 只产固定观测（DeDoDe/投影/遮挡/质量均 no_grad，属冻结或非可导）；`fusion.py` 新增 `MultiViewLeaderForward`，Adapter/Transformer/NULL/voxel 聚合全部在 forward 内 | Check 6：proxy loss 反传后 Adapter、Weighting、NULL token 梯度均非零（PASS） |
| 2 | 黑边判断 `min(ch)>10` 误杀 [200,5,5] 类像素 | `visibility.black_border_mask` 改为 `max(ch)>thresh`（"全通道近零才算黑边"） | 代码级 + 24 帧 sanity 重跑 |
| 3 | invalid 视图未从 self-attention 移除，污染其余 token | `src_key_padding_mask` 在 attention 内屏蔽 + 输出 score −inf（双保险，α 恰为 0） | Check 4 扩展：attention 内屏蔽 + α=0（PASS） |
| 4 | DeDoDe 256D 与 config 128D 冲突 | 接入 PCA128，**严格复用原口径**：`F.conv2d(dense, pca.weight, pca.bias)`（= multicamera.py L93），权重 `WSL:/home/zhang/rscore-l-local/data/proc/pcad3LB_128.pth`（实测存在），config `pca_enabled=true, dim=128` | pca128() 单元路径 + config 一致性 |
| 5 | quantization 聚合语义未确认（mean vs first-hit） | **实测本机 ME 0.14 语义 = 代表点选择**：`fq == feats[index]` 逐位成立（65036 点真实帧，max err 0.0）、index 跨调用稳定、coords_q 行序 = 首次出现序。**既不是 mean 也不是固定 first-hit** | Check 7：`voxel_mapping` 返回 ME 自己的 index；`MultiViewLeaderForward` 用同一 index gather（`lidar_feats[index]` 逐位 == ME feats），v/r 同 index gather，gather 可导 |

**批评 5 的深层含义**（重要口径）：v1 的 `aggregate_to_voxel`（mean）与 ME 语义不一致——若保留，LiDAR voxel 特征与视觉 voxel 特征将来自不同点集，"point↔image 对齐"再次破坏。v2 改为 gather（`v[index]`）：每 voxel 的视觉特征 = **ME 选中的那个代表 raw point** 的视觉特征，与 LiDAR 特征严格同源。代价（已接受，训练轮需知晓）：代表点之外的点的视觉信息不进入该 voxel；这是与原 LEADER 聚合行为保持一致的必然选择。

## 交付结构（v2）

| 文件 | 职责 | 可训练参数 |
|---|---|---|
| `config.json` | 全部参数（含 pca_enabled/pca_weights）| — |
| `projector.py` | uint16 bin（lb3 系）→ 官方 ssc 链 → 6 相机投影 | 无 |
| `visibility.py` | 硬过滤（z>0/图内/黑边 max口径）+ z-buffer 遮挡 | 无 |
| `image_feature.py` | 冻结 DeDoDe-B + grid_sample | 无（冻结）|
| `quality.py` | 9 维质量向量（含 depth_evidence_valid / consistency_valid 分离 flag）| 无 |
| `dataset_hook.py` | **固定观测**：observe()（no_grad）+ voxel_mapping()（ME 原生 index/inverse）| 无 |
| `fusion.py` | **模型侧**：ViewAdapter + Transformer+NULL + index-gather 聚合 + `LEADERFirstLayerExtension`(W_V=0) + `MultiViewLeaderForward` | Adapter/Transformer/NULL/W_V |
| `sanity_check.py` | Check 1/4/5/6/7 + 覆盖统计 | — |

梯度链：`loss → W_V → feats_ext → v[index] gather → v_i → α(Transformer/NULL) → z(Adapter) → 冻结观测`。

## Sanity（24 帧，skip-dedode 结构口径 + 2 帧梯度口径，egonn118 env，GPU）

| 检查 | 结果 |
|---|---|
| Check 1 force-NULL：α_NULL=1 / α_views=0 / v=0（绕过 Transformer，逐位）| PASS |
| Check 4 invalid 视图：α=0 且 attention 内被屏蔽 | PASS |
| Check 5 无有效视图：α_NULL=1、v=0 | PASS |
| Check 6 梯度链：Adapter / Weighting / NULL token 梯度均非零 | PASS |
| Check 7 gather 路径逐位 == ME sparse_quantize 特征（含 index 一致性）| PASS |
| GT 隔离（无 pose/GT 输入路径）| PASS |

覆盖（24 帧，1,750,443 点）：0 视图 37.4% / 1 视图 47.1% / 2 视图 14.7% / 3 视图 0.8% / ≥4 视图 ≈0。
（v1 的 96 帧覆盖数字口径相同、趋势一致；v2 未重跑 96 帧全量，训练轮预注册前补跑即可。）

### 覆盖复核（96 帧，2026-09-18）

以 `coverage_hist_0_to_6_views` 为唯一分子来源，断言其和严格等于 raw-point 总数，且按
`1 - N_0 / N_total` 计算。此次仅跳过 DeDoDe 与质量向量计算；投影、FOV、黑边和 z-buffer
遮挡过滤均完整运行，因此不会影响 `valid`。

| 有效 Camera 数 | raw points | 比例 |
|---|---:|---:|
| 0 | 2,521,638 | 35.8083% |
| 1 | 3,415,178 | 48.4970% |
| 2 | 1,059,900 | 15.0510% |
| 3 | 45,326 | 0.6436% |
| 4–6 | 0 | 0.0000% |
| 合计 | 7,042,042 | 100.0000% |

因此 **raw-point visual coverage = 4,520,404 / 7,042,042 = 64.1917%**，不是 49.5%。
此前所述“96 帧约 348 万点”也不符合当前 96 帧的 raw bin 总数。

每个相机的单视图 raw-point coverage \(R_j=\#\{p_i:\mathrm{Cam}_j\ valid\}/N\) 与六视图并集的关系：

| Camera | valid raw points | \(R_j\) | 六视图相对此单相机增加 | 该 Camera 的独有贡献 |
|---|---:|---:|---:|---:|
| Cam0 | 117,407 | 1.6672% | +62.5244 pp | 0.3363 pp |
| Cam1 | 1,322,715 | 18.7831% | +45.4085 pp | 11.5589 pp |
| Cam2 | 898,959 | 12.7656% | +51.4261 pp | 8.5610 pp |
| Cam3 | 969,614 | 13.7689% | +50.4227 pp | 8.9153 pp |
| Cam4 | 1,426,097 | 20.2512% | +43.9405 pp | 12.3429 pp |
| Cam5 | 936,164 | 13.2939% | +50.8977 pp | 6.7826 pp |

故以覆盖最高的单相机 Cam4 为基线，六视图将 raw-point coverage 从 20.2512% 提升到 64.1917%，即 **+43.9405 个百分点（3.17×）**。独有贡献定义为“该 Camera valid、其余五台都无效”的点占比，不能相加，因为不同相机间存在重叠。

同一批点按 checkpoint 对应的原 LEADER 训练入口 `LEADER/run_mink.py` 实际默认值
`voxel_size=0.2`、`horizontal_res=1024`，并使用 ME 的 `index` 代表点选择量化后，共有
3,081,554 个 voxels。体素大小不再由多视图 `config.json` 独立维护。

| 代表 raw point 的有效 Camera 数 | voxels | 比例 | 模块含义 |
|---|---:|---:|---|
| 0 | 1,352,237 | 43.8817% | 纯 LiDAR，强制 NULL |
| 1 | 1,306,832 | 42.4082% | Camera vs NULL |
| 2 | 401,722 | 13.0363% | 多视图选择 / 互补 |
| 3 | 20,763 | 0.6738% | 多视图选择 / 互补 |
| 4–6 | 0 | 0.0000% | — |
| 合计 | 3,081,554 | 100.0000% | — |

因此当前模型实际输入中，**56.1183%**（1,729,317）个 voxels 的代表点至少有一个 Camera observation；真正进入至少两视图选择的为 **13.7101%**（422,485）个 voxels。

当前 `MultiViewLeaderForward` 使用第一种口径（`v[index]` gather），不是任一成员的聚合。作为独立诊断，若同 voxel 任一 raw point 有观测就算可用，则为 1,771,625 / 3,081,554 = 57.4913%；它不代表当前模型的实际输入。`W_V=0` 初始化时这些均为“可接收视觉”的覆盖，尚不代表 step 0 有非零视觉贡献。

## 与判负家族的关系（引用纪律，不变）

特征层融合 7 家族全负（0.1322–0.1495 vs 0.1298）是 voxel 特征级接口的证据；本模块为 raw-point 级 + NULL 退出的新变体，预注册已写明 novelty。**训练轮跑出三对照之前，不得引用本模块为任何形式的改进。**

## 未解决问题 / 下一轮边界

1. v1 报告中的覆盖数字（96 帧）由 v1 代码产生；黑边口径改动（min→max）会轻微提高覆盖率，96 帧重跑留到训练轮预注册时一并完成。
2. PCA128 路径已在 config 接通但未在 sanity 里端到端跑 DeDoDe（skip-dedode 口径）；训练轮 smoke 应含 2 帧 DeDoDe+PCA 全链路。
3. `LEADERFirstLayerExtension` 与 RPGE stem 的实际对接（checkpoint 加载 + step-0 数值等价验证）未做——这是训练轮第一项验证，不是本轮范围。
4. 训练轮准入：须另行预注册（≤40 min 量级、三对照含同预算 B1、置乱视觉、双基线、失败率+尾部同报）。


