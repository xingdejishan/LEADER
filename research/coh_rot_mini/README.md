# L_coh-rot 最小机制验证（64 帧训练 / 32 帧固定验证 / 40 epoch）

本目录是"TRR + 帧级相干旋转损失"的**最小实现**与**机制筛选结果**，上传供实现审核。
实验为全本地（WSL `egnon118`，RTX 4060 Laptop 8 GB），未使用服务器。

## 文件

| 文件 | 作用 |
|---|---|
| `coh_rot.py` | **损失实现（审核重点）**：加权去中心 → Kabsch 前向（SVD，仅门控/日志）→ 线性化 ω 反向 → `√(1+z)−1`；退化/20°门/权重集中保护 |
| `selftest.py` | 14 项合成自检（纯平移、已知旋转、逐帧独立、有限差分梯度、共线/平面/门控/权重保护） |
| `experiment.py` | A/B 两臂训练 + 评估（64/32 帧、同初始化/同帧序/同优化器） |
| `pose_replay_min.py` | 从 `pose_replay.py` 提取的选点规则（top-k 可靠度） |
| `analyze.py` | 逐帧配对差 + block bootstrap 分析 |
| `PREREGISTRATION.md` | 方案预注册（含 C 臂两次修正记录） |
| `RESULTS_mini.md` | **结果与判定**（结论：未通过推进条件） |
| `results/` | 运行产物：`A/`、`B/` 各 9 个 `eval_*.json` + `training.json`、`run.log`、`subset.json`、`summary.json` |

## 如何运行

```bash
# 环境：WSL egnon118（torch 2.0.1+cu118 + MinkowskiEngine 0.5.4）
cd ~/coh-rot-scr
python selftest.py                 # 14/14 应全过
python experiment.py --epochs 0    # 冒烟（仅初始评估）
python experiment.py --epochs 40   # 完整机制筛选（本目录 results 的来源）
python analyze.py                  # 配对分析
```

依赖外部路径（未随本目录上传）：
- `LEADER` 仓库（`models/model_mink.py`、`models/sc2pcr.py`、`run_mink.py` 的 TRR 类）
- 905 帧缓存 `/home/zhang/crossframe-visual-probe/lidar/*.npz`（`features(512)/source/GT`）
- 划分 `/home/zhang/anchored-contrastive-fusion/protocol.json`（578/145/182）
- 冻结 checkpoint `research/image_gate_checkpoint/{model.safetensors, extra.json}`

## 实现要点（请重点审核这些决策）

1. **加权去中心**：`X_i = ĉ_i − Σw_j ĉ_j`，`Y_i = c_i − Σw_j c_j`，权重先归一化到 Σw=1
   （整帧平移不触发旋转损失）。
2. **Kabsch 方向**：行向量约定，求 `Q` 使 `X·Q ≈ Y`（`H = (X·w)ᵀY`，`Q = V·diag(1,1,det)·Uᵀ`）。
3. **反向梯度**：**不**对 SVD 奇异向量求导。梯度走一阶线性化旋转
   `ω = A⁻¹b`，`A = Σw(‖x‖²I − xxᵀ)`，`b = Σw(x×y)`（对 X≈Y 时与 Kabsch 一阶一致，已用有限差分验证，误差 4.4e-10）。
4. **损失值**：`z = ‖ω‖²/θ₀²`，`L = √(1+z)−1`，`θ₀ = 1°`。小角度下 ‖Q−I‖_F² ≈ 2‖ω‖²，故与方案的 `z = ‖Q−I‖_F²/(2θ₀²)` 一阶等价。
5. **保护**：点数 <16；秩 <2 或奇异值近重复（`S[1] ≤ 1e-7·S[0]` 或相邻奇异值差 ≤ `1e-7·S[0]`）；权重集中 `(Σw²)·n > 5`；非有限坐标；Kabsch 角 > 20°。全部**跳过新项、保留 TRR、逐帧计数**；跳过的帧照常进入评估。
6. **权重固定**：由**冻结模型**对每帧输出可靠度经 `exp(atan(u)·log(10)/π)` 转换（未归一化存储，损失内归一化），训练全程不变，网络无法重整权重绕开需修正的点。
7. **逐帧计算**：`batch_idx` 逐帧切分，绝不把多帧合成一个点集。

## 与方案的差异点（诚实列出，请判断是否接受）

| # | 方案原文 | 本实现 | 影响 |
|---|---|---|---|
| 1 | `z = ‖Q_e−I‖²/(2θ₀²)`，Q_e 由 SVD 精确求解 | `z = ‖ω‖²/θ₀²`，ω 为线性化解（**值与梯度同源**） | 小角度等价；角度接近 20° 门附近会低估。若要求值用精确 SVD，可改"值 SVD / 梯度 ω"混合模式 |
| 2 | — | 前向 SVD 每帧跑一次（`no_grad`），仅用于门控与日志 | 额外开销小（点数 ~1000） |
| 3 | 权重"转换结果" | 采用 `exp(atan(u)·scaler)`，与 TRR 的 `u_scale` 分支一致（不含 `u_cut` 的 clamp 分支） | 对固定权重两者只差截断，无实质影响 |

## 结果摘要（详见 `RESULTS_mini.md`）

| epoch 40 | A (TRR) | B (TRR+λL_coh-rot) | 变化 |
|---|---:|---:|---:|
| MPE | 0.1210 m | 0.1160 m | −4.15%（CI 含零） |
| MOE | 1.3162° | 1.2575° | −4.46%（epoch 20/30 CI 不含零，**epoch 40 含零**） |
| Re（Kabsch 转偏角） | 1.5074° | 1.5045° | **−0.19%（CI 全部含零）** |

- 损失在**训练帧**上确实被优化：`rot` 0.520→0.440（−15%）、SVD 角度 1.165°→1.020°（−12%），全程无跳过。
- **未泛化到验证帧**：主靶点 Re 不动；A 臂自身 MOE **恶化 +4.93%**，B 臂 +0.25%（更像"抑制恶化"）。
- 判定：**不满足推进条件，不支持进入 578/145/182 完整训练。**
- 已知缺口：训练权重未保存（无法补算验证集 `|ω|`）；单 seed；λ=0.1 时该项仅占总损失 ~9%。
