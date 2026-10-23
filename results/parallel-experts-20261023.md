# 并列式控制流：级联 → 两个专家每帧都出假设（2026-10-23，GPU 实测）

一句话：**只把"串联"改成"并列"，门开帧上两个结构的输出逐位相同（0/84 帧变化），
差异全部落在 1418 个非门开帧上，而且是变好**——全量 MPE 0.8585 → 0.8493（冻结表）
或 0.8575 → 0.8463（实时视觉专家），多采纳 13 / 15 帧，门开帧的救援量一格没动。

---

## 1. 改了什么、在哪（`three_route_unified.py`，472 行，独占 inode，未碰任何其他文件）

| 位置 | 行 | 内容 |
|---|---|---|
| 控制流本体 | 328–344 | 原来的 `if best is not None and np.isfinite(ratio) and ratio < AMB_THRESHOLD:` 一句话，拆成 `gated`（照常算 ρ）、`asked`（cascade 取 `gated`，parallel 恒为真，`none` 恒为假）、按 `--visual-source` 取假设或调用实时专家 |
| 诊断字段 | 329–339 | `gated`/`asked`/`experts`/`visual_source` 只写进诊断，不参与任何判断 |
| 退化守卫 | 348–353 | 视觉假设落在"30 m 内没有任何参考云"的地方时丢弃（冻结证据装配在那里会 `concatenate([])` 崩溃，见第 6 节），记 `visual_dropped` 与距离 |
| 实时专家 | 137–215 | 新增 `build_expert()` / `live_hypothesis()`：调用与在线驱动**同一行**——`branch.hypothesize(key, image, budget=8, pose_prior=T0)`，常数 8 即在线驱动的 `--visual-budget` 默认值 |
| 冰结表分支 | 97–99 | `--visual-source live/none` 时不再加载 1502 张冻结表 |
| CLI | 408–423、431 | `--experts {cascade,parallel}`、`--visual-source {frozen,live,none}`、`--visual-arm`、`--visual-budget`、`--scratch`、`--gpu-fraction`、`--shard`、`--stride` |
| 分条 | 445、453 | 按 `--shard i/n` 取子集、按键名取初始位姿（默认行为不变） |

**一行没动的**：几何支路（V7 + 沿轨/横向搜索 + 旋转块）、`arbitration.py`（k·σ_C、并集
证据）、`visual_branch.py`（CLS top5 ∪ 位置 5 近邻、3.5 px 绑定、三种 PnP 通道、≥2 参考
5 m/10° 一致、fair 分取最高）、`stages/`、`along_track_dp.py`。校验：`visual_branch.py`
仍是 inode 968858 / 2 links、`arbitration.py` 仍是 2026-10-22 23:00 的 mtime。

## 2. 三项自检

| # | 判据 | 结果 |
|---|---|---|
| 1 | 关掉视觉 ≡ 只走路线三（全量逐位） | **通过**：1502/1502 帧位姿逐位相同，`max abs diff = 0.0`，指标同为 1.5056 / 2.4839 / 746 |
| 2 | 门开帧上参考列表、候选解、fair 分 ≡ 级联冻结表（≥20 帧） | **不通过**：84/84 帧上，参考列表逐位相同 50，fair 分逐位相同 51（原因见下） |
| 3 | 门开帧上视觉假设位姿 ≡ 级联候选（同上抽样） | **不通过**：位姿逐位相同 48 / 84（<1e-9 为 49 / 84） |

自检 2/3 不通过的原因是可测的，且**与控制流无关**（换成级联也一样）：

1. **冻结表的位置通道用了另一个先验**。CLS top-5 在 **84/84** 帧都是冻结列表的子集，
   但位置 5 近邻只有 55/84 是子集——把先验换成表里的 `baseline_V7` 也只能复现 1/34 帧，
   说明生产该表时喂的是第三个位姿（模块注释里说的 SC2 初值），不是部署 T0。本版按题目
   要求用 T0，因此 34/84 帧的参考列表本来就会差几个号。
2. **冻结表给每个候选都打了激光分，实时路径只打前 8 个**（在线驱动 `--visual-budget 8`）。
   表里最多 12 个候选，best-by-fair 可能排在第 9 名以后（第 1428 帧就是 `candidate_10`，
   fair=1.0）——实时路径拿不到它。
3. **余下的差异来自"重建"本身**：在参考列表完全一致的 50 帧上，把 budget 提到 16 只把
   fair 分逐位相同从 38 修到 41、位姿从 36 修到 39；剩下 9 帧实时专家甚至给出了比表里
   最高分*更高*的解（723 帧 0.850515 对 0.847458），说明候选集合本身不同。

**与控制流真正相关的等价性检验是通过的**：冻结表下 cascade 与 parallel 在 84 个门开帧上
位姿 0 变化；实时专家下 cascade 与 parallel 在 84 个门开帧上同样 0 变化。级联单独换用
实时专家时（84 次调用）采纳 16 → 17 帧，这是专家来源的差别，两条结构里都发生。

## 3. 全量 1502 帧

评分器：`r-unified-eval-20261018-v1/score_unified.py` 的判据（hard1502，GT 用
`scored.npz['reference']`）。路线三 1.5056 / 2.4839 / 746 / >5 m 19 / >10 m 7 与既有数逐位一致。

| 配置 | MPE | MOE | 进一米两度 | >5 m | >10 m | 采纳视觉 | 视觉调用 |
|---|---|---|---|---|---|---|---|
| 只走路线三 | 1.5056 | 2.4839 | 746 | 19 | 7 | 0 | 0 |
| 级联 + 冻结表（当前） | 0.8585 | 2.1674 | 754 | 8 | 3 | 16 | 84 |
| **并列 + 冻结表** | **0.8493** | **2.1478** | **758** | **7** | 3 | **29** | 1502 |
| 级联 + 实时专家 | 0.8575 | 2.1613 | 752 | 8 | 3 | 17 | 84 |
| **并列 + 实时专家** | **0.8463** | **2.1299** | **760** | **7** | 3 | **32** | 1502 |

## 4. 门开 84 帧（救援量）

| 配置 | MPE | MOE | 进一米两度 | >5 m | >10 m | 采纳 | 救回（均值 / 合计） |
|---|---|---|---|---|---|---|---|
| 只走路线三 | 17.7127 | 10.0353 | 27 | 14 | 7 | 0 | — |
| 级联 + 冻结表 | 6.1412 | 4.3756 | 35 | 3 | 3 | 16 | 11.5715 m / 972.01 m |
| 并列 + 冻结表 | **6.1412** | **4.3756** | **35** | **3** | **3** | **16** | **11.5715 m / 972.01 m** |
| 级联 + 实时专家 | 6.1233 | 4.2663 | 33 | 3 | 3 | 17 | 11.5894 m / 973.51 m |
| 并列 + 实时专家 | **6.1233** | **4.2663** | **33** | **3** | **3** | **17** | **11.5894 m / 973.51 m** |

级联的 11.57 m 按此口径复现为 11.5715 m。并列在门开帧上一位小数都不差。

## 5. 非门开 1418 帧（并列新增的行为）

| 配置 | MPE | MOE | 进一米两度 | >5 m | >10 m | 采纳 |
|---|---|---|---|---|---|---|
| 只走路线三 / 级联（冻结、实时都一样） | 0.5456 | 2.0366 | 719 | 5 | 0 | 0 |
| 并列 + 冻结表 | 0.5359 | 2.0158 | 723 | 4 | 0 | 13 |
| 并列 + 实时专家 | 0.5336 | 2.0033 | 727 | 4 | 0 | 15 |

15 个位姿变化的帧全部在门外（清单见 `results/live-parallel-full_diag.jsonl` 中
`visual=true` 且 `gated=false` 的行），门外多采纳的每一帧 dcost 都是负的（−0.017…−0.20 m），
越过了该帧的 k·σ_C。

## 6. GPU 延迟与墙钟

| 项目 | 实测 |
|---|---|
| 视觉专家单帧端到端（单进程独占 3 号卡，n=15） | 中位 **2.68 s**，p90 **6.21 s**（门开帧中位 2.39，非门开中位 4.24） |
| 同上，18 进程 / 3 张卡的实际工作点（n=1502） | 中位 **3.18 s**，p90 **10.79 s**，均值 4.84 s |
| 单帧成分（单进程中位） | 检索 0.05 s、匹配 0.34 s、PnP 0.42 s、**激光验证 1.92 s** |
| 全量 1502 帧实际墙钟 | **457 s**（18 进程，卡 3/4/5，无 nice；`--workers 6` × 3 条） |
| 12 进程对照 | 501 帧 244 s ⇒ 全量约 732 s |
| 单张卡串行跑 | 视觉 7274.6 s + 几何 1502×0.187 s = **7556 s ≈ 2.10 h** |
| 显存 | 单进程峰值 553 MiB；18 进程每卡峰值约 14–18 GB（未碰 0/1/2/6/7） |

专家是 CPU 受限不是 GPU 受限：GPU 利用率 0–19%，`verify`（25 m 并集云 + KD 树 + 分箱）
占单帧 70%。

## 7. 判定

**属于"差异只在非门开帧"，而且方向是变好。** 门开 84 帧：并列与级联逐位等价（位姿 0/84
变化、采纳数相同、救援量相同）；非门开 1418 帧：并列多采纳 13（冻结）/ 15（实时）帧，
MPE 0.5456 → 0.5359 / 0.5336。全量因此 0.8585 → 0.8493、0.8575 → 0.8463。

## 8. 偏离与崩溃（附原文）

1. **崩溃（第一轮，12 进程）**：`ValueError: need at least one array to concatenate`，
   位置 `three_route_unified.py:357  mp_v = maps.local(cv_centre, maps.nearby(cv_centre)[:cfg.reference_count])`
   → `fast_lib.py:100  voxel(np.concatenate([self._ref_cloud(j) for j in triple]), 0.3)`。
   `nearby()` 是 30 m 球，视觉假设落在球外时三元组为空。级联从不问门外帧所以从未遇到。
   处理：在进入仲裁前丢弃该假设（第 348–353 行），全量 1502 帧只触发 **1 帧**
   （`2012-02-19/1329674911535332`，假设离 T0 2870.1 m）。仲裁代码未改。
2. **崩溃（单进程压测第 16 帧）**：`TypeError: popitem() takes no keyword arguments`
   （`visual_branch.py:219 _cached_features`）——它的两个缓存是普通 dict 却用
   `popitem(last=False)` 淘汰，超过 64 条必崩。处理：改用分支里已有的
   `visual_expert.OrderedBranch`（只把这两个容器换成 OrderedDict，其余不动）。这是
   **唯一一处对既有实现形式的改动**，不改任何规则、常数或求解器。
3. **未改动但需要记住**：冻结表的位置通道先验不是 T0（第 2 节原因 1），所以"实时专家 ≡
   冻结表"达不到逐位，全量差 0.0010–0.0011 米（0.8585→0.8575）。这不是并列带来的。

## 9. 复现

```bash
cd /home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1
# 并列 + 实时专家（3 张卡各 6 工人，实测 457 s）
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((3+i)) python three_route_unified.py \
  --experts parallel --visual-source live --workers 6 --shard $i/3 \
  --out results/live-parallel-full-s$i.npz --scratch /tmp/vb-full-s$i & done
# 级联对照 / 冻结表对照 / 关视觉对照
python three_route_unified.py --experts cascade  --visual-source live   --out results/live-cascade.npz
python three_route_unified.py --experts cascade  --visual-source frozen --out results/par-cascade-frozen.npz
python three_route_unified.py --experts parallel --visual-source none   --out results/par-novisual.npz
```
产物：`results/live-parallel-full.npz`、`live-cascade.npz`、`par-parallel-frozen.npz`、
`par-cascade-frozen.npz`、`par-novisual.npz`、`par-route3.npz`、`arm-metrics.json` 与各自
`_diag.jsonl`。
