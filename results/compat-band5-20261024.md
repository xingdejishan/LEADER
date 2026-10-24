# 两处兼容性修复（2026-10-24）

对象：`hypothesis-arbitration` 分支，基线提交 `1ef4c7e`。只有一处代码改动、一处记录，未触碰
`e2e-moe-20261020-v1`，未新增依赖，未调任何阈值。

## 一、"SaladBranch.hypothesize 签名不兼容"：分支内没有这个对象

核实过程与结论：

- `git log --all -S SaladBranch` 在本仓库全历史只命中 `1ef4c7e`，命中的是 `results/` 下两份说明文档
  的字样，不是代码。分支内从未有过 `SaladBranch` 类；视觉专家只有一个类，即 `visual_branch.VisualBranch`。
- 提交 `b3e65d6`（SALAD 单通道）已经把位置先验从视觉链路上删干净：`retrieve(self, image_path, *extra)`
  与 `hypothesize(self, key, image_path, budget=8, scan=None, scan_path=None)`，多传位姿会按 4×4 矩阵、
  3 维位置、3×3 旋转分别报错。
- 真正仍向 `super()` 回传位姿的 `SaladBranch` 在**分支之外**：
  `/home/zhangcy/dsh-work/vpr/code2/salad_branch.py`；同目录 `probe_branch.py` 的注释原文是
  "minus its call into a super() that no longer takes `pose_prior`"。按"只写该分支"的约束，该文件未动。

所以分支内没有需要修改的 `SaladBranch`。要求里的约束"子类签名与父类一致、内部不再回传位姿"落实在本次
还原的 `band5_nopose.py` 上：它的字节码里 `hypothesize` 恰好就是那种"断言为空再原样回传"的旧写法，
照抄必然 `TypeError`。还原时删掉该覆写（继承父类签名即天然一致），`retrieve` 与父类同为
`(image_path, *extra)` 并拒绝位姿参数。

grep 证据（排除 `.backups/`）：

```
视觉链路 7 个文件，pose_prior 命中数全为 0：
  visual_branch.py  visual_ordered.py  visual_accelerated.py  light_selection.py
  visual_live.py  salad_retrieval.py  band5_nopose.py
全树 *.py 只剩 tests/ 里 3 处用例，它们断言的是"该参数不存在"（test_salad_retrieval.py 2 处、
test_light_selection.py 1 处）。
```

`mw_cls.py`、`mw_*.py` 等研究脚本里的 CLS 与 position 用法与取参考无关，按要求未动。

## 二、`band5_nopose.py` 从字节码还原

来源：`__pycache__/band5_nopose.cpython-38.pyc`（5968 字节，2026-10-23 12:01），仓库内唯一副本。
用 `marshal` 读回代码对象后，按全量 `dis` 输出与常量表逐项重建：模块 docstring、函数与类的顺序、
行号、默认值（`build` 仍为 256 / 0.4 / False）、全部错误串、变体表（A5 / B20 / C5）、`CaptureMixin`
的五个捕获点（`hypotheses`、`accepted`、`pnp_calls`、`capped`、`verify`、`retrieval`）都照原样。
文件头注明"由字节码还原，仅用于复现历史实验"。

两处无法逐字保持，均因 `b3e65d6`：

1. 位姿管道（见上一节）。
2. **A5 与 B20 已不可执行**：`retrieve_cls_only` 读 `self.reference_cls` 与 `self.embed`，而这两个成员
   在 `b3e65d6` 里随 CLS 通道一起删除。检索规则原样保留（可读、可对旧的 `visual_branch.py` 复用），
   但在当前父类上这两个变体取参考时会 `AttributeError`。C5（部署版）与捕获框架正常。
   `mw_cls.py` 里那份自带 CLS 嵌入器当时核对过前二十名与旧路径一致，可作为需要重跑 A5/B20 时的入口。

盘面副作用，如实记录：原始 pyc 在首次 `import band5_nopose` 时被 Python 用新源码重写
（5968 → 7311 字节）。还原依据是重写前的完整 `dis` 与常量表，源码已入库，分析信息没有损失。

## 三、验证

1. `CUDA_VISIBLE_DEVICES="" python -c "import visual_branch; import band5_nopose"` → 通过；
   仓库内 22 个 CPU 测试（`test_salad_retrieval`、`test_no_rotation`、`test_light_selection`）全过。
   签名实测：`VisualBranch`、`OrderedBranch`、`NoPoseVisualBranch`、`CapturedDeployed` 的
   `hypothesize` 全为 `(self, key, image_path, budget=8, scan=None, scan_path=None)`。
2. 冒烟 `e2e_online.py --gpu 3 --limit 5`，默认参数：5/5 帧输出有限 4×4 位姿，退出码 0，无异常。
   这 5 帧视觉专家返回空（`visual_no_hypothesis`，仲裁回落几何路线，状态 degraded），与当日
   20 帧冒烟里"14 帧有视觉假设、6 帧为空"的记录一致，不是崩溃。
3. 290 帧难题子集、纯 SALAD 强制采用（口径取自本仓库 `band5_report.py`）：4 分片各 8.6–9.5 分钟。

| 指标 | 本次 | 既有记录 | 差 |
|---|---|---|---|
| 检索命中（参考表内有 ≤5 m 真值） | 210 | 210 | 0 |
| 过 PnP 门 | 258 | 258 | 0 |
| 达一致（≥2 参考 5 m/10°） | 229 | 229 | 0 |
| 最终落进 1 米 | 20 | 20 | 0 |
| 最终落进 5 米 | 31 | 31 | 0 |
| 净省距离（米） | 1325.1401929939 | 1325.1401950438 | −2.05e−06 |
| 最终总误差（米） | 2483.7748990061 | 2483.7748993491 | −3.43e−07 |
| MPE 均值 / 中位 | 8.5647 / 8.8271 | 8.5647 / 8.8271 | 0 |
| MOE 均值 / 中位 | 2.3538 / 2.2271 | 2.3538 / 2.2271 | 0 |

所有离散计数逐项相同，两个连续量差 2 微米量级（相对 1.5e−9），属末位浮点与序列化舍入，不是参数差异；
没有为凑数改动任何设置。每帧参考数 5.00，与单通道定稿一致。

复现器说明：当年的 `band5_run.py` 同样只剩记录、没有源码，本次的 runner 与记分器放在 `/tmp`（未入库），
定义与阈值全部照抄已提交的 `band5_report.py`（准入、一致、强制采用、hit5 四个口径一致）。

## 四、限制

- A5 / B20 在当前父类上不能跑（CLS 通道已删），只有 C5 与捕获框架有效。
- 第 3 项是"强制采用"口径，不是仲裁后的端到端结果。
- 分支外那棵 `dsh-work/vpr/code2/salad_branch.py` 仍然是旧签名，需要单独处置，不在本分支范围内。
