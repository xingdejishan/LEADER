# 52 ms 前端口径核对（2026-10-24）

## 原始记录存在，但不是实时 LEADER + V7

原始六分片绑定文件：`/data2/zhangcy/v7-full-20261006/epoch10-evaluation-20261009/epoch10-evaluation-binding-sharded-20261009.json`。

- 104732 帧，5412 秒，单 GPU 3、六分片、每片四流，聚合 **51.6748 ms/帧**。
- 该历史运行使用 V7 epoch10；本项目使用原始 V7 epoch7，不能宣称检查点或难度分布完全相同。
- `infer_shard_predict_v5.py:123` 从 `loader.frame` 获取输入，再只执行 observation 和 V7 refine。
- `v7_runtime.py:102` 的 `TestLoader.frame` 在 `:105` 调用 `self.cache.load`，并要求 cached 非空，没有缓存便报错；不会实时调用 LEADER encoder/decoder/SC2。
- `cache_runtime_v2.py:9` 明确保存 `initial/source/target/points/pixels/visible_ids/structure`：包含 LEADER 初始位姿、对应与可见性几何，不只是 SAM。
- 当前 `v7_runtime.py` SHA 与历史科学计划绑定一致；科学计划本身 SHA 与 epoch10 绑定一致。推理脚本 SHA 也与六分片运行绑定一致。

因此之前把 52 ms 描述为“实时 LEADER+V7 全前端吞吐”不准确，应更正为“冻结上游输入的 V7 推理聚合吞吐”。SAM 缓存仅复用视觉特征，不会自动省掉实时 LEADER 和几何证据构建。

## 同输入复测

原始 `input-cache-v8` 已为空，不能直接逐文件复现历史全集缓存。此次仅在本项目 `.runtime` 重建了 **384 帧、33.1 MB** 的诊断缓存，不写原缓存目录，不接入生产。

从完整 104732 清单均匀选取 384 个唯一输入，覆盖四个日期；使用本项目相同 epoch7 权重、相同点云、RGB、SAM 和求解参数。GPU 1，三分片，每片四流，显式预热和阶段屏障：

| 路径 | 处理区间 | 聚合耗时 | FPS |
|---|---:|---:|---:|
| 实时 LEADER + SC2 + 几何输入构建 + V7 | 64.14 s | **167.03 ms/帧** | 5.99 |
| 缓存上述上游输入，只运行 V7 | 25.63 s | **66.76 ms/帧** | 14.98 |

两条路径在 384 帧最终位姿的最大平移差为 0。实时路径每帧累计工作量中的 initial 约 302.2 ms、可见性几何约 56.3 ms、frame 构建约 5.0 ms；缓存路径这三项调用为零。阶段工作量不是聚合输出间隔，不能混用。

此三分片短样本对照没有复现历史六分片的完整 52 ms 条件，不能据此保证实时六分片也能达到 52 ms。

## 单进程分项计时

另取上述输入中的 24 帧，单进程、单帧，预热两次，在阶段边界同步 CUDA：

| 部分 | 平均 |
|---|---:|
| LEADER encoder | 110.76 ms |
| LEADER decoder | 1.99 ms |
| SC2 estimator | 6.91 ms |
| initial 整体（含上述网络和 SC2） | 124.00 ms |
| 可见性几何构建 | 58.03 ms |
| frame 构建、搬运和摘要 | 3.39 ms |
| 入口 `leader` 计时字段 | 185.75 ms |
| V7（observation + refine） | 214.27 ms |

这里 initial 与 encoder/decoder/SC2 是包含关系，不能重复相加。该 24 帧分布与之前的 96 帧困难样本不同，也增加了同步计时，不能直接将 185.75 与之前 263.21 的差解释为加速。分项计时没有改变这 24 帧相对 384 帧回放的位姿，最大矩阵元素差为 0。

## 六分片与安全边界

六份实时前端和六份仅 V7 前端的两次单卡启动尝试，均在进入正式计时前触及 GPU 空闲显存 4 GiB 保护线，启动器仅停止自己创建的进程组，没有降低保护线或停止其他任务。六分片仅 V7 的工作进程完成预热，但未获准开始正式计时，故没有成功六分片新成绩。

每进程 PyTorch 显存上限 8%；这个限制不覆盖全部驱动、CUDA 模块及其他库的显存，不能仅按分配上限相加推断真实全卡占用。保留主机内存、磁盘与超时保护，全部使用现有 leader 环境，没有安装依赖。

本次只新增诊断脚本、缓存和报告，没有修改生产前端、默认选择模式或专家触发门控，没有读取 GT。几何和视觉专家仍按此前完整配置触发；是否按需调用需另做独立验证，不能用最终采纳率代替前置触发率。

## 产物

- `results/frontend_alignment_audit_20261024.json`
- `results/frontend-alignment-20261024-1792819510500170805/summary.json`
- `results/frontend_components24_20261024.json`
- `benchmark_frontend_alignment.py`
- `profile_frontend_components.py`

两次显存保护尝试目录分别为 `frontend-alignment-20261024-1792819228479035513`、`frontend-alignment-20261024-1792819687742036926`，不能作为成功测速引用。
