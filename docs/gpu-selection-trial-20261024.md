# GPU 共享几何评分实验（2026-10-24）

## 入口与范围

`--selection-mode gpu_sparse` 为实验入口，默认仍为 `full`。

```bash
cd /home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1
/home/zhangcy/.conda/envs/leader/bin/python -B e2e_online.py \
  --keys-file results/sparse_validation_keys_20261024.json \
  --selection-mode gpu_sparse --gpu 3 \
  --out results/gpu_sparse_another_run.jsonl
```

这不是全集启动命令。保留同一候选池至多 8 个不同位姿，不按 PnP qualified 或 Top2 丢弃候选。所有候选和几何位姿使用同一份独立证据，排除所有候选生成参考的并集；参考采样、1000 点查询、1 m 截断代价、12 次扰动和 k=8.4 均不变。它与 CPU sparse 对齐，不是与完整 full 选法等价。

## 实现

`gpu_selection.py` 使用现有虚拟环境的 PyTorch、NVRTC 和 CUDA 驱动，在内存编译 CUDA 核，不安装软件或训练参数。对稀疏 1 m 单元构建有序键，查询 27 个相邻单元内所有点，计算双精度截断最近邻距离；不是占用格近似。索引排序、几何距离计算在 GPU，参考解码和裁剪仍在 CPU。裁剪优化保持原地图点集合完全不变，96 帧原 CPU 输出与代价均逐元素一致。

地图最多 405000 点、查询最多 2000 点、位姿含校准最多 21 个、单元最多 4096 点；异常输入明确报错，不静默改变代价。不构建稠密空间网格或完整查询×地图距离矩阵。保留入口的单进程 15% 显存上限、主机资源检查、写入目录限制；CUDA 编译缓存限定在项目 `.runtime/cuda-cache`。模型初始化、CUDA 编译和显式小样本预热不计入选择模块稳态耗时。

## 配对回放：相同证据、无模型、无 GT

| 样本 | CPU 平均 / P95 | GPU 平均 / P95 | GPU 最大 | 超过 100 ms |
|---|---:|---:|---:|---:|
| 48 帧验证池 | 96.72 / 169.44 ms | 41.90 / 80.10 ms | 96.58 ms | 0/48 |
| 96 帧回归池 | 133.08 / 235.01 ms | 56.87 / 97.19 ms | 127.35 ms | 3/96 |

排除无候选回退后，48 帧中 45 次有效决策的 GPU 平均 44.68 ms、P95 80.51 ms；96 帧中 92 次有效决策平均 59.33 ms、P95 97.64 ms。144 帧 CPU/GPU 位姿和选择均一致，最大代价差 3.33e-16。不能将这理解为对全集的数学等价证明。

96 帧配对实验的 PyTorch 峰值分配 27595776 字节（26.32 MiB）；这不含 CUDA 上下文、模块和驱动自身开销，也不是整个模型的显存。

## 从输入重新运行的完整 GPU 链路

48 帧均重新读取扫描、使用已有 SAM 缓存并实时执行 LEADER/V7、几何与视觉候选生成，没有冻结上游。

- 选择模块全 48 帧：平均 41.82 ms，P95 71.37 ms，最大 107.96 ms，1 帧超过 100 ms。
- 45 次有效决策：平均 44.58 ms，P95 71.52 ms；3 帧没有视觉候选，保留几何结果并标记 degraded。
- 最终位姿、选择与先前 CPU sparse 的同 48 帧全部一致。
- 各阶段工作耗时之和平均 1.384 s/帧；不是单帧端到端延迟或六分片吞吐，不能宣称整体达到 100 ms。

这些样本复用先前已固定的验证/回归池，用于等价性与加速核对，不是新的盲测精度集合。此次没有读取 GT，也没有据真值调参。CPU sparse 先前相对 full 存在选择差异，仍须保留该限制。

## 结论与产物

选择模块在本次两组回放及完整链路样本上达到 P95 < 100 ms，但不保证每帧小于 100 ms，也不代表完整算法小于 100 ms。保持实验开关，不替换默认 full。37 项单元测试通过，含 GPU 随机点、负坐标、跨 400 m 地图、旋转、单元边界、截断及密度上限测试。

- `results/gpu_selection_paired48_gpusort_20261024.json`
- `results/gpu_selection_paired96_20261024.json`
- `results/gpu_sparse_live48_20261024.jsonl`
- `results/gpu_sparse_live48_summary_20261024.json`
- `benchmark_gpu_selection.py`

早期诊断文件 `paired48`、`paired48_warm`、`paired48_crop` 为优化过程留档，不应混同最终 GPU 排序版。
