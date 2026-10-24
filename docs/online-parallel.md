# 实时并列专家入口

入口为 `e2e_online.py`，必须使用 `/home/zhangcy/.conda/envs/leader/bin/python`。

## 输入与输出

输入是 NCLT LiDAR 点云、同步 Cam5 图像、标定 `T_BC` 和已建参考地图。输出 `T` 为 4×4 `world_from_body`，满足 `p_world = R @ p_body + t`，平移单位为米。

这是视觉/激光多模态算法，**不是仅凭点云进行无图像定位**。自定义传感器仍需匹配 NCLT Cam5 的内参、图像尺寸 616×808 以及坐标约定；不是通用传感器适配器。

完整路径：

```
点云 + RGB + 标定
  → 当前图像已预处理的 SAM ViT-L 特征（校验后读取缓存，不重算）
  → 当前点云的 LEADER/SC2 初始定位
  → 当前帧的 V7 appearance-field refine → T0
  ├→ 几何平移搜索 → T_geometry（保持 T0 旋转）
  └→ 当前图像检索 + SuperPoint/LightGlue + PnP + 点云验证 → T_visual
  → 两个候选局部地图的并集作为同一份 Evidence
  → C_geometry − C_visual > 8.4 × sigma_C 时采用视觉位姿
  → 最终 T
```

两路专家每帧均执行，不由歧义门控决定是否调用视觉专家。“并列”是候选生成/决策结构；当前入口默认每进程 4 个 V7 模型副本/4 条 CUDA 流，CPU 预取下一批输入，帧内的几何、视觉和最终仲裁依次执行。

不读取 `inputs/initial.npz`、冻结 query 视觉候选表或 query 位姿/匹配输入缓存。SAM 图像特征是允许复用的预处理数据，默认读取 `--sam-cache-root/<date>/cache_manifest.json` 及对应特征文件，校验图像、清单和特征摘要；缺失时明确报错，不回退 CPU 重算。LEADER/V7 和视觉模型仍在 GPU 上实时计算，几何搜索和仲裁使用 CPU。

新图像没有预处理缓存时可显式指定 `--sam-source live --sam-device cuda`，仍受同一显存限制，不支持自动 CPU 回退。

## 使用

先进入本项目目录。只检查全集输入路径，不加载模型或 CUDA：

```bash
/home/zhangcy/.conda/envs/leader/bin/python -B e2e_online.py --check-inputs
```

从公开测试清单取单帧，自动查找同步点云、图像和标定，并保留 canonical index 随机种子：

```bash
/home/zhangcy/.conda/envs/leader/bin/python -B e2e_online.py \
  --key 2012-02-12/1329068220036076 --gpu 3 \
  --out results/parallel_one.jsonl
```

显式指定原始传感器输入，不需要该帧存在于测试清单中：

```bash
/home/zhangcy/.conda/envs/leader/bin/python -B e2e_online.py \
  --scan /absolute/path/scan.bin \
  --image /absolute/path/cam5.jpg \
  --key 2012-02-12/1329068220036076 \
  --calibration /absolute/path/calibration.json \
  --gpu 3 --out results/parallel_custom.jsonl
```

`calibration.json` 为 `{"T_BC": [[...], [...], [...], [...]]}`。`.bin` 为 NCLT 8 字节点格式 `(uint16 x,y,z,uint8 intensity,ring)`，不是 float32 KITTI `.bin`。也支持米制 `N×4` XYZI `.npy`。自定义输入默认 `--canonical-index 0`；若需复现公开清单某帧，使用 `--key` 查询方式，或显式传入其 canonical index。

默认枚举服务器四个公开测试日期的全部清单，不限制为 hard1502。日期为 2012-02-12、2012-02-19、2012-03-31、2012-05-26；这不代表 NCLT 所有采集日期。支持 `--dates`、`--keys-file`、`--limit`、`--stride`、`--shard i/n`。未进行全集实际运行认证。

```bash
/home/zhangcy/.conda/envs/leader/bin/python -B e2e_online.py \
  --gpu 3 --limit 10 --out results/parallel_smoke.jsonl
```

全量运行前应先核对参考地图覆盖和资源；不是通过小样本就保证全量精度。默认参考库仍是 `inputs/refs.json`，支持 `--refs /absolute/path/refs.json` 替换为更完整的训练日期参考库，格式与现有文件一致，不能使用测试日期的位姿构图。

## 结果与恢复

每帧即时写一条 JSONL，并输出最终矩阵到标准输出。包含 `T`、`T0`、`T_geometry`、`T_visual`、两路状态、仲裁代价、sigma、阈值及各阶段耗时。无有效视觉候选或地图覆盖不足时，保留几何/T0，状态为 `degraded` 并写明原因；不会伪造视觉输出或零矩阵。

默认遇到异常停止，失败批次记录 `status=error,T=null`；错误帧不会在恢复时被当作已完成，不将占位矩阵当作有效结果。

相同命令加 `--resume` 恢复。恢复会校验运行参数、关键代码和参考库/权重指纹、已完成输入的路径/尺寸/修改时间；参数改变必须换输出文件。源输入按 manifest SHA-256 核验。自定义路径不带 manifest hash，由用户保证标定和同步正确。

## 资源边界

- 单次入口为一个进程，默认四帧一批，另预取一批；BLAS/KD 默认一个 CPU 线程，V7 的四条流使用已有 `ParallelFrames` 执行器。
- 默认 GPU 3，仅作为启动选择，不是独占声明；启动前检查显存，不改动其他进程。
- PyTorch 显存分配上限 15%，不支持在此入口提高上限。
- SAM 默认复用预处理特征，不加载 SAM 编码器；只有显式 `--sam-source live` 才在 GPU 上重新计算，仍可能因显存上限 OOM。
- RSS 看门狗默认 12 GiB，系统可用内存低于 32 GiB 时停止；看门狗是采样保护，不是操作系统级硬内存隔离。
- 使用 `statvfs` 检查真实磁盘空闲，低于 1 GiB 时停止，不依赖环境现有的磁盘空间报告 shim。
- 几何参考缓存最多 12 份点云、4 份合并地图；输出逐帧落盘，不累计全部结果。
- 仅允许写本项目 `results/` 与 `.runtime/`，不安装依赖、不改动其他项目或环境、不删除其他文件、不在线下载权重。

已有的 `three_route_unified.py` 仍是需要现成 T0 的离线/消融入口，不应把它称为完整原始输入端到端入口；其默认现改为 `parallel + live`，冻结视觉表仅在显式指定 `--visual-source frozen` 时使用。

## 历史实时重算 SAM 验证（2026-10-23，非当前默认配置）

- 21 项单元/回归测试通过，包括几何旋转保持、并列调用、共享地图仲裁、输入解析、清单分片和断点恢复。
- 四个测试日期共 104,732 帧的点云与图像路径检查通过，无缺失；这只是文件存在性检查，不是逐帧推理认证。
- `2012-02-12/1329068220036076` 分别通过清单输入和显式点云/图像/标定输入，均完成实时 SAM、LEADER/V7 和实时视觉候选生成，最终 T 逐元素一致。
- 最终共享地图仲裁代码另行通过该真实帧回归；视觉改善约 0.08287 m，阈值约 0.09284 m，因此保留几何专家。这说明视觉已实际参与，不代表必须选择视觉。
- 这次历史 CPU SAM 试跑约 79–86 秒，其中 SAM 重算约 64–69 秒；该配置已纠正，不能把此耗时作为当前默认缓存 SAM + GPU 推理的速度。尚未进行全集推理或精度评估。

真实试跑输出保存在 `results/online_parallel_smoke_v2.jsonl` 与 `results/online_parallel_raw_smoke.jsonl`。开发期间更新过代码，历史试跑的代码指纹不一定能在最终代码上直接 `--resume`；正式任务请使用新的输出文件。

## 历史串行配置验证：预处理 SAM + GPU 推理

- 四个日期的 SAM 清单覆盖全部 104,732 个 query key，无缺失条目；不是逐文件内容全量校验。
- 22 项测试通过，新增 SAM 图像绑定、特征摘要、缓存缺失不重算等测试。
- 在当时空闲的 GPU 0 上试跑 hard1502 清单前三帧，均为 `success`，SAM 来源均为 `cached`；仲裁分别选几何、视觉、几何。
- 三帧处理耗时分别为 15.78、8.49、7.76 秒，平均 10.68 秒；后两帧约 8.12 秒/帧。上述均不含入口的模型初始化，前三帧不能代表全集平均耗时。
- 平均阶段耗时：预处理特征读取与样本准备 0.44 秒、LEADER 0.82 秒、V7 refine 1.09 秒、几何 0.19 秒、实时视觉专家 8.02 秒、仲裁 0.09 秒。
- 输出为 `results/online_parallel_cached_sam_smoke.jsonl`；之前 CPU 重算 SAM 的 79–86 秒/帧及由此推算的全量工期不适用于当前默认配置。

## 六分片加速配置（2026-10-23）

当前默认：`--frontend-workers 4 --fast-pnp 1 --verify-cache 1 --verify-backend kernel`。保留 `--frontend-workers 1 --fast-pnp 0 --verify-cache 0` 作为串行/RANSAC 对照，SAM 缓存始终复用。

接入内容：原有 `infer_batch_engine_v2.ParallelFrames`、首批串行预热、单批 CPU 预取、固定预算参考图批量匹配、已有确定性快速 PnP、adaptive 1000/1500 几何搜索、参考点云 LRU、同参考集合的地图索引复用，以及 `v11-arbiter-fast-20261018-v1` 已有 `grid.c/libgrid.so` 有界精确距离核。没有采用已否决的单归属占用格近似、时序门控或跳过视觉验证。

C 核只返回 1 米内的精确最近允许参考点距离，超过范围的距离只用于阈值外判断；仍逐候选排除生成参考。只支持所有验证阈值不超过 1 米，索引大小有上限，不支持时明确报错而不静默替换规则。`nearest_unresolved` 是诊断字段，C 核的远距离处理不等同有限 k 梯度查询；判决分数需另行核对。

有限基准入口：

```bash
/home/zhangcy/.conda/envs/leader/bin/python -B benchmark_acceleration.py \
  --frames 48 --shards 6 --gpus 0,1,3
```

只有被选 GPU 利用率低、分配预算不超过各卡当前空闲的一半且主机内存有余量时启动；每卡两个分片，每进程 PyTorch 显存上限 15%。启动器只停止本次创建的进程组，不操作其他进程。此有限基准不是自动启动全集的命令。

同一批 hard1502 清单前 48 帧，6 分片分布在 GPU 0/1/3（每卡两片），全部成功：

| 配置 | 处理区间 | 区间/输出帧数 | 含启动墙钟 |
|---|---:|---:|---:|
| 六片，每片串行前端，旧 RANSAC 和旧验证 | 61.29 s | 1276.9 ms/帧 | 79.55 s |
| 六片 × 四流，快速 PnP，复用 KD 树 | 47.35 s | 986.4 ms/帧 | 66.76 s |
| 六片 × 四流，快速 PnP，C 距离核与缓存 | 29.21 s | 608.5 ms/帧 | 47.92 s |

第三行约为第一行吞吐的 2.10 倍，但没有达到 100 ms/帧。区间从六进程最早批处理开始到最晚批处理结束，未含入口模型初始化，但含首批流预热；每片只有 8 帧，不是充分预热后的全集稳态，也不是单帧延迟。

第三行平均阶段工作耗时：SAM 读取/样本准备 0.191 s、LEADER 0.466 s、V7 refine 按批均摊 0.541 s、几何 0.155 s、视觉 2.161 s、最终仲裁 0.092 s。视觉中 PnP 已约 0.031 s，激光验证仍约 1.613 s，是主要瓶颈。各阶段为进程内部计时，不能与六片聚合毫秒逐列直接相加比较。

数值核对：48 帧前端 T0 相对串行对照逐元素不变；C 核相对快速 PnP＋KD 树版本的最终 T 与最佳视觉分数也逐元素不变。**快速 PnP 相对旧 RANSAC 有 1/48 帧切换选择变化，最终平移差最大 0.897 m，未读 GT，不能声称精度无损。**

完整对账：`results/acceleration-comparison-20261023-v1.json`。基准输出分别位于 `results/acceleration-bench-20261023-110515/`、`results/acceleration-bench-20261023-110904/`、`results/acceleration-bench-20261023-111054/`；它们保留各自运行参数与源文件摘要，开发期间变更代码后不一定可直接恢复旧输出。


## GPU shared selection (2026-10-24)

Experimental `--selection-mode gpu_sparse`; default stays `full`. See [GPU selection trial](gpu-selection-trial-20261024.md) for fixed-evidence equivalence, latency scope, bounds, and limitations. This opt-in uses sparse shared evidence instead of the full visual verifier.
