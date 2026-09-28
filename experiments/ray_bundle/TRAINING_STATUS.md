# GPU 训练任务状态

用户已授权正式训练；本文件覆盖 `RUN_STATUS.md` 中截至实现完成时“尚未正式训练”的历史状态。

任务已在 GPU 3（RTX 4090，`GPU-ba134650-773c-df34-5e38-9e2c8d1fa892`）启动，启动前相隔 30 秒的两次空闲检查通过。本人 guard PID `966333`、训练入口 PID `966688`；已有其他 GPU 任务未修改。仅使用现有 `/home/zhangcy/.conda/envs/leader`。

服务器目录 `/home/zhangcy/projects/leader-ray-bundle-20260928/run-train-v1`，运行代码 `source-training-v1`，对应提交 `4a257ea3683ed94b5b34845d858da93d1658d01d`；58 文件部署包 SHA256 `d6a05ca246edb6a3537883740719fd92fa00a8a23917615843d8faf867a0c548`。全部运行 Python 文件与通过 GPU 检查的源码逐字节相同。冻结 settings SHA256 `ce7693de4dc8234355ca17805ce0445ceac2fadae6140ee2adb4b1010725ca0d`。

最近核实阶段为训练缓存 `100/1200`，进程仍存活、guard 正常；GPU 总占用约 1888 MiB，进程 RSS 约 3.46 GiB。之后自动完成开发缓存、训练噪声拟合、互斥参考地图、同代码 L0 基线，再进入 40 轮参数训练。当前尚未确认正式优化器更新，没有整轮 checkpoint 或新开发精度。不要重复启动该任务。

固定 1200 训练/1200 开发帧、40 轮、物理 batch 1 累积 4、Adam 1e-4、冻结 LEADER，仅训练对应模块；详见 `TRAINING.md`。运行只读状态为 `progress.json`、`guard.json`、`console.log`，结束信息为 `exit.json`。每轮完成后生成 `history.jsonl`、`latest.pt`、三项 `best_*.pt`，正式完成才写 `completion.json`。

GPU 真帧两步检查已通过：64 特征，梯度非零，LEADER 不变，PyTorch 峰值预留 432 MiB、进程峰值 RSS 3627.74 MiB，检查权重丢弃。此前 GPU 2 在复查时被占用，未启动计算；首次 GPU 3 检查遇到旧 PyTorch/NVRTC 的 `slogdet` 架构兼容错误，改用数学等价的 2×2 显式行列式后通过，没有更换或安装环境。失败证据保留，不作为科学负结果。

实际显存限制为卡容量 35%，RSS 内部 10 GiB/外部 12 GiB、磁盘 3 GiB 下限；同卡出现其他进程时只停止本任务进程组。本轮未创建自动跟进或通知任务。
