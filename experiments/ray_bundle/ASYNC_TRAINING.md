# 双 GPU 训练与验证

本次仅改变运行调度，模型、训练样本、损失、优化器、地图、噪声和评价器沿用原冻结训练版本。

训练在每轮结束时原子保存独立 checkpoint，包含模型、Adam 状态、NumPy / PyTorch / CUDA 随机状态，再发布带 SHA-256 的验证任务。训练随即进入下一轮；另一张 GPU 只加载已冻结的 checkpoint，生成完整 1200 帧预测，再由独立 CPU 评价器评分。验证前后均核对 checkpoint 哈希，评价结果与对应轮次绑定。

验证进程独占更新 best 记录，沿用 MPE、MOE 和相对同代码 L0 的归一化综合指标。队列最多积压两轮，达到上限时训练等待，避免验证异常造成无界积压。两个进程分别使用已有资源保护；其中一个失败时，另一个在队列边界检查退出状态。

原串行入口在验证完成后才保存优化器与随机状态，因此迁移必须等待完整轮次落盘；不能直接拿仅含模型的候选 checkpoint 续训。迁移保留原日志和 checkpoint，从已完成轮次的下一轮继续。原源码文件保持不变，新入口会核对原 settings、prepared、baseline、源码、地图、标签和噪声的哈希。

新增入口：`raybundle.async_train`、`raybundle.async_validate`；运行路径由参数传入。队列与结果保存在共享输出目录的 `queue/` 和 `results/`，训练和验证进度分别保存在 `training/` 与 `validation/`，完整训练 checkpoint 保存在 `checkpoints/`。

新增两项检查覆盖续训后下一次 Adam 更新及随机排列完全一致，以及拒绝被修改或重复发布的 checkpoint。此次没有新增科学实验、消融或改变研究机制。
