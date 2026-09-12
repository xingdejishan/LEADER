# GLACE 本地局部实验集

本目录包含从服务器实物复制的数据和权重，无需连接服务器读取缓存。训练 907 张、验证 303 张、测试 148 张；范围是原半径 40m 区域。验证日期为 2012-02-18，测试日期为 2012-02-12，测试集已用于开发诊断，不能当作新的最终留出集。

当前模型尚未通过稳定融合收益验收，详见 [实验状态](reports/EXPERIMENT_STATUS.md)。

研究目标、方案实施状态和下一阶段数据要求见 [RESEARCH_GOAL.md](RESEARCH_GOAL.md)；固定候选对应点替换实验见 [回放报告](code/research/glace_fusion/replay_results_20260912/REPORT.md)。

最新的同一 v1 起点视觉开关实验见 [后端与视觉损害归因](code/research/glace_fusion/fixed_origin_results_20260912/REPORT.md)。

## 内容

- `code/`：服务器分支的源码快照，含本轮未提交改进；本地路径适配记录在 LOCAL_PATCHES.json。
- `data/`：三组局部图像、RGB 特征、相机标定及位姿、有效视野掩码、训练三维监督、验证和测试所需的可用同时间戳扫描。
- `models/selected/`：验证选出的三维损失权重 5 模型；另有 previous、uniform、balanced 对照，以及全局特征模型。
- `cache/`：各模型对应点、验证对应点、置信度模型和筛选结果，以及真实 LEADER 对应点与候选池。
- `reports/`：下载时已完成的评估报告；PROVENANCE.json 保存实验来源和限制。
- `outputs/`：本地新实验输出，不覆盖原缓存。

## Windows 使用

在本目录打开终端，使用隔离环境中的 Python：

```powershell
.\.venv\Scripts\python.exe local.py verify
.\.venv\Scripts\python.exe local.py summary
.\.venv\Scripts\python.exe local.py audit --variant selected --out outputs\my-audit.json
.\.venv\Scripts\python.exe local.py joint --variant selected --out outputs\my-joint
.\.venv\Scripts\python.exe local.py joint --variant confidence --out outputs\my-filtered-joint
.\.venv\Scripts\python.exe local.py confidence --out outputs\my-confidence
.\.venv\Scripts\python.exe local.py infer --variant selected --limit 2 --out outputs\my-inference
.\.venv\Scripts\python.exe local.py replay --variant selected --out outputs\my-replay
.\.venv\Scripts\python.exe local.py refine-ablation --variant selected --out outputs\my-refine-ablation
```

audit、joint 和 confidence 可在 CPU 上运行；infer 使用已有 CUDA PyTorch 和本地 GPU。输出路径必须不存在；省略 limit 表示全部测试帧。

## 修改代码和训练

优先修改 code/research/glace_fusion/ 内的对应点、置信度和共享求解代码，通过上述入口做局部实验。原历史实验脚本保留服务器绝对路径用于溯源，本地请使用 local.py 入口。

训练环境已配置在 WSL Ubuntu 的 `/home/zhang/.venvs/glace`，使用 Python 3.11、PyTorch 2.6.0 + CUDA 12.4 和 torchvision 0.21.0。Windows 入口会自动转到 WSL，无需手动激活环境。在本目录运行：

```powershell
.\.venv\Scripts\python.exe local.py train --variant selected --iterations 10000 --out outputs/new-training
```

先检查流程可将迭代数改为 100；每次使用新的输出目录。训练完成后，加载新权重进行推理：

```powershell
.\.venv\Scripts\python.exe wsl.py infer --model-dir outputs/new-training --limit 2 --out outputs/new-training-inference
```

该入口从头训练并保留原模型，不会续接原优化器状态。不要用验证/测试扫描作为训练输入；训练入口只读取 data/train_scene/train/lidar_world。

已有模型的 vendor 代码与权重哈希配套。训练入口会复制 vendor 到新输出目录，通过公开优化器 hook 兼容新版 PyTorch，并更新新配置中的 vendor_hashes。需要进一步修改训练实现时，应在新实验中修改并更新哈希，不能把改后的实现冒充原模型。

MANIFEST.json 保存服务器原始文件校验值，LOCAL_PATCHES.json 记录本地适配文件的新校验值。修改代码后 verify 会报告对应文件变化，这是预期行为。

## 本机验收

已通过文件完整性校验、9 项单元测试、148 帧对应点审计、2 帧融合重放、2 帧保存置信度模型精确重放，以及 RTX 4060 Laptop GPU 上的单帧神经网络推理。原验收详见 LOCAL_VALIDATION.json。

WSL 环境已用全部 907 张训练图像构建缓存，完成 100 步 GPU 训练、保存权重，并成功重新加载新权重完成单帧推理。训练总耗时约 37 秒，输出为 `outputs/wsl-training-ready/head.pt`；这仅验证训练流程可运行，不代表模型已经收敛或精度提升。详见 WSL_VALIDATION.json。

训练三维监督实际覆盖 905 / 907 张图像；缺失扫描对应的图像仍保留重投影训练，不凭空补造深度。
