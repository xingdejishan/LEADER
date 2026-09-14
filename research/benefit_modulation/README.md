# 收益监督的视觉条件通道调制

独立分支 `research/benefit-channel-modulation`，从 `807c9ff` 的原 LEADER / 修复投影缓存基线开始。

用户研究要求保存在 [proposal.md](proposal.md)，实现见 [model.py](model.py)，实验入口见 [experiment.py](experiment.py)。结果和边界见 [results/REPORT.md](results/REPORT.md)。

运行环境：现有 WSL `egonn118`（PyTorch 2.0.1 + CUDA 11.8、MinkowskiEngine 0.5.4、NumPy、SciPy、safetensors）。从本目录依次运行：

```bash
python check.py
python experiment.py
python reference.py
python provenance.py
python report.py
```

默认输入为 `/home/zhang/leader-image-gate-raw` 的 manifest、lidar 和 visual 缓存；原 LEADER 权重使用工作区 `research/image_gate_checkpoint/model.safetensors`；①的权重是缓存目录 `aligned.pt`。路径集中在 experiment.py 的 `ARGS` / `OUT`。结果首先写到 `/home/zhang/benefit-channel-modulation`，报告脚本再复制到本目录 results。

已有 development.json 的组会跳过训练；需要完全复跑时改 OUT 指向新的空目录。未完成组会从第 1 轮重新开始，不支持中途优化器断点续训。六组均为完整 100 轮训练，内部选出的 best.pt 用于报告，last.pt 保留末轮结果。

本轮不采用上一轮可靠度残差头的“第 0 轮可获选”规则，也不更改投影来追求覆盖率；13 帧内部留出仅用于选模，不参加参数更新。所有数据仍是小规模本地开发数据，不能代表完整 NCLT。

追加的无需训练 GT 辅助诊断由 `python oracle.py` 执行，结果见 [results/oracle/REPORT.md](results/oracle/REPORT.md)。它复用六组 best.pt，在原预测、正常门控和有界全开尝试之间按逐点 GT 坐标误差选择，并固定原候选索引与顺序；不是可部署方法或定位性能上限。
