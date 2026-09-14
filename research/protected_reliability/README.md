# Protected visual reliability

独立分支：`research/protected-visual-reliability`，从多模态交接提交`807c9ff`创建。

- [研究结果](results/REPORT.md)
- [训练前固定协议](results/protocol.json)
- [用户研究方案](proposal.md)
- [用户原型，保持原样](prototype.py)
- [训练与评估入口](experiment.py)

原型仅修改可靠度，固定所有原始坐标，保护原高可靠核心。复用原LEADER预测及投影修复后的①缓存，不使用②微调过的回归头。

运行（原本机WSL的egonn118环境）：

```bash
python research/protected_reliability/prototype.py
python research/protected_reliability/experiment.py prepare
python research/protected_reliability/experiment.py pilot
python research/protected_reliability/experiment.py refit
python research/protected_reliability/experiment.py evaluate
python research/protected_reliability/report.py
```

固定输入位于`/home/zhang/leader-image-gate-raw`，实验输出独立保存到`/home/zhang/protected-visual-reliability`。其他机器按上层[交接文档](../image_gate/HANDOFF.md)恢复缓存并适配路径；小头权重随本目录结果提交。
