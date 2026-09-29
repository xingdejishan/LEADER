# LEADER + Forward Appearance Registration (FAR)

原版 FAR：冻结纯 LiDAR LEADER 与 SAM，以训练场景的多分辨率哈希外观场和图像观测做局部三维相关，求共享6DoF修正。保留 LiDAR 初值精修和 FAR 后的原对应后端。

方法：[IMPLEMENTATION](experiments/far/IMPLEMENTATION.md)；已核实第1–10轮：[RESULTS](experiments/far/RESULTS.md)；结构化证据：[results.json](experiments/far/results.json)；数据边界：[AUDIT](experiments/far/AUDIT.md)；开销：[COST](experiments/far/COST.md)。当前第10轮0.160502869m/1.471401649°，是开发集选模结果，非独立测试；40轮尚未完成。去精修版不在本分支。

## 环境与资产

实际运行使用Python3.8、PyTorch1.12和MinkowskiEngine，沿用LEADER的SC2-PCR后端。依赖信息见 requirements-magic.txt；原项目说明见 UPSTREAM_README.md。数据、SAM缓存和官方L0权重需另行准备；严格冻结的日期、输入清单和资产SHA由代码检查，不能使用任意重新采样输入冒充同一实验。

本仓库提供方法代码和结果，不包含模型权重、传感器数据、SAM缓存、机器连接信息或私有运行配置。下面的相对路径是使用者本机的示例位置。

## 运行

先生成包含本机路径与当前源文件SHA的配置；生成的配置和运行输出默认被Git忽略。

```bash
python -m tools.make_far_config --base-checkpoint assets/official_l0.pt --data-root data/NCLT --train-sam assets/train-sam/cache_manifest.json --dev-sam assets/dev-sam/cache_manifest.json --output configs/local/far.json
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 python -m tools.train_far --config configs/local/far.json --output outputs/far
CUDA_VISIBLE_DEVICES=1 python -m tools.validate_far --config configs/local/far.json --run outputs/far --gt-root data/NCLT/evaluator_only/test_gt
```

训练和验证在两个终端启动。配置固定seed20、40轮、Adam1e-3、每4帧累积更新；训练只读取四个训练日期的GT，预测使用四个不含GT的开发清单，独立CPU评价器在冻结预测后读取评价真值。

单独推理与评价：

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.predict_far --config configs/local/far.json --checkpoint outputs/far/checkpoints/10.pt --output outputs/prediction10
CUDA_VISIBLE_DEVICES=-1 python -m tools.evaluate_far --config configs/local/far.json --checkpoint outputs/far/checkpoints/10.pt --predictions outputs/prediction10/predictions.jsonl --freeze outputs/prediction10/prediction_freeze.json --gt-root data/NCLT/evaluator_only/test_gt --output outputs/prediction10/evaluation.json
python -m unittest discover -s tests -p test_far.py
```

隐式场约12MiB，定位每帧最多51.2万次局部查询。现有时间记录使用预计算SAM缓存，不包含在线SAM编码；详见COST.md。
