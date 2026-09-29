# LEADER＋MaGiC DeepFusion（VRA＋MMA）

当前实验源码快照，来源提交 `094926b46c8605bfaada4c1ee462f2f9ecca6182`。核心源码与该提交逐字节一致；新增文档及便于选择方法的训练入口不改变训练数学。

## 方法

RPGE的32/128/384通道、stride 2/8/16三层几何特征，分别通过VRA（Voxel Region Attention，体素区域注意力）读取Cam5的局部图像区域，再通过MMA（多尺度自适应聚合）融合，零初始化输出投影形成512D残差，加回原LiDAR特征后送入原MMRegressor。SAM-L冻结，RPGE、VRA、MMA及MMRegressor联合微调。

这是当前已运行的LEADER＋MaGiC式VRA＋MMA版本，不是原MaGiC/SGLoc独立模型。三尺度图像特征由一个冻结SAM `[256,64,64]` embedding按有效覆盖归一化池化成64/32/16金字塔；不是分别读取SAM三个不同中间层。保留实际实现，不能把“完整VRA＋MMA”表述为已证明逐项等同论文全部实现。

核心文件：`models/model_mink.py`、`models/magic_paper_fusion.py`、`tools/magic_nclt_runtime.py`。训练入口：`python -m tools.train_leader_magic`。推理与独立评价入口：`tools/predict_magic_nclt.py`、`tools/evaluate_magic_nclt_dev.py`。

## 训练与复现

- 从官方LEADER L0权重开始；RPGE学习率1e-5，MMRegressor 1e-4，融合模块1e-4。
- Adam，StepLR每轮乘0.9，seed20，batch4，40轮目标，每轮300次更新。
- 原预训练BN统计冻结，affine参数可训练；原TRR损失及SC2-PCR＋1.2/0.6m精修保持。
- 训练日期：2012-01-22、2012-02-02、2012-02-18、2012-05-11，各300帧。
- 开发日期：2012-02-12、2012-02-19、2012-03-31、2012-05-26，各300帧，每轮评价选权重。
- 本协议没有执行地面水平校正，T_corr=I。
- 四个开发日期参与选模，成绩不是独立测试结果，也不代表完整NCLT序列。

依赖与上游署名见 [上游说明](UPSTREAM_README.md) 和 LICENSE。使用已经配置好的LEADER环境；本快照不附数据、SAM缓存或模型权重。

完整命令及冻结资产要求见 [运行说明](experiments/publication/RUNNING.md)。必须重新生成本公开源码对应的L0报告，不能复用绑定旧源文件路径/哈希的报告。服务器正在运行的实验未切换到此公开快照；不要据此覆盖其settings或恢复状态。

## 已核验成绩

见 [结果与比较限制](experiments/publication/RESULTS.md)。两组共同前14轮选出的综合最佳，纯LEADER为0.179498m/1.624231°，完整融合为0.177698m/1.610887°，额外改善约1.00%/0.82%；单次开发集结果不能证明稳定因果收益。

源文件逐SHA与来源见 [source_manifest.json](experiments/publication/source_manifest.json)。
