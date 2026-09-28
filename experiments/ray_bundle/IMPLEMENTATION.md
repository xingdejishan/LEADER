# LiDAR 锚定的射线束对应生成

实现依据是同目录 `PROPOSAL.md`（用户第二份材料）。这是一个联合对应推断模块：冻结 LEADER 提供原始点对及可靠性；参考图像形成带不确定性的射线束地图；模块交替推断视觉地标身份、共享地标位置和查询位姿。没有采用此前建议的“只把 LEADER 压成固定六维位姿先验”替代方案。当前已接入 Cam5 和有限地标的局部高斯近似；真实运行记录见 `RUN_STATUS.md`。

## 模块行为

1. 建图只读取训练日期的参考图像、相机标定与建图位姿。保留真实相机光心和射线，通过多视图轨迹形成地标均值及各向异性信息矩阵，同时保存原始射线。视觉特征不要求附近有 LiDAR 回波。
2. 冻结官方 L0 的编码器、坐标回归器和 BN。SC2PCR 与原两步精修给出初始位姿，原始源点、预测世界点和可靠性仍进入联合求解；查询 GT 不进入模块。
3. 每个查询特征保留最多 4 个离散地标候选及 null 状态。外观得分来自 RootSIFT128 上的可训练残差描述子适配器；候选位置绝不加权平均成一个虚构对应点。按参考相机位置筛选附近的射线束，不按地标深度截断远景。
4. 对当前观测组排除其视觉因子，使用 LiDAR 与其余组形成局部 cavity 分布。候选评分包括预测角残差、传播的位姿/地标联合协方差及 log-determinant 归一化项，随后替换关联消息。
5. 联合正规方程同时更新一个 SE(3) 位姿和各个共享三维地标。每个地标只有一份地图先验，多相机观测复用同一变量；通过 Schur 消元求解。训练使用软责任，推理使用每相机一对一的离散指派并允许 null。

LiDAR 点误差采用“共享六维局部刚体误差＋独立点残差”模型，通过 Woodbury 消元处理相关性，避免重复点无限增加位姿精度。每帧真实点几何及可靠性仍决定正规方程。可靠性是权重，不能直接解释为已校准的概率。

输出 `pose`、各候选的 `probabilities`、原候选身份 `candidates`、离散匹配 `selected`（-1 表示 null）、共享地标 `active_ids/points` 及其局部后验 `point_covariance`。`probabilities` 是逐观测边缘评分归一化，`selected` 还受一对一约束，二者不必逐行 argmax 相同。

## 已实现的接口

- `raybundle/model.py`：可微对应生成与联合更新。
- `raybundle/geometry.py`：实际光心射线雅可比、相关 LiDAR 误差、共享地标消元与 cavity 评分。
- `raybundle/mapping.py`：NCLT Cam5 全图特征及训练参考射线建图。
- `raybundle/leader.py`：冻结官方 L0 的真实点云推理，返回原始对应与实际相机射线。
- `raybundle/cli.py`：建图、缓存、训练、预测；训练 GT 放在独立标签文件，预测只读取无 GT 缓存。
- `raybundle/calibration.py`：仅以至少 12 个训练帧拟合点误差尺度，输出可供训练/预测覆盖缓存噪声模型的 JSON。

服务器执行目录限定 `/home/zhangcy/projects/leader-ray-bundle-20260928`，解释器限定 `/home/zhangcy/.conda/envs/leader/bin/python`。执行前设置 `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`；本轮只用单线程 CPU，不安装依赖或占 GPU。CLI 限制地址空间 8 GiB、CPU/墙钟时间，磁盘低于 3 GiB 或系统可用内存低于 32 GiB 时停止。已有输出拒绝覆盖。

示例（路径变量均指本人项目内的输入/输出；显式传入所需训练步数）：

```bash
python -B -m raybundle.cli build-map --data-root "$DATA" --manifests "$TRAIN_MANIFEST" --start 0 --count 12 --output "$PROJECT/map.npz"
python -B -m raybundle.cli prepare --data-root "$DATA" --manifests "$TRAIN_MANIFEST" --split train --checkpoint "$L0" --noise-model "$NOISE" --start 12 --count 12 --output "$PROJECT/train-frames"
python -B -m raybundle.calibration --frames "$PROJECT"/train-frames/*.npz --labels "$PROJECT/train-frames/training_labels.json" --output "$PROJECT/fitted-noise.json"
python -B -m raybundle.cli train --map "$PROJECT/map.npz" --frames "$PROJECT"/train-frames/*.npz --labels "$PROJECT/train-frames/training_labels.json" --noise-model "$PROJECT/fitted-noise.json" --steps 2 --output "$PROJECT/train-run"
python -B -m raybundle.cli predict --map "$PROJECT/map.npz" --frames "$PROJECT"/query-frames/*.npz --model "$PROJECT/train-run/last.pt" --noise-model "$PROJECT/fitted-noise.json" --output "$PROJECT/predictions"
```

`$NOISE` 必须显式提供 `independent_sigma`（米）和 `shared_covariance`（平移在前、旋转在后的 6×6 正定矩阵，右扰动、米/弧度）。初次缓存可用明确标注的工程尺度，随后只用训练误差拟合并覆盖；功能检查中的尺度不是正式科学结论。参考地图不得包含当前查询图像；正式训练应使用分离的建图/查询帧，或为每个训练折单独建图。缓存及地图均绑定 SHA256；模型绑定具体地图，预测保留全部输入的成功/失败分母。

## 实现边界

`prepare --split dev` 必须使用既有无 GT 开发清单；数据加载器会拒绝带 `ground_truth` 或 `pose` 字段的开发清单。训练建图可使用训练 GT，在线模块只接收显式白名单字段的查询缓存。此次未运行开发集评价。

当前真实数据接入 NCLT 单相机 Cam5；核心张量接口支持不同 `camera_ids` 与实际光心，但未运行六相机完整数据链路。RootSIFT 是已有环境无需新增依赖的特征前端，训练的是其描述子适配器，而非原始像素特征提取器。

参考射线束用有限三维地标的局部各向异性高斯近似，允许弱深度信息但不人为增加其精度；无法辨识深度、纯无穷远或超出 0.5–2000 米范围的轨迹不入图。本版本没有实现完整逆深度非高斯分布。参考建图位姿视为固定，尚未传播其相关误差。

cavity 与联合求解是局部线性化、软责任交替更新的近似：排除了当前组显式因子，不意味着从整个迭代历史严格消除所有间接反馈，也不保证全局最优。匹配硬指派与训练软责任存在差异。默认每帧最多 64 个视觉特征、4 候选、3 轮软更新；推理另做 1 轮离散更新。数值步长上限仅用于稳定优化，不是性能安全保证。

已有三角化＋GLS、固定视觉匹配精修的历史负结果仍有效。这里的实现增量是可学习外观、保留多候选身份、cavity 重关联与共享未知地标联合求解；没有把三角化、消元或射线本身称为创新。`model_mink.py` 的稀疏坐标对齐、`pose_util.py` 的 CPU 兼容和原精修工具来自既有已审计代码，不算本机制贡献。

实现及功能检查不等于正式训练，更不构成优于第二份原提案、MaGiC 或 L0 的精度证据。
