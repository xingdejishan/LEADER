# R-SCoRe-L：NCLT 局部对比

在 LEADER 的 `upstream/main` 起点上实现独立视觉前端，使用与现有 GLACE 相同的训练 907 张、验证 303 张、开发测试 148 张。分支为 `research/rscore-l-local`。

## 训练协议

SCRStudio 固定在 `a2b40bd19f63803be2e527b829ddb71b02b0a473`，GLACE 子模块固定在 `e704a8c718f25dea026a3d5b70776363da7dd665`。`vendor/` 保留官方源码；新增适配在本目录，通过配置继承使用官方 Trainer、优化器、调度器、两级回归网络和 Node2Vec。

使用官方 DeDoDe L/B、NetVLAD 预训练权重并冻结。PCA 仅用本次 907 张训练图像拟合到 128 维；Node2Vec 在本地训练共视图上重新训练 256 维编码；场景 MLP 从头训练，宽度 768，保留 `sc0 → refinement → sc`。

用户已选择先按 GLACE 的 10,000 步做局部对比。每次优化使用 8,192 个样本，分两次 4,096 样本累积；缓存为每图 1,024 个增强特征，共 928,768 个。保留官方 AdamW、OneCycleLR 和损失调度形式，将时间轴调整为 10,000 步。Node2Vec 保留官方 5,000 步及有效批量 256：位姿图使用 32×8 累积；确认显存占用后，LiDAR 图使用 64×4，保持有效批量不变。上述设置不等于官方四卡、100,000 步的大场景预设。

PCA 使用双精度协方差的精确特征分解，数学目标与官方无白化、中心化 PCA 相同，计算实现不同于 cuML。训练关闭旋转增强，保留官方缩放和光度增强；训练、PCA、推理均传递真实黑边掩码。

## 方法与实验臂

依次比较 `scrfacto`、原生深度初始化 `depth`、持续几何监督 `geometry`、加入 LiDAR 共视图的 `lidar`、训练后可靠性头 `reliable`，以及现有 `glace`。

表面目标通过单帧训练扫描的局部平面与像素射线相交构造，检查平面秩、拟合残差、支持范围、视角、深度竞争和近表面可见性。明确保存有效性、质量、沿视线与垂直视线误差尺度及统一世界体素标识。共视图另聚合最多 9 个、距离 5m 内的训练扫描，要求体素在至少两帧出现，再按当前相机的视锥、黑边掩码与 Z-buffer 检查可见性；不读验证或测试扫描。

这些目标尚无独立标定核验或充分的跨帧静态性核验，属于经过局部检查的弱几何参考，不能称为已证实的静态表面真值。几何图采用共同体素的对称重叠，要求至少 16 个共同体素及每图至少 3 个空间块；孤立节点保留自环，不伪造共视边。

持续几何监督显式处理有效标签，对粗层和最终层分别施加 0.5 和 1 倍几何损失，权重从 1 衰减到 0.25。已知几何不再受到固定 10m 伪目标牵引；无几何样本继续使用原版重投影与初始化路径。

可靠性头用训练集内按日期留出的坐标模型产生样本；冻结坐标预测后训练小型 MLP，单独保留一个训练日期的查询用于温度校准。验证 303 张、测试 148 张不参与可靠性拟合。候选排序损失的实现与后端使用相同混合评分，但局部包缺少训练侧真实 LEADER 候选池，因此这项监督尚不能实际拟合；报告必须保留这项限制，不能称为附件中全部训练目标已完成。新增候选池应放到运行目录的 `data/train_candidate_pools/`，按训练帧 ID 命名 NPZ，不能放入测试候选。

## 推理及评价

新增 `lidar-multiframe` 对照固定 `lidar` 的共视图、Node2Vec、特征缓存、损失和 10,000 次迭代预算，只改变坐标标签：保留原有单帧有效标签，从同一天、5m 内最多 9 个训练扫描聚合表面，按 0.05m 体素去重，使用相同射线与平面检查和 Z-buffer，再要求新增目标在至少两个扫描中获得 0.2m 内的空间支持。标签分别存储，不覆盖单帧缓存；空间重复支持只是弱静态一致性证据。`multiframe_report.json` 分别记录增强训练缓存与非增强特征缓存的覆盖率。

导出保留官方 10 个检索假设，每个假设独立输出三维点及可靠性，文件不包含真值。参考位姿仅由评价入口读取。不同假设不会平均或逐点混合。

相机位姿分别报告原生关键点数量和每假设 256 点两种 PnP 结果。系统回放对全部方法使用同一 LiDAR 候选池、每假设 256 个空间均衡点、内外点混合评分、空间棋盘格留出接受判断。LEADER 对照是原局部实验的 `v1_two_stage` 缓存，没有重新运行 main 的完整点云测试。R-SCoRe 有 10 个检索假设，GLACE 缓存只有一组坐标；每假设预算一致不等于总计算量一致，耗时另行报告。全体视觉可靠性过低时返回原 LEADER 位姿；独立的零视觉对照实际执行 LiDAR-only 精修。

已有 148 张测试和 303 张验证曾用于开发，结果只作为局部开发证据。三维正确率没有独立几何参考时不报告；不会用构造监督的同一位姿回投误差冒充独立标定验证。

## 运行

独立 WSL 环境：`/home/zhang/.venvs/rscore-l`，PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、PyG 2.6.1。

工作目录：`/home/zhang/rscore-l-local`。

```sh
bash launch.sh prepare
bash launch.sh features
bash launch.sh geometry
bash launch.sh overlap
bash launch.sh node2vec --graph pose
bash launch.sh node2vec --graph lidar
bash launch.sh train --variant depth
bash launch.sh all
bash launch.sh status
bash launch.sh check
bash launch.sh topk --split test
bash launch.sh topk-report
```

完整队列在 `state.json` 和 `logs/` 记录状态，失败后停止；完成标记只在相应子进程成功返回后写入。训练保存可恢复检查点。对比汇总只有全部评价完成后才会生成 `comparison.json`。

官方依据：[SCRStudio](https://github.com/cvg/scrstudio)、[R-SCoRe 论文](https://arxiv.org/html/2501.01421v2)。
