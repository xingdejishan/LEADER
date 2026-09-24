# LiDAR有限三维表面＋PixLoc多尺度直接配准

## 结果

本实验从仅含训练集观测的LiDAR视觉地图拟合有限局部表面；在一个共享六自由度车体位姿下重投影表面采样点，并直接对齐预训练PixLoc多尺度特征。只训练PixLoc原有的特征适配层。使用47帧训练20个固定epoch，权重固定取最后一轮，没有用开发集挑选检查点。

冻结的32帧开发集评估**没有显示该方法优于`peak_then_pose`**。新方法的平均误差略高，配对区间跨零。32/32帧都产生有限且成功的位姿。这32帧来自反复使用的同一条2012-02-18开发路线，不是独立序列测试。

| 方法 | 平均MPE | 平均MOE | MPE P90 | MOE P90 |
|---|---:|---:|---:|---:|
| 纯几何对照 | 89.916 mm | 0.902451° | 133.623 mm | 1.408789° |
| `peak_then_pose` | **87.139 mm** | **0.881202°** | **129.616 mm** | **1.392802°** |
| PixLoc表面直接配准 | 89.338 mm | 0.898745° | 133.025 mm | 1.406420° |

相对`peak_then_pose`，新方法的均值变化为**MPE +2.199 mm、MOE +0.017544°**（正值表示更差）。以相邻4帧为块的配对描述性95% bootstrap区间为`[-2.312, +6.719] mm`和`[-0.008527, +0.036616]°`，均跨零。MPE在13/32帧改善，MOE在10/32帧改善，两项同时改善仅2/32帧；两项P90都比peak差。中位数略好不改变均值和尾部不占优的结论。当前证据不支持替换peak基线。

## 固定方案

- 40,335条训练视觉地图观测来自已有LiDAR reference map，拟合后得到2,743个有有限范围和LiDAR支持的局部表面。训练帧共使用11,282个表面patch和75,480个表面采样点。
- 表面由16个邻近LiDAR地图点拟合；通过局部凸包、最近地图支持距离不超过0.25 m、patch半径不超过0.5 m等条件筛除无支持样本。没有把平面无限延伸，也不使用LEADER预测的场景点替代LiDAR地图真值。
- 参考表面使用训练帧LiDAR缓存中的训练位姿，从参考像素反投影生成。query验证真值没有用于训练或预测。
- 冻结官方PixLoc MegaDepth VGG19编码器、解码器和不确定度分支，只训练三个多尺度特征适配层。六相机共同优化一个位姿，特征尺度由粗到细为`[16, 4, 1]`，保留一份帧级LiDAR信息先验；patch内采样权重归一化。
- 训练损失第1轮为1.49948，最终第20轮为1.50063，整体变化很小且最终略高。此项作为方法风险报告，不根据开发集改变epoch或挑权重。
- 预测run先冻结，之后由单独evaluator读取GT。32帧平均覆盖237.4个patch，范围154–312；solver失败数为0。总solver墙钟时间103.82秒，不包括全部数据预处理与PixLoc图像特征开销。

## 文件

- `../../pixloc_surface_direct.py`：训练、预测和评价实现。
- `pixloc_surface_adapters.pt`：最后一轮特征适配层及协议元数据。
- `training_report.json`：数据/模型哈希、表面覆盖、训练曲线和配置。
- `validation_run.json`：冻结的逐帧预测，不包含query GT误差。
- `evaluation.json`：冻结后独立评价、逐帧误差、比较区间、覆盖率和耗时。
- `SHA256SUMS`：源码及已提交产物的SHA-256。

不包含图像、LiDAR、逐帧中间缓存、PixLoc仓库或官方预训练checkpoint；训练报告保留对应输入哈希。

## 复现环境

- PixLoc官方仓库`cvg/pixloc`，commit `6f7a943afc34183654754f9c4e90672e491a629b`。
- 官方MegaDepth checkpoint SHA-256：`4718a885b3e0e8852157d57f82f3221fe971a8f6b8ac3a5a4681b677ccf05dc7`。
- WSL conda环境`/home/zhang/miniconda3/envs/romav2`，PyTorch 2.6、CUDA、RTX 4060 Laptop 8 GB。
- 脚本默认数据路径指向本机分支外文件；在其他环境复现时需通过CLI参数指定输入。

```bash
python research/pixloc_surface_direct.py train --epochs 20
python research/pixloc_surface_direct.py run
python research/pixloc_surface_direct.py evaluate
```

这些开发路线已多次用于方法研究。结果只支持开发阶段判断；独立泛化结论需要未参与方法选择的序列。