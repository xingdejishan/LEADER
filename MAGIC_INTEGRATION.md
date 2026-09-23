# MaGiC 融合分支

本分支从 `main` 的 `4a1bde8` 创建，随后引入已冻结的 SC2-PCR＋`(1.2, 0.6)` 两阶段全池精修作为共同位姿后端。`--magic_manifest` 缺省时仍运行该纯 LiDAR 路径。

## 接入内容

- **增强恢复（AR）**：LEADER 的极坐标体素中心先转回校正后的 LiDAR 笛卡尔坐标，再用 `T_corr` 的逆变换回数据集 `scan` 在地面校正前的坐标，最后按显式标定投影到相机。当前 main 无随机几何增强，因此恢复项只处理原有地面校正；融合前不使用查询 GT 位姿。
- **体素区域注意力（VRA）**：以 LiDAR 特征为 query，在投影位置周围的 3×3 图像特征区域生成 key/value，softmax 加权后与 LiDAR 特征融合；深度非正或视野外的体素不接收视觉增量。
- **多尺度聚合（MMA）**：在 LEADER 输出体素上按 1、2、4 倍网格聚合 LiDAR 特征，与图像编码器的 1/8、1/16、1/32 特征分别执行 VRA，再用跨尺度门控融合。聚合结果以零初始化残差接到原有坐标回归头，载入 LiDAR 权重时初始输出严格相同。
- 坐标监督、TRR 损失、对应点选择和 SC2-PCR＋两阶段全池精修保持 LEADER 口径。

这是**面向 LEADER 的 MaGiC 原语移植**，不是论文的逐项复现：论文使用 SAM-large 图像特征与原生多尺度 3D 骨干，本分支使用可训练的轻量图像金字塔与 LEADER 输出体素的分组聚合；论文的 L1 坐标损失也未替换 LEADER 的 TRR。因此不能引用论文指标作为本分支效果。

## 标定清单

`--magic_manifest` 指向 UTF-8 JSON。顶层 `frames` 的键是扫描文件相对于 `--dataset_folder` 的路径，统一使用 `/`。每一帧必须含有：

| 字段 | 含义 |
|---|---|
| `image` | RGB 图像路径；相对路径从清单目录解析 |
| `K` | **原图**像素坐标的 3×3 内参矩阵 |
| `T_camera_lidar` | 从数据集 `scan` 在地面校正前的坐标系到相机坐标系的 4×4 刚体变换；Oxford 的轴变换已经由数据加载器执行 |

键的形状示例为 `NCLT/2012-01-22/velodyne_sync/<timestamp>.bin`。程序按图像最长边缩放至 `--magic_image_size` 并在右侧、底部补零，同时缩放 `K`；不会猜测时间同步、外参或相机型号。训练和评价数据中的每个扫描都要有记录，缺项直接报错。清单不能包含查询 GT、预测后筛选结果或以查询 GT 选择的参考帧。

```powershell
python run_mink.py --dataset NCLT --mode train --dataset_folder <数据根目录> --magic_manifest <标定清单.json> --magic_init_weights <LiDAR模型状态字典> --log_dir <新输出目录>
```

`--magic_init_weights` 接受本地 PyTorch 状态字典文件，可从现有 LiDAR-only checkpoint 的 `pytorch_model.bin` 初始化编码器和回归头；融合层新建。继续训练已保存的多模态 checkpoint 时使用原有 `--resume_model`，两者不可同时传入。图像尺寸至少 64 且须为 32 的倍数，默认 320。

## 验证边界

当前只有合成几何与接口单元验证，没有用真实同步图像训练或报告 MPE/MOE。`run_mink.py --mode test` 是原仓库的开发诊断入口，仍在同一进程读取 GT，不能充当研究交接记录规定的正式在线评价。正式比较需要独立在线预测与 GT evaluator，并将纯 LiDAR 与多模态放在同一图像、扫描及完整帧分母上；32 帧历史开发集不得称为独立测试。
