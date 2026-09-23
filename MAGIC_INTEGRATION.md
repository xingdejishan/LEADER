# MaGiC 融合分支

本分支从 `main` 的 `4a1bde8` 创建，随后引入已冻结的 SC2-PCR＋`(1.2, 0.6)` 两阶段全池精修作为共同位姿后端。`--magic_manifest` 缺省时仍运行该纯 LiDAR 路径。

## 接入内容

- **增强恢复（AR）**：LEADER 的极坐标体素中心先转回校正后的 LiDAR 笛卡尔坐标，再用 `T_corr` 的逆变换回数据集 `scan` 在地面校正前的坐标，最后按显式标定投影到相机。当前 main 无随机几何增强，因此恢复项只处理原有地面校正；融合前不使用查询 GT 位姿。
- **体素区域注意力（VRA）**：以 LiDAR 特征为 query，在投影位置周围的 3×3 图像特征区域生成 key/value，softmax 加权后与 LiDAR 特征融合；深度非正或视野外的体素不接收视觉增量。
- **SAM-large 图像特征**：使用 Meta 官方 SAM ViT-L 权重及图像编码器，按其 1024 像素预处理生成 256×64×64 特征，冻结并缓存。缓存由图像文件和 checkpoint 的 SHA-256 定址，训练时不能静默换成轻量编码器。
- **多尺度聚合（MMA）**：在 LEADER 输出体素上按 1、2、4 倍网格聚合 LiDAR 特征，将 SAM-L 的 64×64 输出分别池化至 64×64、32×32、16×16 后执行 VRA，再用跨尺度门控融合。聚合结果以零初始化残差接到原有坐标回归头，载入 LiDAR 权重时初始输出严格相同。
- 坐标监督、TRR 损失、对应点选择和 SC2-PCR＋两阶段全池精修保持 LEADER 口径。

这是**面向 LEADER 的 MaGiC 原语移植**，不是论文的逐项复现：论文没有公开 SAM 三尺度特征的具体抽取代码，本分支从官方单层 SAM-L 输出确定性池化三个尺度；LiDAR 侧使用 LEADER 输出体素的分组聚合而非论文原生 3D 骨干层，且保留 LEADER 的 TRR 损失。SAM 编码器冻结是本分支的实现选择，不应冒充论文训练设置。因此不能引用论文指标作为本分支效果。

## 标定清单

先准备原始 UTF-8 JSON 清单。顶层 `frames` 的键是扫描文件相对于 `--dataset_folder` 的路径，统一使用 `/`。每一帧必须且只能含有：

| 字段 | 含义 |
|---|---|
| `image` | RGB 图像路径；相对路径从清单目录解析 |
| `K` | **原图**像素坐标的 3×3 内参矩阵 |
| `T_camera_lidar` | 从数据集 `scan` 在地面校正前的坐标系到相机坐标系的 4×4 刚体变换；Oxford 的轴变换已经由数据加载器执行 |

键的形状示例为 `NCLT/2012-01-22/velodyne_sync/<timestamp>.bin`。不会猜测时间同步、外参或相机型号。清单不能包含查询 GT、预测后筛选结果或以查询 GT 选择的参考帧。

先用 [Meta 官方 SAM ViT-L checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) `sam_vit_l_0b3195.pth` 生成冻结图像特征；脚本核对 SHA-256 `3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622`。它输出含每帧 `sam_features`、图像 SHA-256、checkpoint SHA-256 和 1024 像素缩放尺寸的新清单。训练和评价所需扫描都必须覆盖，缺项直接报错。

```powershell
pip install -r requirements-magic.txt
python tools/cache_sam_l_features.py --manifest <原始标定清单.json> --checkpoint <sam_vit_l_0b3195.pth> --out_dir <SAM特征缓存目录>
```

编码使用官方最长边 1024 像素的缩放、RGB 归一化、右下补零和 `image_encoder`；每帧保存 256×64×64 的半精度特征。融合分支只读取这个真实 SAM-L 特征，并按同一缩放调整 `K`。缓存脚本不读 GT；原 `run_mink.py` 的训练/评价路径仍读取 LEADER 的位姿标签，正式在线评价须另行隔离。

```powershell
python run_mink.py --dataset NCLT --mode train --dataset_folder <数据根目录> --magic_manifest <SAM特征缓存目录/manifest.json> --magic_init_weights <LiDAR模型状态字典> --log_dir <新输出目录>
```

`--magic_init_weights` 接受本地 PyTorch 状态字典文件，可从现有 LiDAR-only checkpoint 的 `pytorch_model.bin` 初始化编码器和回归头；融合层新建。继续训练已保存的多模态 checkpoint 时使用原有 `--resume_model`，两者不可同时传入。

## 验证边界

当前只有合成几何与接口单元验证，没有用真实同步图像训练或报告 MPE/MOE。`run_mink.py --mode test` 是原仓库的开发诊断入口，仍在同一进程读取 GT，不能充当研究交接记录规定的正式在线评价。正式比较需要独立在线预测与 GT evaluator，并将纯 LiDAR 与多模态放在同一图像、扫描及完整帧分母上；32 帧历史开发集不得称为独立测试。
