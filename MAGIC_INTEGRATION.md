# MaGiC 融合分支

本分支从 `main` 的 `4a1bde8` 创建，随后引入已冻结的 SC2-PCR＋`(1.2, 0.6)` 两阶段全池精修作为共同位姿后端。`--magic_manifest` 缺省时仍运行该纯 LiDAR 路径。

## 接入内容

- **增强恢复（AR）**：LEADER 的极坐标体素中心先转回校正后的 LiDAR 笛卡尔坐标，再用 `T_corr` 的逆变换回数据集 `scan` 在地面校正前的坐标，最后按显式标定投影到相机。当前 main 无随机几何增强，因此恢复项只处理原有地面校正；融合前不使用查询 GT 位姿。
- **体素区域注意力（VRA）**：以 LiDAR 特征为 query，将极坐标体素的八个角点投影到图像，以其包围框内固定 3×3 位置生成 key/value，softmax 加权后与 LiDAR 特征融合。角点深度无效的体素不取视觉候选；有效图像外的特征格在插值和图像金字塔池化中被 mask 排除。八角点包围框是本分支的几何实现选择，论文未规定具体区域算法。
- **SAM-large 图像特征**：使用 Meta 官方 SAM ViT-L 权重及图像编码器，按其 1024 像素预处理生成 256×64×64 特征，冻结并缓存。缓存由图像文件和 checkpoint 的 SHA-256 定址，训练时不能静默换成轻量编码器。
- **多尺度聚合（MMA）**：从 RPGE 的第 1/3/5 个 encoder 阶段取真实浅/中/深层稀疏特征，各层分别执行 VRA，再按带 batch 的稀疏体素坐标对齐至原有回归头体素。SAM-L 的单张 64×64 输出经有效区域加权池化成为 64×64、32×32、16×16；跨尺度门控结果以零初始化残差接到坐标回归头，载入 LiDAR 权重时初始特征相同。
- 坐标监督、TRR 损失、对应点选择和 SC2-PCR＋两阶段全池精修保持 LEADER 口径。

这是**面向 LEADER 的 MaGiC 原语移植**，不是论文的逐项复现：论文没有公开 SAM 三尺度特征的具体抽取代码，本分支从官方单层 SAM-L 输出池化三个尺度；LiDAR 侧使用 RPGE 的真实阶段特征，跨层对齐使用较粗层的稀疏坐标分组，而非论文原生 3D 骨干。区域使用极坐标体素八角点投影包围框，且保留 LEADER 的 TRR 损失。SAM 编码器冻结是本分支的实现选择。不能引用论文指标作为本分支效果。

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

`--magic_init_weights` 接受 LiDAR-only checkpoint 目录中的 `pytorch_model.bin`，并要求同目录有 `extra.json`，从中恢复有效的三维 `center_t`；缺失或无效时直接失败。融合层新建。旧 checkpoint 没有记录完整数据配置，仍需人工核对体素大小、水平分辨率和坐标约定。继续训练已保存的多模态 checkpoint 时使用原有 `--resume_model`，两者不可同时传入。

## 验证边界

WSL Ubuntu 的 `/home/zhang/.venvs/leader-magic/bin/python` 继承原 `egonn118` 环境的 PyTorch 2.0.1+cu118 与 MinkowskiEngine 0.5.4，在隔离的虚拟环境中补齐 LEADER 入口、SAM 与数据加载所需依赖。可从本仓库根目录运行：

```bash
/home/zhang/.venvs/leader-magic/bin/python -m unittest discover -s tests -p test_magic_fusion.py
/home/zhang/.venvs/leader-magic/bin/python -m tools.smoke_magic_full
```

合成几何与接口单元测试通过；完整 RPGE→阶段视觉融合→回归头的128体素 CUDA 合成前后向通过，浅层投影收到非零梯度，峰值分配约623 MiB。另有单张真实图像的 SAM-L 编码 smoke。上述显存数字不能外推到真实点云或训练 batch。图像范围外及 `valid_mask.npy` 指定的非矩形无效区，在直接采样和池化中排除；这不消除 SAM 编码图像时可能形成的上下文影响。`run_mink.py --mode test` 是原仓库的开发诊断入口，仍在同一进程读取 GT，不能充当正式在线评价。正式比较需要独立在线预测与 GT evaluator，并将纯 LiDAR 与多模态放在同一图像、扫描及完整帧分母上；32 帧历史开发集不得称为独立测试。

## 本地 905 帧训练协议

`tools/prepare_local905.py` 从 `glace-local/data` 中读取真实原始 NCLT `.bin`、同步图像、内参及相机到车体的外参，固定生成 `split.json` 和不含 GT 的 `raw_manifest.json`。905帧按采集日期划分：`2012-01-22` 的319帧与 `2012-02-02` 前233帧训练，`2012-02-02` 末40帧内部验证，整段 `2012-05-11` 的313帧留作测试。这个测试日期属于历史已审查过的905开发池，只是本次训练过程中的日期留出，不宣称外部独立测试。

`data/local905_mink.py` 从原始扫描解码车体坐标，按扫描路径的 SHA-256 种子固定抽取4096点；对应位姿为原图 `T_WC @ inverse(T_BC)`。已抽样核对从原始扫描重建的历史 `lidar_world` 与现存文件逐点完全相同。SAM-L 缓存由不含 GT 的清单生成，模型训练从随机初始化开始，避免用曾在测试日期预训练过的 LiDAR 权重。当前固定 batch size 16、体素大小0.2 m、TRR、Adam 初始学习率0.001；至少训练12个 epoch，内部验证 TRR 连续8个 epoch 未改善0.2% 时早停，上限80个 epoch。纯 LiDAR 对照使用同一划分、点抽样、训练预算和 SC2-PCR＋两阶段精修后端。

`tools/eval_local905_online.py` 独立运行，只读取原始测试扫描、SAM 特征、标定、训练 checkpoint 和无 GT 的划分；先将完整预测输出落盘并记录 SHA-256。`tools/eval_local905_gt.py` 在另一个进程核验 hash 与313帧分母后才读取测试位姿。失败帧明确计数；如有失败，所有帧 MPE/MOE 不报告为看似完整的有限均值。支持在验证集先检查接口，以及固定置换图像特征的对照。

首轮 `train_magic_v1` 在第35轮前停止：原融合只处理矩形 padding，抽样帧约21%的图像范围内投影点处于原数据非矩形无效视野。该轮 checkpoint 保留供审计，不能作为最终结果。`tools/prepare_local905_mask.py` 在不改变905帧划分的前提下生成绑定 `valid_mask.npy` 哈希的 `split_masked.json`。训练和在线预测共同使用该掩码；修正后的 `train_magic_v2` 从随机初始化重训，禁止从 v1 恢复。

训练日志中的 train TRR 在 `model.train()` 下计算，不能直接用来判断与 `model.eval()` 验证 TRR 的差距。`tools/audit_local905_eval_trr.py` 从同一最佳检查点完整重算训练和验证，报告原日志采用的批次均值及按帧加权均值。epoch75检查点的同口径批次均值为训练 `0.392846`（552帧）、验证 `1.007656`（40帧）；按帧加权为 `0.393168 / 0.949324`。差距存在，但日期迁移和过拟合均可能参与，不能只凭该差距区分。为满足训练到收敛，从 `last.pt` 续训时只延长上限到120轮，原8轮早停标准保持不变。

用户随后要求停止训练、使用现有最优 checkpoint 推理。`train_magic_v2/best.pt` 来自零基 epoch75，未满足8轮早停条件；续训在下一轮遇到8GB显存不足后已停止。40帧内部验证在线预测均值 MPE/MOE 为`0.238598 m / 1.523946°`。313帧整日留出测试均有输出，均值`0.788368 m / 11.466380°`，P90`1.435654 m / 9.447810°`；15个连续帧旋转误差超过90°。固定置乱SAM特征的313帧对照均值为`0.838706 m / 12.054500°`；正确图像相对置乱的配对均值差为`−0.050337 m / −0.588120°`，旋转分块区间跨零。尚无同划分纯LiDAR训练结果，不能声称优于LEADER。已准备500/52同域诊断划分，但用户停止训练后没有运行，现有checkpoint不能用于该划分的留出结论。
