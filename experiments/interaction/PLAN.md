# LEADER主干内双向多模态交互：最小框架

用户2026-09-25授权架构级创新、预训练LEADER初始化和联合微调。停止末端重力/法向等小修正，不执行未实现的upright-constraints。目标仍为同313帧MPE或MOE降低10%，另一项不恶化；40和313均为开发集。

## 从证据到机制

旧MaGiC/SurfaceToken MPE改善4.42%但旋转几乎不变；三维融合解码器仍在已完成的几何编码后融合，138步未充分验证收敛能力。误差memory、共享刚体和表面配准均失败。它们不能证明主干交互无效，因为没有让图像参与RPGE后续几何表示的形成。

筛选三个原语：DeepFusion（CVPR2022，https://openaccess.thecvf.com/content/CVPR2022/html/Li_DeepFusion_Lidar-Camera_Deep_Fusion_for_Multi-Modal_3D_Object_Detection_CVPR_2022_paper.html）强调增强恢复与深层对齐，已经是本地MaGiC基础，不重命名为创新；DeepInteraction（NeurIPS2022，https://proceedings.neurips.cc/paper_files/paper/2022/file/0d18ab3b5fabfa6fe47c62e711af02f0-Paper-Conference.pdf）保留模态表示并双向交换，选择其表示交互原语；PTv3（CVPR2024，https://arxiv.org/abs/2312.10035）高效扩大点集交互范围，但替换原主干会损失checkpoint兼容性，当前不选。

本框架不是上述论文精确复现，也不宣称已证原创。两处交互构成最小闭环：RPGE第2和第4编码阶段（64/256通道）将真实占据点支持的几何特征投影写入64通道图像状态，经3x3图像卷积更新后，由LiDAR query注意力读取对应支持点图像特征，残差更新几何流；图像状态持续传至下一交互阶段。原RPGE后续卷积、解码器和MMRegressor因此处理视觉参与形成的特征。图像没有独立坐标/位姿头，几何仍是唯一定位主路径。

复用SurfaceToken的每体素最多8个真实支持点是既有代码，不是新贡献。新增信息流为几何写图像、持久图像状态、图像回写中间几何表示及后续几何传播。相较旧单点门控/晚期MMA/新增末端稀疏卷积，改变主干内计算路径。只做这一框架，不加入额外监督、参考地图、记忆库或新的位姿后端。

## 冻结实现与训练

official_l0.pt完整加载原参数；新增回写层零初始化以保持初始输出，训练后无幅度上限。SAM预训练特征冻结；RPGE/MMRegressor联合微调，BN统计冻结适应microbatch1，新增使用GroupNorm/LayerNorm。AdamW：原参数1e-5、新参数1e-4、weight_decay1e-4，effective batch8、micro1，seed37。552训练帧全点数、原TRR监督和固定SC2-PCR＋两阶段全池后端。

至少40轮，每5轮在完整40帧选模；连续30轮无0.1%验证目标改善且至少两次LR下降才停止，160轮未平台则同配置延长。优先两均值不恶化的checkpoint，再按较佳归一化指标选；无合格者按预定归一化平均选用于报告失败。选择后冻结预测再独立GT评价313一次。仅保留best与latest完整训练状态控制磁盘，所有候选评价及权重哈希保留。

必要验证：真实训练批次初始LEADER数值一致、非零跨模态梯度、有效完整反传及显存；实际训练与query投影对齐、原始内参到SAM缩放的独立计算、人工非单位增强的逆变换不变性。当前Local905返回单位T_corr，没有几何随机增强；不得把缺少增强恢复解释成已发现本地错误。投影自洽不等于物理外参/时间同步绝对无误，保留遮挡/同步误差风险。

预期价值：判定主干内交互能否把互补信息转成更强坐标回归，而不是修补输出。达到明确收益则沿框架改进；严重退化也如实接受并据验证/训练行为决定修改机制。时间由真实smoke测得后记录，不预设短训即失败。最终收益含联合微调因素，未做对照不能全部归因图像或单个模块。
