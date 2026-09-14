# LEADER 局部图像门控：本地必要条件实验

**当前完整交接入口：[HANDOFF.md](HANDOFF.md)。** 包含主baseline、①②③④、修复后的投影、最新诊断、所有报告索引及Release权重/缓存下载方式。下文为最初低覆盖率实验的历史协议，不能替代修复后的配置；其中voxel中心投影已被真实Cartesian代表点采样替代。

基于 research/rscore-l-local 的 f84b5f1 新建 research/leader-image-gate；原 LEADER 编码器、MMRegressor、TRR 和 Matcher 源码保持不变。首轮冻结官方模型，训练约 8 万参数的门控残差。该实验判断的是冻结主干条件下的局部补充能力，不能排除后续联合微调有不同结果。

## 假设与依据

LEADER 的 LiDAR 局部几何可能存在重复结构歧义；TRR 能抑制不可靠对应，但无法凭空提供纹理。DeDoDe 将检测与描述解耦，可直接使用其稠密描述器输出：[官方实现](https://github.com/Parskatt/DeDoDe)。投影图像输出到点云已有 [PointPainting](https://openaccess.thecvf.com/content_CVPR_2020/html/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.html) 的检测任务证据；这不能直接证明 LEADER 定位会提升。

假设：在标定和可见性正确时，局部纹理能提供 LiDAR 特征之外的条件信息。f'=f+m sigmoid(g([LN(f),LN(v)])) W2 GELU(W1 LN(v))，W2 零初始化；无图像/遮挡/越界 m=0，严格返回原特征。门控没有可靠性概率校准含义。冻结骨干使零视觉退回原版性质在训练后仍成立。

PCA 是仿射投影，在有效内部采样位置与双线性插值可交换；采用 DeDoDe-B 的 256 维稠密输出、已有训练日期拟合的 128 维 PCA。仅使用外参和当前扫描投影；不使用 GT 校正图像对齐。原版默认关闭 level correction；投影模块仍验证逆 correction 的正确性。voxel 中心必须从 LEADER 的极坐标网格恢复到笛卡尔坐标。可见性以真实扫描构造 4 像素 Z-buffer、0.5m 深度容差和真实黑边掩码估计；这不是独立标定验证。

## 冻结协议

本地完整配对扫描共 301 帧，均来自官方训练日期 2012-02-18。按时间排序：前 201 帧均匀取 64 帧训练，间隔 20 帧，后 80 帧均匀取 32 帧验证。原 907 图 PCA 来自其他训练日期。验证数据历史上已用于其他方向开发，不称独立盲测。不会用 LiDAR-world 标签伪造缺失扫描或强度。

固定种子 2089、600 步、Adam 1e-4、每步随机 1024 voxel，使用原版 TRR；原版模型、对齐图像门控和打乱可见 voxel 图像对应的同预算门控作对照。打乱臂只改变对应关系，保留图像描述子分布、可见掩码和容量；验证另测训练后对齐模型的错误对应、缺图像。固定官方 Matcher 参数和原置信度 top-50% 规则。

通过条件：32 帧验证中 1m/5deg 的 rescue 大于 damage，平均平移降低至少 5%，平均旋转和 p95 平移不恶化超过 5%，且对齐臂平均平移优于打乱臂。任何条件不通过即停止当前冻结主干方案，不测试集调参。通过才扩充开发验证和独立种子；本地实验不能代替 104735 帧全集验证。

运行入口为 run.py 的 manifest、lidar、visual、train、evaluate；lidar/train/evaluate 使用本地 egonn118 稀疏卷积环境，visual 使用 rscore-l 环境。check.py 验证几何、PCA 采样交换、梯度和退回原版性质。输出在 /home/zhang/leader-image-gate，protocol.json 和 manifest.json 先于训练生成。
