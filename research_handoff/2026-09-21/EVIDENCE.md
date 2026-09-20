# 实验证据索引

下列按研究主题排列，不要求接手agent追逐小版本。数值仅在同一行所述协议内比较；像素误差和位姿误差不等价。链接指向原始结果，旧代码文件名相同不代表当前源码可复现旧运行。

| 主题 | 观察与证据 | 解释边界 |
|---|---|---|
| 早期多相机特征路线 | [原始报告](evidence/historical_notes/REPORT_prevoxel_sixview_null.md)、history中的cross_frame_visual_probe与dedode_local_probe系列 | 报告保留当时条件，不与后期32帧表直接拼接 |
| RoMa / V1 / V2 | [V1视觉](evidence/history/lscr_v1_refuv_validation8_geometry020.json)、[V1几何](evidence/history/lscr_v1_refuv_validation8_geometry_only.json)、[V2](evidence/history/lscr_v2_refuv_validation8.json)、[V2几何](evidence/history/lscr_v2_refuv_validation8_geometry_only.json)；RoMa median约6.99px、V1约4.13px且几何对照约4.14px，V2约3.19px且zero-correlation近似 | reference-specific像素修复后结果；不能把几何预测改善归为视觉相关性收益 |
| V3视觉残差 | [修复版评价](evidence/history/lscr_v3_refuv_validation8_fixed.json)：V2 3.188px、V3多尺度5.022px、stride4 5.013px、zero-correlation V3 4.780px | 特定训练与迁移设置失败，不是特征融合普遍无效 |
| XRefine | [评价](evidence/history/xrefine_validation8.json)：RoMa→XRefine7.106px，V2→XRefine3.918px，原RoMa6.992px/V2 3.188px | 冻结迁移未改善，不代表重新训练的上限 |
| 直接投影与V2-G | [诊断](evidence/history/v2g_geometry_diagnostic.json)：直接LiDAR投影median2.732px，V2 3.188px；逐点模型2.922px | 历史共同线性模型有分组赋值问题，部分诊断存在GT可见性边界；不是严格在线收益证据 |
| 视觉候选＋位姿监督 | [评价](evidence/history/native_candidate_pose_development.json)、[报告](evidence/historical_notes/REPORT_native_candidate_pose_development.md)：47训练/8开发，基线0.08742m/0.74358°、视觉候选0.10794m/1.08328°、zero-visual0.08673m/0.85456° | 257共享位姿候选，部分固定上下文广播，不等于每候选重采样query图像 |
| patch rotation-only | [评价](evidence/history/surface_patch_evaluation.json)：MPE固定0.09087m，MOE约0.90439→0.87201°，6帧求解失败 | 平移被固定，有失败且旋转区间跨零，不满足目标 |
| 早期六维patch | [评价](evidence/history/surface_patch_joint_evaluation.json)：约0.088802m/0.883869° | 后补纯几何对照及mask修复；收益不可全归视觉，也不可全归bug |
| mask / 可见性 | [评价](evidence/history/surface_patch_visibility_evaluation.json)：几何0.087866m/0.894343°、mask修复视觉0.088908m/0.884584°；可见性有2失败 | 当时删点还耦合损失尺度，不是纯观测清理比较 |
| IRLS单级轨道 | [预测](evidence/history/official_irls_B0_B1_C_20260921.json)、[评价](evidence/history/official_irls_B0_B1_C_20260921_evaluation.json)：B0≈B1，视觉平移+1.23mm/旋转−0.00905°，区间跨零 | 已退出当前基线；其目标也不同于SC2轨道，不能当仅替换初值 |
| 整块稳健与尺度修正 | history内patchrobust_20260921是旧D，视觉二次区间少1/64；aligned修正乘8，optimized只轻量化且位姿不变 | 旧D≈G不能用来否证整块稳健化；当前D看CURRENT |
| 几何传播与训练侧留出 | 当前全部原始材料在evidence/current | 模型未通过训练側覆盖率验证，微小位姿收益不证明误差归因 |

## 源码审查发现的边界（事实与待核实项）

- U使用全局4×4平面参数协方差，初值处计算每块变换，再进入Huber与soft_l1；不是逐块独立标定的可信度。
- 标定描述帧间波动；未直接估计融合地图的系统偏差；标定1.2m/在线0.8m半径不同，source/projection_xyz口径不同。
- W按原始位姿Jacobian范数比标定，m/rad混合，且缩放在Huber后；它不是最终稳健目标的严格等强度对照。
- U的visual_rmse已经经过变换，不能与D原始RMSE直接比较。
- 多数旧因果解释仍未证实：地图偏差、遮挡、纹理、标定、可观测方向都不是本快照指定的核心问题。
- 32帧反复参与设计；配对逐帧bootstrap不等于跨序列独立性保证。

历史报告中的建议仅为当时讨论记录，不对新agent构成指令。完整历史JSON清单在manifest，未逐一认证的旁支结果不得仅凭文件名提升为正式证据。
