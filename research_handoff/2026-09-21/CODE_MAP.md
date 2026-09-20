# 固定源码入口

源码在code/LEADER，保留相对目录与import关系；额外依赖full_pool也放在code/glace-local对应位置。不要以仓库外层工作分支的变化替换本快照代码。

| 内容 | 文件与函数 |
|---|---|
| 当前基线、patch选择、D/U/W求解 | [surface_patch_refinement.py](code/LEADER/research/prevoxel_multiview/surface_patch_refinement.py)：pose_from_baseline_online、select_surface_patches、refine_pose |
| SC2-PCR | [models/sc2pcr.py](code/LEADER/models/sc2pcr.py)：Matcher.estimator |
| 两阶段全池 | [full_pool_robust_v1.py](code/glace-local/code/tools/full_pool_robust_v1.py)：full_pool_refine |
| 训练侧平面模型 | [calibrate_surface_geometry_uncertainty.py](code/LEADER/research/prevoxel_multiview/calibrate_surface_geometry_uncertainty.py) |
| 留出诊断 | [diagnose_surface_geometry_uncertainty.py](code/LEADER/research/prevoxel_multiview/diagnose_surface_geometry_uncertainty.py) |
| GT隔离评价 | [evaluate_surface_patch_refinement.py](code/LEADER/research/prevoxel_multiview/evaluate_surface_patch_refinement.py) |
| 参考地图、投影 | local_visual_refinement_roma.py、oracle_pose_refinement.py（同目录） |
| 特征与像素精修历史 | lscr_refinement.py、lscr_v2_offset_head.py、lscr_v3_visual_residual.py、xrefine_adapter.py |
| 学习位姿候选历史 | visual_pose_candidates.py |
| 早期特征路线 | leader_model.py、fusion.py、image_feature.py、dataset_hook.py及对应probe脚本 |

当前runner记录invalid方向统计；与之前未记统计的U数值完全一致。历史结果引用的旧runner哈希不一定对应当前代码，历史源码未全部保存，不能假称每轮都可精确复跑；当前诊断runner及配套脚本按最新哈希清单核验。

代码包含相关测试原件；本次只整理证据，不修改算法，不声称重跑了GPU实验。源文件中的旧绝对路径和默认值是实验痕迹，执行时以current_protocol.json与外部数据映射为准，特别是默认patch_robust_scale不等于正式0.41133925。
