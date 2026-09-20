# 任务条件、评价边界与数据可用性

## 已有输入

- 单个查询LiDAR扫描及其同步相机视图、内外参、mask。
- 训练帧建立的参考地图、图像观测及已声明使用的训练参考位姿。
- 冻结LEADER预测缓存；当前基线是SC2-PCR＋两阶段全池精修＋中心还原。
- 当前研究为NCLT相关单帧地图定位；已用清单来自2012-02-18，64训练帧、32开发帧；早期部分实验47/8，不可混为同一分母。

没有授权把相邻查询帧、轨迹、IMU或查询GT作为在线输入。架构与融合层级不作固定要求；若研究建议需要新增输入，应明确说明条件变化。

## 评价约束

查询GT只能由独立离线评价读取，不能出现在在线查询缓存中；训练参考GT用于地图是已声明条件。单帧唯一输出；不得依据GT挑候选、逐帧择优回退、删除难帧或以baseline/challenger gate包装成绩。失败、空观测、超时计入完整分母。报告MPE/MOE、P90、改善/损坏帧、配对区间与完整耗时；区间按帧计算不代表序列独立。输出落盘并记SHA-256后评价，评价不改输出。

评价器在有失败时会对有限值计算均值并另报failure_count，不能忽略失败数、只取均值；本轮完整32帧无失败。不得反复扫开发集参数并只报最优。历史特定路线的准入条件属于历史协议，不是新架构必须沿用的形式。

## 本快照实际包含什么

- 当前及历史原始JSON、192行统一逐帧表、固定代码、相关测试、运行参数和原始哈希清单。
- 不包含原始图像/点云、mask、标定文件、网络大权重和在线预测数组；这些存在本地/远程实验环境，GitHub上的agent可以完整审阅材料，但不能仅凭此仓库声称独立复跑。
- 未上传SSH配置、凭据、私人操作脚本。结果中原有路径仅保留实验溯源意义，不是新机器可以直接访问的资源。

## 外部资源角色（路径来自正式protocol）

| 资源 | 既有路径/作用 |
|---|---|
| 帧清单 | /home/zhang/leader-image-gate-multicamera/all_views.json，视图路径与split |
| 查询在线缓存 | results/official_irls_cache_20260921，文件夹历史命名不代表使用IRLS求解；读取source/prediction/center |
| 参考LiDAR与评价GT | /home/zhang/leader-image-gate/lidar；在线仅允许训练参考部分，查询GT仅评价 |
| 投影映射 | /home/zhang/leader-image-gate/projection_audit/mapping |
| 参考地图缓存 | results/official_surface_reference_20260921.npz |
| 冻结网络权重 | 原清单记录的 research/image_gate_checkpoint/model.safetensors（工作区路径）；权威身份是清单中的checkpoint SHA-256 |

具体路径字段、参数和地图观测数在current_protocol.json，checkpoint与输入哈希在evidence/current的原始清单。重新运行需要用户提供以上资源的可访问映射；本次未重新获取远程原始数据，也不将清单声明当成重新独立认证。

## 阅读与复跑

先从README完成研究分析，无需安装环境。若后续获授权复跑：依据code/LEADER/README.md设置原项目依赖，提供上述资源，向surface_patch_refinement.py显式传递正式protocol中的参数、geometry model、full-pool路径与输出路径；不要用缺参默认命令。evidence/current中的模型可用作冻结输入，但旧模型标定有效性已未通过，不是新的可信先验。
