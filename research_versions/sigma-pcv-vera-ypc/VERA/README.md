# VERA：Visual Endpoint Re-Anchoring

思路：图像在LiDAR实测局部表面选择对应端点，视觉／位移特征帮助MMRegressor预测地图坐标；最终沿用SC2-PCR和Tukey/SVD。

代码版本：`full-standard-v2/`为原版四轮入口；`cam5-support-v1/`加入真实回波支撑；`cam5-support-fov-v1/`加入官方FOV／遮挡；`cam5-repair-v2/`加入零初始化回退、离散端点、地图坐标运输、质量监督及保护入池。各目录保留对应运行／训练／预测／评价代码；source/是当时冻结的LEADER依赖源码。

1200Dev MPE/MOE：原e4=0.285464m/1.820229°；support e30=0.230667m/1.688297°；FOV追加e10=0.220302m/1.675971°；repair e1=0.422104m/2.173120°。历史纯LEADER微调=0.179498m/1.624231°。未胜对照。

最新状态：用户已停止所有VERA续训。repair原计划五轮仅完成首轮正式Dev评价，不能宣称完成五轮或完整收敛。此上传只归档源码，不恢复实验。原提案与当前repair实现有区别：repair使用原预测对应估计的内部旋转运输坐标，不是完全不使用任何预估旋转。

证据：evidence/四份report、训练计划、用户stop回执及原验收记录。历史对照训练范围／坐标拼接不同，结果不能作为隔离模块消融。
