# Σ-PCV：协方差塑形的点级相关体

思路：MMRegressor新增Cholesky协方差分支；Σ展开位姿无关125候选，FAR共享外观场给出后验对应／协方差，FAR按协方差几何度量求刚体位姿。

实跑：第一阶段3轮联合适配，第二阶段3轮固定LEADER训练Σ与FAR，最终stage2-03。1200Dev MPE/MOE=0.141791m/1.414982°，对照0.138862m/1.404053°；位置545帧改善、655帧变差。负结果，非充分收敛证明。

入口：`experiment/source-v2/models/sigma_pcv.py`、`sigma_far.py`；训练／推理`tools/run_sigma.py`，独立评价`tools/score_sigma.py`。第二阶段实际通过`experiment/run_repair_v3.py`续接，修正JSON list与dataset tuple身份比较；模型／损失／预算未改变。保留该修复入口及配置，不能只运行原stage2入口并忽略已知错误。

原提案见PROPOSAL.md，执行及评价见evidence/。实际训练checkpoint SHA：81e25fabe8f1fa4615b9c5326ac8d06e5e5f9f50a37f2144a652291f310486c5。
