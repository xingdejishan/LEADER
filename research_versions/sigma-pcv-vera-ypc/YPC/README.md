# YPC：Yaw-Phase Closure

思路：每帧共享一个右body-z偏航角；固定真实raw回波和对应Cam5像素，只旋转地图查询点，查询冻结V2/FAR场并训练图像／场adapter与点权重，整帧相位CE监督。

实跑：±4°、0.4°步长、21格；1200Train按index%5分960fit/240cal；冻结LEADER/FAR主体，固定4轮3836更新，取e4。240cal相位RMSE0.955345°，零修正0.641481°（差48.93%）；Top1 24.58%，常数0°29.58%。预注册phase gate失败。

入口：`experiment/ypc_head.py`与`experiment/entry.py`；执行链`experiment/supervisor.py`，原环境配置`experiment/config.json`；结果evidence/。checkpoint e4 SHA：4b04e077e5a293bc652b6472e5d4bcaad33c4dff45c5a7ecf19953b55a1d53a6。

这是粗相位诊断，不是完整定位结果；fine17、完整Dev位姿评价、前端相位条件化及SO3训练均未执行。旧V2基线MPE/MOE不写成YPC成绩，四轮也不证明充分收敛。
