# 本地训练环境恢复

2026-09-25，Ubuntu WSL；解释器`/home/zhang/.venvs/leader-magic/bin/python`，Python3.9.25、torch2.0.1+cu118、CUDA11.8、RTX4060 Laptop 8GiB。保留继承的egonn118环境，依赖安装目标为leader-magic虚拟环境。

MinkowskiEngine0.5.4源码`/home/zhang/src/MinkowskiEngine`，commit`02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`。原有未提交修改仅补充Thrust的execution_policy/remove/unique头文件，已保留。此前构建日志显示编译被中断，没有可导入的安装包。

使用CUDA11.8、gcc/g++11、`TORCH_CUDA_ARCH_LIST=8.9`、OpenBLAS、MAX_JOBS=2构建wheel，安装到上述虚拟环境。第一次启动因setup.py调用PATH中的系统pip触发PEP668而停止；把PATH指向虚拟环境后构建成功，没有绕过系统包保护。

Wheel：`/home/zhang/src/MinkowskiEngine/dist/minkowskiengine-0.5.4-cp39-cp39-linux_x86_64.whl`，SHA256 `01e01d17b5a045fc627ee4f76626d63b85c9764da96d41fd1fa0f54738b7fe8b`。

补齐缺失Open3D、matplotlib、pandas、h5py、transforms3d、safetensors、protobuf、huggingface-hub及依赖；完整版本见environment.txt。`pip check`无依赖冲突，torch/ME及训练入口成功导入，CUDA可用，GPU稀疏坐标错序/梯度测试通过。旧MaGiC10项测试与新模块6项测试通过。新环境不宣称与历史二进制逐位等价，因此本轮重新生成L0，所有条件在同一环境比较。

构建日志保留在`/home/zhang/minkowski_restore_20260925.log`和`/home/zhang/minkowski_restore_20260925_retry.log`。C盘恢复后空闲约22GiB；无需删除旧权重或重建WSL虚拟盘。
