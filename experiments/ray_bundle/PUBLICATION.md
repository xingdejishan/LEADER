# 公开版本与运行配置

本分支是独立的源码快照，使用单个根提交，不继承私有运行记录所在的提交历史。保留上游代码、LICENSE和署名；修复核心来自研究实现版本，模型、几何、标签生成及训练损失的文件SHA见results.json。

公开版本仅将机器专属的项目目录、虚拟环境和功能检查输入路径改为显式配置。训练过程中使用的冻结源码、数据、权重与进程没有修改；公布的结果不是用这些配置改动重新训练产生的。旧运行的原始源码哈希不能直接用公开版本替换，原有绑定检查仍会拒绝不匹配。

在已有、依赖齐全的Linux虚拟环境中运行，显式设置：

```bash
export RAYBUNDLE_ROOT="$(pwd)"
export RAYBUNDLE_ENV_PREFIX="$CONDA_PREFIX"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`RAYBUNDLE_ROOT`必须是当前用户拥有的独立项目目录；源码与运行输出应位于其子目录。`RAYBUNDLE_ENV_PREFIX`必须指向正在使用的已有解释器环境，非Conda环境也可直接设置其路径。程序不会创建或修改其他人的环境。路径归属、GPU显存、RSS、可用内存和磁盘下限检查均保留。

`raybundle.gpu_probe`现通过`--data-root`、`--checkpoint`、`--map`和`--output`接收功能检查输入。其他训练入口沿用原CLI参数；新修复训练入口为`raybundle.repair_train`，共享验证入口为`raybundle.validation_pool`。注册表由使用者提供自己的输入、输出和源码路径，不包含任何实际服务器配置。

本次未包含服务器账号、认证信息、连接地址、专属绝对路径、GPU UUID、实际进程号、原始终端日志或私有审计日志。实验数据采用字段白名单导出，保留数值、训练条件和制品哈希。

这份说明仅覆盖当前发布分支；仓库中此前已经存在的其他分支或历史不由本快照改写。
