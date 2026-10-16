# 正式V8：几何正对应监督微调的LightGlue

正式模型为best-lightglue.pt、第200步；不是第300步终版权重。使用GT位姿与LiDAR重投影构造正对应，冻结SuperPoint、忽略未标注点；标签不来自LightGlue预测。

这两个版本都是 LightGlue 匹配器，SuperPoint 冻结；不是 LEADER 主干版本。
训练/评分数据、NCLT图像、LiDAR地图和SuperPoint权重不随本快照分发。

## 下载与加载

在已有 Python/PyTorch 环境中使用。匹配器推理依赖 `torch`、`numpy`；Release下载脚本还依赖`requests`。
当前仓库为私有仓库，下载脚本从环境变量`GH_TOKEN`或`GITHUB_TOKEN`读取有本仓库读取权限的凭据。

```bash
python download_weights.py
```

`checkpoint.json`绑定权重名称、Release标签、大小和完整SHA-256；下载及加载均核对SHA。
Release保存服务器原始checkpoint字节，没有重存权重。核心LightGlue源码来自实际运行的vendor，
`lightglue/__init__.py`仅改为延迟导出，避免特征输入推理强制加载其他提取器；核心数学源码未改。
LightGlue许可见`third_party/LightGlue-LICENSE`。

## 输入和运行

每张图片一个`.npz`，包含：

- `keypoints`: `[N,2]`或`[1,N,2]`，浮点像素坐标。
- `descriptors`: `[N,256]`或`[1,N,256]`，冻结SuperPoint描述子。
- `image_size`: `[2]`或`[1,2]`，顺序为宽、高。

提特征时保持与关键点坐标一致的图像预处理。历史NCLT实验使用RGB、旋转270度、最长边至多560、
双三次缩放、中心裁剪至14倍数；坐标在裁剪后的图像上。便携入口接收已提取特征，不擅自重采样。
Python接口为`model.load_model`、`model.load_features`和`model.match_features`。

历史训练脚本与配置保存在`experiments/`，保留实际运行的源文件及原始路径，便于审计；
这些历史driver需要原NCLT冻结资产和路径配置，不是无需数据即可一键重训的承诺。
新的根目录加载/匹配入口不依赖服务器绝对路径，也不会自动下载原始LightGlue权重。

## 评测解释

统一对照为48个2012-05-11 Query，同组3/10参考（不含Query），3张为10张的子集，阈值0.1。
正确比例仅以具备双侧稀疏LiDAR支持的可判定预测为分母：3.5px绑定、5px重投影、0.6m深度一致。
未知预测不计错，不能将该比例称完整真实匹配Precision，也不能直接与旧24对稀疏标签19.51%混比。
这里是已使用开发评测集合上的后续受控对比，不是全新独立盲测；没有PnP/MPE/MOE定位成绩。

| 模型 | 参考数 | 可判定正确比例 | 正确对应总数 | 每Query去重正确点 |
|---|---:|---:|---:|---:|
| 原V8 | 3 | 22.103% | 1175 | 19.021 |
| 强交互V8 | 3 | 24.092% | 146 | 2.417 |
| 原V8 | 10 | 19.728% | 2784 | 32.542 |
| 强交互V8 | 10 | 26.253% | 419 | 5.229 |

强交互版本未同时提高比例与正确数量，当前不建议替代正式V8。固定系数为1不等同已经学会有效协同。
完整结果、训练和协议在`results/`，源码逐SHA在`source_manifest.json`。

```bash
python match.py --checkpoint weights/best-lightglue.pt --query query.npz --references ref1.npz ref2.npz ref3.npz --output matches.npz
```
