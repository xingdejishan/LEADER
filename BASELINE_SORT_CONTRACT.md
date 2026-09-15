# 服务器基线并列排序约定

## 生效入口

`run_mink.py` 导入 `models.sc2pcr.Matcher`。种子排序、第一层共识集排序、第二层共识集排序统一调用 `models.server_sort.server_argsort`。

基准是服务器 PyTorch 1.12.0+cu116 的 CUDA 默认排序行为。此兼容实现复现服务器的位比较排序网络，包括补齐长度和分数相等时的交换规则；不是简单稳定排序，也没有更改匹配参数、top50%集合、候选评分或精修算法。

## 为什么不能直接用 stable=True

同样40个并列点，稳定排序选择前20行，而服务器旧版CUDA排序可能选择另一组行。直接改成稳定排序会改变服务器基线。这里固定旧服务器的行为，不宣称算法精度改进，也不宣称输入排列不变性。

## 支持范围

有限浮点分数。长度不超过2048时显式复现服务器分组宽度32/128/1024/2048的排序网络；更长时使用稳定排序，对应服务器的大数组路径。非法分数立即报错。

评分仍在原设备计算。排序索引通过CPU NumPy生成并返回原设备，因此不要求更换PyTorch、CUDA或GPU，但会增加CPU传输与排序耗时。兼容规则只解决相同分数输入的排序行为，不保证网络非align前向或非并列数值跨设备逐位一致。

## 回归检查

从本目录执行：

```bash
python -m unittest discover -s tests -p test_server_sort.py -v
```

固定样本来自未修改的服务器原排序，而不是由待测实现生成：

- 20种长度，包括32、128、1024、2048前后边界，升降序共40组；每组包括全并列、严格递增和重复随机分数。
- 12帧、降序和random_2089、三处排序，共72组真实SC2-PCR中间数据。

不要为了使新方法更优而变更这份参考。若有意更换并列规则，必须作为新基线，并在两端重测。

## 上游依据

- https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/Sort.cu
- https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/SortUtils.cuh

