# Query × 图像特征交叉消融

64训练/32开发帧，三配对种子2089/2090/2091，600步，六相机，完全相同two-stage后端。不是独立测试集；三个种子不构成96个独立帧。
两骨干输入均448×616；相同像素位置采样，5×5间隔14像素；中心组使用相同缓存的第12号中心token。独立128D PCA使用完全相同训练可见中心索引拟合。
层次注意力在相机内选择token，再在相机间选择；两级共享query/key。四视觉组参数量相同，但邻域组计算量较大。

| 方法 | MPE m ±种子标准差 | MOE ° ±种子标准差 |
|---|---:|---:|
| pretrained | 0.089917 ± 0.000000 | 0.902717 ± 0.000000 |
| lidar | 0.116364 ± 0.026083 | 0.936320 ± 0.026117 |
| dedode_center | 0.115823 ± 0.025637 | 0.932871 ± 0.023351 |
| dedode_center_missing | 0.114609 ± 0.025318 | 0.934164 ± 0.023660 |
| dedode_center_shuffled | 0.115479 ± 0.025819 | 0.937752 ± 0.028331 |
| dedode_patch | 0.113907 ± 0.025972 | 0.936630 ± 0.025612 |
| dedode_patch_missing | 0.112361 ± 0.025785 | 0.934277 ± 0.023712 |
| dedode_patch_shuffled | 0.113813 ± 0.025925 | 0.936117 ± 0.025711 |
| dino_center | 0.116136 ± 0.027179 | 0.937704 ± 0.020172 |
| dino_center_missing | 0.113479 ± 0.024927 | 0.934330 ± 0.025307 |
| dino_center_shuffled | 0.115002 ± 0.026101 | 0.940389 ± 0.022232 |
| dino_patch | 0.117170 ± 0.029836 | 0.939028 ± 0.020762 |
| dino_patch_missing | 0.113652 ± 0.026022 | 0.934124 ± 0.025383 |
| dino_patch_shuffled | 0.116064 ± 0.028330 | 0.936287 ± 0.020555 |

## 配对差值

| 对比（前减后，负数为误差降低） | ΔMPE m | ΔMOE ° |
|---|---:|---:|
| patch_on_dedode | -0.001916 | 0.003758 |
| patch_on_dino | 0.001034 | 0.001324 |
| backbone_at_center | 0.000313 | 0.004832 |
| backbone_at_patch | 0.003263 | 0.002398 |
| interaction | 0.002950 | -0.002434 |
| dedode_center_minus_lidar | -0.000541 | -0.003449 |
| dedode_center_minus_dedode_center_missing | 0.001214 | -0.001293 |
| dedode_center_minus_dedode_center_shuffled | 0.000344 | -0.004881 |
| dedode_patch_minus_lidar | -0.002457 | 0.000309 |
| dedode_patch_minus_dedode_patch_missing | 0.001546 | 0.002352 |
| dedode_patch_minus_dedode_patch_shuffled | 0.000093 | 0.000513 |
| dino_center_minus_lidar | -0.000228 | 0.001383 |
| dino_center_minus_dino_center_missing | 0.002657 | 0.003374 |
| dino_center_minus_dino_center_shuffled | 0.001134 | -0.002685 |
| dino_patch_minus_lidar | 0.000806 | 0.002708 |
| dino_patch_minus_dino_patch_missing | 0.003518 | 0.004904 |
| dino_patch_minus_dino_patch_shuffled | 0.001106 | 0.002741 |

优化参数量：{'lidar': 265732, 'dedode_center': 391685, 'dedode_patch': 391685, 'dino_center': 391685, 'dino_patch': 391685}
历史原始LEADER two-stage基线最大差异：0.0
missing使用该融合模型自身微调的回归头，并不恢复原始预训练模型；shuffled在每个相机和邻域位置的有效voxel间打乱图像对应，保持mask。
邻域仅为视觉上下文，未新增3D点或对应，也不假设25个像素属于同一物理表面。PCA、训练及checkpoint选择不使用开发集。
骨干对比包括预训练模型架构与各自PCA；并非只分离语义知识。5×5是相同像素采样范围，不代表两骨干相同感受野。
