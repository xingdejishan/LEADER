import json
import shutil
from probe_data import HERE, OUT, ARGS, run

dest = OUT/'probe'
summary = json.loads((dest/'summary.json').read_text())
rows = json.loads((OUT/'manifest.json').read_text())
hashes = dict(source={p.name:run.digest(p) for p in HERE.glob('probe*') if p.is_file()},
              checkpoint=run.digest(ARGS.checkpoint/'model.safetensors'), pca=run.digest(ARGS.pca),
              dedode=run.digest(ARGS.workspace/'rscore-assets/dedode_descriptor_B.pth'), frames=[])
for row in rows:
    hashes['frames'].append(dict(frame_id=row['frame_id'],image_sha256=run.digest(__import__('pathlib').Path(row['image'])),
                                 **{k:run.digest(OUT/k/(row['frame_id']+'.npz')) for k in ['lidar','visual_raw','mapping']}))
run.save_json(dest/'provenance.json',hashes)
o = summary['overall']
verdict = '通过固定必要条件检查' if summary['passed'] else ('未通过固定必要条件检查' if summary['viable'] else '样本不足，探针无足够判别力')
lines = ['# 新数据集跨帧视觉消歧探针', '', f'**{verdict}。** 此处仅判断固定 LiDAR 候选内的视觉消歧证据，不代表定位提升。', '',
         '## 固定范围', '',
         '原 907 帧训练集合中 905 帧精确配对，缺少原扫描的 2 帧不以邻帧替代。每日期按时间前 80% 构成参考库，共 723 帧；后 20% 共 182 帧作查询。原 32 帧开发集及 test_scene 不参与。', '',
         '所有帧重新统一生成冻结原 LEADER 编码器的 512D 特征，并在同一次前向中保存对应 cell 的原始 Cartesian 表面代表点。DeDoDe/PCA128 沿用既有权重、Cam5 投影、有效 mask 和遮挡阈值；CPU 映射 mask 与 GPU 实际视觉采样 mask 逐帧一致。没有训练、坐标修正或新定位模型。', '',
         '参考点必须具备有效图像特征。只凭 L2 归一化 LiDAR 特征余弦相似度取 16 候选，再分别按 LiDAR、正确视觉、三种帧内置乱视觉排序；候选集合完全相同。置乱在参考帧及查询帧分别执行，只改变有效描述子关联。', '',
         'GT 只用于标签及近重复视角排除，不用于检索或插入正例。正例采用真实代表点经 GT 变换后的世界位置距离 ≤0.5 m；明确负例 ≥2 m。歧义查询必须同时包含正负例且 LiDAR 前两名相似度差 ≤0.02。排除同帧、同日期相隔不足 10 秒，以及相机中心距离 <5 m 且姿态差 <15° 的近重复视角（包括跨日期）。', '',
         '## 有多少可判别样本', '',
         f'- 图像有效查询点：{o["query_points"]}。',
         f'- 合格参考库中存在空间正例：{o["geometric_positive_any"]}（{o["geometric_positive_any"]/max(o["query_points"],1):.2%}）。',
         f'- LiDAR 前 16 候选包含正例：{o["candidate_positive"]}；相对存在正例查询的覆盖率 {o["candidate_positive"]/max(o["geometric_positive_any"],1):.2%}。',
         f'- 满足歧义定义且有正例：{o["ambiguous"]}，占全部有效查询 {o["ambiguous_fraction"]:.2%}，分布于 {o["contributing_frames"]} 个查询帧。', '',
         '## 歧义子集 Top-1 正确率', '',
         '| 范围 | 点数 | LiDAR | 正确视觉 | 置乱2089 | 置乱2090 | 置乱2091 |', '|---|---:|---:|---:|---:|---:|---:|']
for name,r in [('整体',o)]+list(summary['dates'].items()):
    lines.append('| '+name+' | '+str(r['ambiguous'])+' | '+' | '.join(f'{r["accuracy"][k]:.2%}' for k in ['lidar','aligned','shuffle2089','shuffle2090','shuffle2091'])+' |')
lines += ['', '## 配对变化与边界', '',
          f'正确视觉相对 LiDAR：纠正 {o["rescue"]["aligned"]} 个、损害 {o["damage"]["aligned"]} 个。MRR、各帧分母、三组置乱纠正／损害和逐候选世界距离保存在 JSON／NPZ。', '',
          f'正确视觉减 LiDAR、减三种置乱均值的帧重采样 95% 区间：{summary["paired_frame_bootstrap95"]}。相邻帧仍可能相关，不能视为独立场景泛化置信区间。', '',
          '判别力规则固定为至少 200 个歧义正例查询、覆盖至少 20 帧和两个日期。通过规则要求正确视觉整体及每个至少 50 点的日期均胜过 LiDAR 和三个置乱种子，且两个配对区间下界大于零。没有根据结果调整 16 个候选、正例距离、歧义或排除阈值。', '',
          'PCA 已在这些训练日期图像上拟合，本实验不是视觉预处理层面的未见数据盲测。逐 cell 的一个真实表面代表点具有稀疏性；0.5 m 匹配只是空间同位标签，不等于人工确认的同一物体表面身份。近重复排除可能降低重叠，但未据结果放宽。阴性结论仅约束当前描述子、采样和数据条件。', '',
          '## 复现', '',
          '依次执行 probe_data.py manifest、probe_data.py lidar（egonn118）、bash probe_visual.sh（rscore-l）、probe.py、probe_report.py（egonn118）。缓存根目录 /home/zhang/crossframe-visual-probe；输入清单、冻结协议和逐帧指纹随结果保存。只有诊断结果上传仓库，原始数据及大型特征缓存在本地。']
(dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copytree(dest,HERE/'results/crossframe_probe',dirs_exist_ok=True)
print(verdict)
