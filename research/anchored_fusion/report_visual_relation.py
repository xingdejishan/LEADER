import json
import shutil

import numpy as np

import visual_relation as v


def main():
    summary=json.loads((v.OUT/'summary.json').read_text())
    verification=json.loads((v.OUT/'training_verification.json').read_text())
    relation=json.loads((v.OUT/'relations.json').read_text())
    lines=['# H1：视觉筛选关系监督的受控实验','',
        '**本轮未建立视觉筛选的增量价值，H1在本协议下未通过。** V相对B0的平均位置误差观察改善约2.87mm，但普通TRR微调B1已经获得约2.88mm改善；V反而比B1高0.00768mm，较S低0.01808mm，两项配对区间均包含零。不能把继续微调本身的改善归功于视觉。','',
        '三种微调的平均旋转误差均高于原始B0，位置P95改善；按日期看，位置改善集中于2月2日，1月22日和5月11日均退化。V相对B0的位置差值区间也包含零，尚未建立跨轨迹一致的净收益，继续保留原始LEADER为主baseline。','',
        '固定578/145/182划分；单种子2089；B1/V/S各完整100epoch、7300次更新。182帧为已反复接触的开发评估，不是盲测。所有定位均为LiDAR-only，原RPGE冻结，MMRegressor结构不变、权重微调。','',
        '## 最终定位','',
        '| 条件 | 选中epoch | 位置均值 cm | 旋转均值 ° | 位置P95 cm | 1m/5°成功 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ['B0','B1','V','S']:
        r=summary[arm]; m=r['metrics']; epoch=r.get('selection',{}).get('epoch',0)
        lines.append(f"| {arm} | {epoch} | {100*m['mean'][0]:.5f} | {m['mean'][1]:.6f} | {100*m['p95'][0]:.5f} | {m['successes']}/{m['count']} |")
    lines+=['','B0为未训练原模型；B1仅TRR；V为TRR+正确视觉筛选关系；S为TRR+分层置乱筛选标记。B0输出与既有182帧baseline逐帧完全一致。所有检查点仅按145帧位置均值选取，epoch0参与候选。','',
        '| 配对差值（V减对照） | 均值 mm | 轨迹块95%区间 mm |','|---|---:|---|']
    for name,r in summary['paired'].items():
        lines.append(f"| {name} | {1000*r['mean']:.5f} | [{1000*r['ci95'][0]:.5f}, {1000*r['ci95'][1]:.5f}] |")
    lines+=['','负差值表示V较好。按日期内连续轨迹块bootstrap10000次，固定seed271828；区间仅描述当前开发集，不是跨种子不确定性。','',
        '## 逐日期定位','', '| 日期 | 条件 | 位置均值 cm | 旋转均值 ° | 位置P95 cm |','|---|---|---:|---:|---:|']
    for date in summary['B0']['dates']:
        for arm in ['B0','B1','V','S']:
            m=summary[arm]['dates'][date]
            lines.append(f"| {date} | {arm} | {100*m['mean'][0]:.5f} | {m['mean'][1]:.6f} | {100*m['p95'][0]:.5f} |")
    lines+=['','## 同一现有隐藏层的机制读数','',
        '查询和参考都通过各条件自己的当前MMRegressor；不把原RPGE的13.79%当作隐藏层基准。完整16候选保留灰区，float64余弦、1e-7并列容差、固定原顺序优先。','',
        '| 集合 | 条件 | 点数 | 正例Top1 | 正例率 % | N→P纠正 | P→N损害 | P→G | G→P | 灰区Top1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['all','ambiguous','matcher','ambiguous_matcher']:
        for arm in ['B0','B1','V','S']:
            r=summary[arm]['mechanism'][group]; t=r['transitions']
            lines.append(f"| {group} | {arm} | {r['count']} | {r['P']} | {100*r['P']/r['count']:.5f} | {t['-1:1']} | {t['1:-1']} | {t['1:0']} | {t['0:1']} | {r['G']} |")
    lines+=['','所有转换以上述B0隐藏表示为参照。N→P才是明确错误纠正，灰区变化不混入该分母。matcher指原B0实际候选索引，微调后候选集合允许依原规则变化。','',
        '## 教师、训练与验证','',
        f"- 固定关系池{relation['pool']:,}条；V/S各选{relation['selected']:,}条，每臂关系曝光1,868,800次。标签来自训练GT，视觉只做筛选。",
        f"- V/S重合{relation['overlap']:,}条，占每组选择{100*relation['overlap_fraction']:.4f}%；因此两组并非完全不同的训练集合。S保留日期/难度层内的视觉选择比例，不是彻底消除一切视觉偏好。",
        '- 独立检查重新枚举全部候选组合、核对Cartesian距离及视觉选择标记、逐层核对选择数量；训练与参考仅来自578帧。',
        f"- 微调现有回归头共{verification['runs']['V']['parameter_count']:,}个参数。关系损失更新pred_out[2]之前的现有隐藏路径，最终线性层仍由TRR更新；没有新增投影头或推理图像模块。",
        '- 三臂最终帧序RNG、关系索引日程RNG、排列及游标一致；V/S所有100epoch都检测到辅助损失向现有隐藏权重的非零梯度。',
        '- 原缓存、16候选、投影mask及coarse voxel定位监督保持不变；没有使用开发GT生成关系或调节超参数。',
        '', '| 条件 | 训练用时秒（不含选模和保存） | 首轮TRR | 末轮TRR | 首轮关系损失 | 末轮关系损失 |','|---|---:|---:|---:|---:|---:|']
    for arm in ['B1','V','S']:
        logs=json.loads((v.OUT/arm/'training.json').read_text())
        lines.append(f"| {arm} | {sum(r['seconds'] for r in logs):.1f} | {logs[0]['loss'][0]:.6f} | {logs[-1]['loss'][0]:.6f} | {logs[0]['loss'][1]:.6f} | {logs[-1]['loss'][1]:.6f} |")
    passed=all(summary['V']['metrics']['mean'][0]<summary[arm]['metrics']['mean'][0] for arm in ['B0','B1','S'])
    lines+=['','## 结论边界','',
        'V满足三项平均位置误差比较，仍需结合区间、旋转/P95和机制读数判断是否值得多种子复验。' if passed else 'V未同时优于原始B0、普通微调B1和置乱教师S，未通过预注册的净定位收益判据；保留原始LEADER作为主baseline，不追加调参或训练。',
        '在6937个固定歧义查询上，B0隐藏层正确Top1为1395个，B1/V/S均为1233个；V没有相对B1或S增加正确数量。V明确N→P纠正56个、P→N损害132个，其余灰区转换见表。关系损失确有非零梯度，但本轮没有建立预期的关系区分机制。近似的结果也不构成统计等效证明。',
        '本轮只检验冻结RPGE、当前MMRegressor微调范围以及固定关系筛选协议，不能推出视觉信息必然无法由LiDAR学习，也不能证明所有训练期蒸馏无效。排序变化、训练损失和视觉/置乱差异不能替代净定位收益。','',
        '## 复现','',
        '`visual_relation.py prepare` → `verify_visual_relation.py prepare` → `visual_relation.py train` → `verify_visual_relation.py final` → `visual_relation.py assess` → `report_visual_relation.py`。',
        '运行环境和原始缓存沿用此前研究；本目录提供协议、输入hash、固定关系、三组已选回归头、逐帧定位和排序、训练日志及验证。原始大体积LiDAR/图像缓存仍在本地，不包含在此结果目录。']
    (v.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    target=v.e.HERE/'results/visual_relation'; target.mkdir(exist_ok=True)
    for name in ['REPORT.md','protocol.json','relations.json','relations.npz','inputs.json','relation_verification.json','training_verification.json','summary.json','baseline_internal.json','evaluation_masks.npz']:
        shutil.copy2(v.OUT/name,target/name)
    for arm in ['B0','B1','V','S']:
        for suffix in ['_development.json','_ranking.npz']:
            shutil.copy2(v.OUT/(arm+suffix),target/(arm+suffix))
        if arm!='B0':
            folder=target/arm; folder.mkdir(exist_ok=True)
            for name in ['best.pt','selection.json','training.json','complete.json']:
                shutil.copy2(v.OUT/arm/name,folder/name)
            for file in (v.OUT/arm).glob('internal_*.json'):
                shutil.copy2(file,folder/file.name)
    print('REPORT',target,flush=True)


if __name__=='__main__':
    main()
