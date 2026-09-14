import json
import numpy as np
from diagnose_fusion import OUTPUT, ROOTS
from pathlib import Path
import run


def main():
    results = json.loads((OUTPUT/'summary.json').read_text())
    old, current = results['old'], results['line1']
    valid, selected = current['val']['valid'], current['val']['selected_before']
    rejected_count = valid['count']-selected['count']
    excluded_before = (valid['mean_error_before']*valid['count']-selected['mean_error_before']*selected['count'])/rejected_count
    excluded_after = (valid['mean_error_after']*valid['count']-selected['mean_error_after']*selected['count'])/rejected_count
    coordinates = ['| 固定点集，修复后的① | 点数 | 融合前坐标误差m | 融合后坐标误差m |', '|---|---:|---:|---:|',
        f"| 全部图像有效点 | {valid['count']} | {valid['mean_error_before']:.4f} | {valid['mean_error_after']:.4f} |",
        f"| 原LEADER top50%中的图像有效点 | {selected['count']} | {selected['mean_error_before']:.4f} | {selected['mean_error_after']:.4f} |",
        f'| 原LEADER未入选的图像有效点 | {rejected_count} | {excluded_before:.4f} | {excluded_after:.4f} |']
    pose_table = ['| ① 固定权重的诊断干预 | 平移均值m | 旋转均值° |', '|---|---:|---:|']
    for label,key in [('原始LEADER','reference'), ('完整融合','fused'), ('只替换坐标，保留原可靠度','coordinates_only'),
        ('保留原坐标，只替换可靠度','confidence_only'), ('随机打乱有效图像描述子','wrong_random'), ('训练集均值描述子','constant_image'), ('GT逐点选择较好坐标（不可部署）','oracle_coordinate_choice')]:
        values=current['pose'][key]['mean']
        pose_table.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    all_methods = ['| 方法 | 训练集高可靠点误差变化m | 验证集高可靠点误差变化m | 验证门控收益AUC |', '|---|---:|---:|---:|']
    for name,label in [('old','修复前'),('line1','①'),('line2','②'),('line3','③'),('line4','④')]:
        r=results[name]
        all_methods.append(f"| {label} | {r['train']['selected_before']['mean_error_delta_m']:+.6f} | {r['val']['selected_before']['mean_error_delta_m']:+.6f} | {r['val']['selected_before']['gate_helpfulness_auc']:.4f} |")
    report='\n\n'.join([
        '# 融合修复后表现下降：冻结权重诊断',
        '结论：投影修复后，模型并非完全没有利用图像。①/③/④都降低了图像可见点的总体场景坐标误差，但收益集中在原LEADER未选中的低可靠点；对原本已准确、参与求姿态的点，训练集上有收益，验证集上反而略有伤害。当前门控无法可靠区分这些好坏修正。这比“图像本身不可用”更符合现有证据。',
        '## 检查范围与对照',
        '不训练、不修改任何已有模型，不调投影或门控阈值。检查修复前版本及①②③④，同一64帧训练、32帧开发数据。每个方法重放最终姿态结果，与原先保存的逐帧误差一致；未投影点的预测与无图预测一致，非②方法也与原始LEADER预测一致。②以自身已微调回归头的无图输出为参照，不能混用原始回归头。',
        '下述“坐标误差”是每个coarse voxel预测世界坐标对原LEADER GT目标的距离，单位米；它不是相机预测深度，也不是最终相机/车体位置误差。原始raw代表点仅作图像采样，不改变定位GT。',
        '## 1. 图像提供了有效修正，但未改善主要求姿态点集',
        '\n'.join(coordinates),
        '固定原始LEADER的可靠度top50%点集再比较，避免由于换了筛选点而造成统计假象。原未入选点误差较大，其改善占据总体均值收益；高可靠点平均误差增加约5.5毫米，虽然幅度小，但原始定位已处于约13厘米水平。不能把总体坐标均值下降直接解释为姿态会改善。',
        f"原始top50%累计12726个点，其中12725个已经满足2m坐标内点阈值；当前图像融合只改变其中{current['val']['selection']['entered_selection']}个点的入选资格。基线已几乎没有粗大离群点可供拯救，重点是保护精确对应关系。top50%只是Matcher输入，后续Matcher还会进一步筛选，不能说每个点都等量参与最终位姿。",
        '## 2. 为什么修复前最终误差反而更接近基线',
        f"修复前32帧只有{old['val']['valid']['count']}个图像有效点，其中{old['val']['selected_before']['count']}个进入原top50%；修复后是{valid['count']}个和{selected['count']}个。参与候选对应集的受影响点约扩大{selected['count']/old['val']['selected_before']['count']:.1f}倍。",
        '修复前这些高可靠点的平均坐标误差其实也恶化（0.3297→0.3440m），比修复后的单点平均伤害更大，只是影响范围很小。修复前整体位姿接近基线不能证明错误投影更好；修复后更广的影响暴露了现有残差对准确点的干扰。旧、新图像有效点集合不同，不应将两者的原始均值直接当作同一批点比较。',
        '## 3. 门控未学会稳定保护准确点',
        f"在修复后①的全部有效点中，改善超过1cm的占{valid['improved_over_1cm']*100:.2f}%，恶化超过1cm的占{valid['harmed_over_1cm']*100:.2f}%，其余变化不超过1cm。改善点门控均值{valid['gate_helpful_mean']:.4f}，伤害点{valid['gate_harmful_mean']:.4f}，几乎相同；原top50%内的门控收益AUC为{selected['gate_helpfulness_auc']:.4f}，没有呈现可靠的有益修正排序。",
        '\n'.join(all_methods),
        '变化定义为融合后减融合前，负值表示改善；AUC以改善/恶化超过1cm的点为正/负样本，分数为实际gate，0.5表示没有排序区分力。gate控制残差强度，不是经过校准的有益概率，该统计是行为诊断而非独立因果证明。所有方法训练集准确点改善、验证集准确点不改善，支持存在泛化不足；仍不能据此确定究竟由小数据、特征分布、损失设计或网络容量哪个因素单独造成。',
        '## 4. 坐标变化还是可靠度变化在造成问题',
        '\n'.join(pose_table),
        '①仅替换坐标时已经复现大部分平均定位恶化；仅替换可靠度时在当前开发集略好。这使“坐标残差干扰原有准确对应”成为更直接的定位问题线索，而不是单纯可靠度排序崩溃。非线性Matcher意味着这些变化不可简单相加；0.1288m这一数字仅为事后诊断，不能当成已验证的新方法。',
        'GT逐点选择在原预测和融合预测之间取坐标误差更小者，仅用于确认现有残差中存在可利用修正；该选择读取查询GT，不可用于推理，也不是最终位姿的严格上界。即使如此，①平移收益也很小，不能推断换门控就能大幅提升。',
        '## 5. 图像内容是不是完全没有被用到',
        f"固定①权重，在全部有效点上，正确图像平均坐标误差{valid['mean_error_after']:.4f}m，随机置换描述子{valid['wrong_random_mean_error']:.4f}m，换训练集均值{valid['constant_image_mean_error']:.4f}m。正确对应的中间坐标确实更好，不能说输出只是与图像无关的固定偏移。",
        '但①正确图像没有稳定转化为更好的最终姿态：随机输入本次位姿甚至略好。③④的正确图像最终平移优于随机和均值输入，②也有类似现象，但都不足以超越原始LEADER。均值替换本身可能引入分布外输入，因此不能仅凭均值对照变差就断言图像包含可泛化的定位信息；随机置换保留有效描述子集合，是额外对应关系对照。',
        '## 下一步建议',
        '优先验证保护原始坐标的最小方案：冻结原LEADER坐标预测，只让视觉调整可靠度；或用明确的训练约束保护高可靠LiDAR对应点。先判断有效视觉信号能否进入最终定位，再决定是否扩大图像网络。此处仅提出下一步，不修改已有模型或启动新训练。',
        '所有结论局限于同一条已接触本地轨迹的32帧开发集，点之间、帧之间相关，不能把5629个点当成5629个独立试验来声称统计显著；也没有证明图像分支在困难季节或更大数据上不可用。',
        '复现：egonn118环境执行diagnose_fusion.py与report_fusion_diagnostic.py。协议、各方法逐帧统计、全部姿态干预指标保存在本目录。',
    ])
    (OUTPUT/'REPORT.md').write_text(report,encoding='utf-8')
    hashes={name:run.digest(Path(root)/'aligned.pt') for name,root in ROOTS.items()}
    hashes['LEADER']=run.digest(run.WORKSPACE/'research/image_gate_checkpoint/model.safetensors')
    run.save_json(OUTPUT/'checkpoint_hashes.json',hashes)
    print('\n'.join(coordinates))
    print('\n'.join(pose_table))


if __name__=='__main__':
    main()
