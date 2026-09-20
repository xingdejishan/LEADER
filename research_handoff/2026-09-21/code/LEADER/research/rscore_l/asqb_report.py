import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .prepare import save_json


def summarize_constraints(arms):
    eligible = [a for a in arms if a['constraint']['good_points'] >= 20]
    out = dict(frames=len(arms), diagnostic_eligible=len(eligible), fewer_than_20_good=len(arms)-len(eligible))
    if not eligible:
        return out
    out.update(median_good_points=float(np.median([a['constraint']['good_points'] for a in eligible])),
        median_largest_cell_fraction=float(np.median([a['constraint']['largest_cell_fraction'] for a in eligible])),
        median_occupied_cells=float(np.median([a['constraint']['occupied_cells'] for a in eligible])),
        median_angular_rms_deg=float(np.median([a['constraint']['angular_rms_deg'] for a in eligible])),
        median_rotation_condition=float(np.median([a['constraint']['rotation']['condition'] for a in eligible])),
        median_rotation_schur_condition=float(np.median([a['constraint']['rotation_schur']['condition'] for a in eligible])),
        majority_in_one_cell=sum(a['constraint']['largest_cell_fraction'] >= .5 for a in eligible))
    return out


def report_asqb(root):
    folder = root / 'asqb/test'
    records = json.loads((folder / 'records.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    base = [r['arms']['uniform_100'] for r in records]
    errors = np.array([a['camera_error'] for a in base])
    diagnosis = dict(baseline_groups={
        'rotation_at_most_5deg': summarize_constraints([a for a in base if a['camera_error'][1] <= 5]),
        'rotation_over_5deg': summarize_constraints([a for a in base if a['camera_error'][1] > 5])},
        tail=dict(over_10deg=int((errors[:, 1]>10).sum()), over_30deg=int((errors[:, 1]>30).sum()),
            over_10deg_share_of_rotation_error_sum=float(errors[errors[:, 1]>10, 1].sum()/errors[:, 1].sum())),
        per_arm={key: summarize_constraints([r['arms'][key] for r in records]) for key in summary})
    eligible = [a for a in base if a['constraint']['good_points'] >= 20]
    rotation = np.array([a['camera_error'][1] for a in eligible])
    features = {
        'largest_cell_fraction': [a['constraint']['largest_cell_fraction'] for a in eligible],
        'angular_rms_deg': [a['constraint']['angular_rms_deg'] for a in eligible],
        'log10_schur_condition': [np.log10(a['constraint']['rotation_schur']['condition']) for a in eligible],
        'good_fraction': [a['constraint']['good_fraction'] for a in eligible]}
    diagnosis['spearman_with_rotation_error'] = {key: float(spearmanr(value, rotation).statistic) for key, value in features.items()}
    diagnosis['paired'] = {}
    for percent in (10, 20, 50):
        for comparison in ('uniform', 'top_grid'):
            pairs = [(r['arms'][f'asqb_{percent}'], r['arms'][f'{comparison}_{percent}']) for r in records]
            differences = np.array([a['camera_error'] for a, _ in pairs])-np.array([b['camera_error'] for _, b in pairs])
            eligible_pairs = [(a, b) for a, b in pairs if min(a['constraint']['good_points'], b['constraint']['good_points']) >= 20]
            condition_improved = [a['constraint']['rotation_schur']['condition'] < b['constraint']['rotation_schur']['condition'] for a, b in eligible_pairs]
            rotation_improved = [a['camera_error'][1] < b['camera_error'][1] for a, b in eligible_pairs]
            diagnosis['paired'][f'{percent}_vs_{comparison}'] = dict(mean_error_delta=differences.mean(0).tolist(),
                rotation_improved=int((differences[:, 1] < 0).sum()), rotation_degraded=int((differences[:, 1] > 0).sum()),
                eligible_pairs=len(eligible_pairs), schur_condition_improved=sum(condition_improved),
                condition_and_rotation_improved=int((np.array(condition_improved) & np.array(rotation_improved)).sum()))
    save_json(folder / 'diagnosis.json', diagnosis)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for axis, (title, metric) in zip(axes[0], [('Mean rotation error (deg)', lambda a:a['camera']['mean'][1]),
            ('Median rotation error (deg)', lambda a:a['camera']['median'][1]),
            ('PnP success <1 m, 2 deg (of 148)', lambda a:a['camera']['success_1m_2deg'])]):
        for method, label, color in [('uniform', 'Spatial uniform', '#64748b'), ('top_grid', 'Reliability Top-K + grid', '#e49c27'), ('asqb', 'ASQB rotation modes', '#147d92')]:
            percentages = [10, 20, 50, 100]
            axis.plot(percentages, [metric(summary[f'{method}_{p}']) for p in percentages], marker='o', label=label, color=color)
        axis.set_title(title, fontsize=11)
        axis.set_xticks([10, 20, 50, 100])
        axis.set_xlabel('Retained points (%)')
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for axis, key, label in zip(axes[1], ['largest_cell_fraction', 'angular_rms_deg', 'log10_schur_condition'],
            ['Largest 4x4 cell share of directional inliers', 'Directional-inlier angular RMS (deg)', 'log10 rotation Schur condition']):
        axis.scatter(features[key], rotation, s=18, alpha=.65, color='#147d92')
        axis.set_yscale('log')
        axis.set_xlabel(label, fontsize=9)
        axis.set_ylabel('Full-point rotation error (deg, log)')
        axis.set_title(f"Spearman rho = {diagnosis['spearman_with_rotation_error'][key]:.3f}", fontsize=11)
        axis.grid(alpha=.2)
    fig.suptitle('ASQB existing-correspondence selection | 148 development test images', fontsize=14)
    fig.savefig(folder / 'asqb_curves.png', dpi=180)
    fig.savefig(folder / 'asqb_curves.pdf')
    plt.close(fig)
    lines = ['# ASQB：既有对应点的旋转约束分组诊断', '',
        '本地原型不生成、不平均、不修正对应点，只从每个原有假设的 5,000 点中选索引。使用旋转雅可比的归一化信息矩阵划分 64 个几何 mode，在 mode 内按已有可靠性排序、跨 mode 轮询。mode 只读取像素与 K，不使用预测位姿或真值。', '',
        '固定 10 个完整假设、原坐标与可靠性、PnP 最大 10,000 次迭代和种子 2089。与此前同点数空间均匀和 Top-K+网格约束比较；输入哈希逐帧核验，100% 全点基线保持一致。', '',
        '## 视觉定位', '', '| 保留比例 | 方法 | 平均平移 m | 平均旋转 ° | 中位旋转 ° | 成功数 <1m、2° |', '|---|---|---:|---:|---:|---:|']
    for percent in (10, 20, 50, 100):
        for method, name in [('uniform', '空间均匀'), ('top_grid', '可靠性 Top-K'), ('asqb', 'ASQB')]:
            c = summary[f'{method}_{percent}']['camera']
            lines.append(f"| {percent}% | {name} | {c['mean'][0]:.3f} | {c['mean'][1]:.3f} | {c['median'][1]:.3f} | {c['success_1m_2deg']} / 148 |")
    lines.extend(['', '## 原全点基线：高旋转误差帧是否集中于局部', '',
        '下表的“方向一致点”为真值位姿下正深度且重投影误差 <10px 的点，不是独立核验的三维正确点。只对至少 20 个方向一致点的帧汇总几何分布，少于 20 点的帧单列，不默认为分布良好。', '',
        '| 旋转误差组 | 帧数 | 可诊断帧数 | 单格最大占比中位数 | 角度 RMS 中位数 | 旋转 Schur 条件数中位数 |', '|---|---:|---:|---:|---:|---:|'])
    for name, group in diagnosis['baseline_groups'].items():
        lines.append(f"| {name} | {group['frames']} | {group['diagnostic_eligible']} | {group.get('median_largest_cell_fraction', float('nan')):.3f} | {group.get('median_angular_rms_deg', float('nan')):.3f} | {group.get('median_rotation_schur_condition', float('nan')):.2f} |")
    tail = diagnosis['tail']
    lines.extend(['', f"全点基线中，旋转误差 >10° 有 {tail['over_10deg']} 帧，>30° 有 {tail['over_30deg']} 帧；>10° 帧贡献了旋转误差总和的 {tail['over_10deg_share_of_rotation_error_sum']:.1%}。", '',
        '条件数使用点均值信息矩阵，并用 Schur 补消去平移参数；数值秩不足时条件数上限为 1e12。该计算依赖预测深度，是几何诊断代理，不是校准后的姿态不确定性。真值仅用于事后诊断。', '',
        '相关性均基于全点基线输出的完整假设；它不能单独证明因果。不同采样方法最终可能选择不同假设，因此条件数与误差同步变化也不能独立归因为去冗余。', '',
        '## 固定 LEADER 候选池：联合评分留出网格', '',
        '沿用此前 Top-K 诊断评分，不是原融合后端最终验收。仅有 43 帧同时具备正确和错误候选，正间隔表示最低错误候选代价高于最低正确候选代价。', '',
        '| 比例 | 方法 | 正间隔帧 | 平均联合间隔 |', '|---|---|---:|---:|'])
    for percent in (10, 20, 50, 100):
        for method, name in [('uniform', '空间均匀'), ('top_grid', '可靠性 Top-K'), ('asqb', 'ASQB')]:
            s = summary[f'{method}_{percent}']['candidates']['joint_holdout']
            lines.append(f"| {percent}% | {name} | {s['positive_margin']} / {s['evaluable']} | {s['mean_margin']:.6f} |")
    (folder / 'report.md').write_text('\n'.join(lines)+'\n')
