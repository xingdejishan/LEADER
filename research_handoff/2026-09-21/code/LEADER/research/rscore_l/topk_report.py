import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .prepare import save_json


def report_topk(root):
    destination = root / 'topk'
    test = json.loads((destination / 'test/summary.json').read_text())
    records = json.loads((destination / 'test/records.json').read_text())
    baseline = json.loads((root / 'evaluation/reliable/test/records.json').read_text())
    assert [r['frame_id'] for r in records] == [r['frame_id'] for r in baseline]
    np.testing.assert_allclose([r['arms']['uniform_100']['camera_error'] for r in records],
        [r['camera_error'] for r in baseline], rtol=1e-8, atol=1e-8)
    paired = {}
    for percent in (10, 20, 50, 100):
        top = [r['arms'][f'top_grid_{percent}'] for r in records]
        uniform = [r['arms'][f'uniform_{percent}'] for r in records]
        errors_top = np.array([a['camera_error'] for a in top])
        errors_uniform = np.array([a['camera_error'] for a in uniform])
        success_top = (errors_top[:, 0] < 1) & (errors_top[:, 1] < 2)
        success_uniform = (errors_uniform[:, 0] < 1) & (errors_uniform[:, 1] < 2)
        pair = dict(pnp_new_success=int((success_top & ~success_uniform).sum()),
            pnp_new_failure=int((~success_top & success_uniform).sum()),
            correspondence_fraction_delta=float(np.mean([a['correspondence_fraction']-b['correspondence_fraction'] for a, b in zip(top, uniform)])))
        for key in ('camera_fit', 'camera_holdout', 'joint_fit', 'joint_holdout'):
            deltas = [a['candidates'][key]['margin']-b['candidates'][key]['margin'] for a, b in zip(top, uniform)
                if a['candidates'][key]['margin'] is not None]
            pair[key] = dict(evaluable=len(deltas), mean_margin_delta=float(np.mean(deltas)),
                margin_improved=int((np.array(deltas)>0).sum()))
        paired[str(percent)] = pair
    save_json(destination / 'paired.json', paired)
    panels = [('Correspondences <10 px (%)', lambda a: 100*a['correspondence_fraction']),
        ('PnP success <1 m, 2 deg (%)', lambda a: 100*a['camera']['success_1m_2deg']/148),
        ('Mean translation error (m)', lambda a: a['camera']['mean'][0]),
        ('Mean rotation error (deg)', lambda a: a['camera']['mean'][1]),
        ('Correct candidate wins: holdout (of 43)', lambda a: a['candidates']['joint_holdout']['positive_margin']),
        ('Mean joint margin: holdout', lambda a: a['candidates']['joint_holdout']['mean_margin'])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for axis, (label, value) in zip(axes.flat, panels):
        for mode, title, color in [('uniform', 'Spatial uniform', '#64748b'), ('top_grid', 'Reliability Top-K + grid cap', '#147d92')]:
            percentages = [10, 20, 50, 100]
            axis.plot(percentages, [value(test[f'{mode}_{p}']) for p in percentages], marker='o', label=title, color=color, linewidth=2)
        axis.set_title(label, fontsize=11)
        axis.set_xlabel('Retained points per hypothesis (%)')
        axis.set_xticks([10, 20, 50, 100])
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 43)
    axes[1, 2].axhline(0, color='#94a3b8', linewidth=.8)
    fig.suptitle('R-SCoRe-L reliability filtering | 148 development test images', fontsize=15)
    fig.savefig(destination / 'topk_curves.png', dpi=180)
    fig.savefig(destination / 'topk_curves.pdf')
    plt.close(fig)
    lines = ['# 可靠性 Top-K 与同点数空间均匀采样', '',
        '复用已训练的单帧 R-SCoRe-L 可靠性版本，固定 10 个检索假设、预测坐标、PnP 最大 10,000 次迭代及种子 2089；不重新训练。100% 结果与此前 55 / 148 的视觉基线逐帧一致。', '',
        'Top-K 按可靠性排序，4×4 网格每格最多 ceil(2K/16) 个点，仅当容量不足时放宽至可满足 K 的最小上限；均匀对照在格内固定种子随机排序，再跨格轮询。两者严格保留相同点数，并按原始索引顺序输入 PnP。', '',
        '## 测试集视觉定位', '',
        '| 保留比例 | 采样 | <10px 点比例 | 成功数 <1m、2° | 平均误差 m / ° | 中位误差 m / ° |',
        '|---|---|---:|---:|---:|---:|']
    for p in (10, 20, 50, 100):
        for mode, name in [('uniform', '空间均匀'), ('top_grid', '可靠性+网格约束')]:
            a = test[f'{mode}_{p}']
            c = a['camera']
            lines.append(f"| {p}% | {name} | {a['correspondence_fraction']:.2%} | {c['success_1m_2deg']} / 148 | {c['mean'][0]:.3f} / {c['mean'][1]:.3f} | {c['median'][0]:.3f} / {c['median'][1]:.3f} |")
    lines.extend(['', '## LEADER 候选区分', '',
        '正确候选定义为误差 <1m 且 <2°；只在候选池同时有正确、错误候选的帧上统计评分间隔。间隔 = 最低错误候选代价 − 最低正确候选代价，正值表示正确候选占优。真实位姿只用于事后标记，未用于采样、评分或选择假设。', '',
        '沿用原概率混合视觉评分和 0.5 / 0.5 LiDAR 联合评分，对每个候选选择完整假设，分别计算棋盘格拟合/留出网格；不混合不同假设的点。本诊断不执行最终精修或接受门控，不能把候选胜出数当作系统最终融合成功数。', '',
        '| 比例 | 采样 | 可区分帧数 | 正间隔帧：拟合 / 留出 | 平均联合间隔：拟合 / 留出 | 平均视觉间隔：拟合 / 留出 |',
        '|---|---|---:|---:|---:|---:|'])
    for p in (10, 20, 50, 100):
        for mode, name in [('uniform', '空间均匀'), ('top_grid', '可靠性+网格约束')]:
            c = test[f'{mode}_{p}']['candidates']
            f, h = c['joint_fit'], c['joint_holdout']
            lines.append(f"| {p}% | {name} | {f['evaluable']} | {f['positive_margin']} / {h['positive_margin']} | {f['mean_margin']:.6g} / {h['mean_margin']:.6g} | {c['camera_fit']['mean_margin']:.6g} / {c['camera_holdout']['mean_margin']:.6g} |")
    lines.extend(['', '## 解释边界', '',
        '四个比例构成预定诊断曲线；未按测试结果回调模型、阈值或采样规则。此测试集曾用于开发，单种子结果不能作为独立盲测结论。<10px 是方向一致性指标，不是三维坐标精度证明；即使筛选有收益，也不足以单独证明坐标模型已经足够。', '',
        '缺少训练侧真实候选池，当前可靠性头没有经过候选排序监督；因此 PnP 与 LEADER 候选区分可能出现不同趋势。'])
    (destination / 'report.md').write_text('\n'.join(lines)+'\n')
