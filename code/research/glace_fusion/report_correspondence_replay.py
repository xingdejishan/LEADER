import argparse
import json
from pathlib import Path

import numpy as np

from .correspondence_replay import dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefix', default='correspondence-replay')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    folders = {v: args.bundle / 'outputs' / (args.prefix + '-' + v) for v in ('selected', 'balanced')}
    reports = {v: json.loads((p / 'summary.json').read_text()) for v, p in folders.items()}
    rows = {v: [json.loads(line) for line in (p / 'records.jsonl').read_text().splitlines()] for v, p in folders.items()}
    if [r['image'] for r in rows['selected']] != [r['image'] for r in rows['balanced']]:
        raise ValueError('Frame sets differ')
    for row in rows['selected']:
        stem = row['image']
        with np.load(folders['selected'] / 'frames' / (stem + '.npz')) as a, np.load(folders['balanced'] / 'frames' / (stem + '.npz')) as b:
            for name in ('candidate_T_WB', 'common_indices', 'common_uv', 'supervision_xyz', 'full_indices'):
                np.testing.assert_array_equal(a[name], b[name])
    qa = dict(frames=len(rows['selected']), identical_candidates_pixels_and_supervision_across_heads=True,
        source_pool_hashes_match=all(a['hashes']['pool'] == b['hashes']['pool'] for a, b in zip(rows['selected'], rows['balanced'])),
        maximum_legacy_vs_boundary_penalty_score_difference=max(arm['legacy_strict_score_max_difference'] for row in rows['selected'] for arm in row['arms'].values()),
        oracle_matched_block_counts_verified=True)
    from .replay_geometry import grid_cells
    for row in rows['selected']:
        with np.load(folders['selected'] / 'frames' / (row['image'] + '.npz')) as data:
            uv, shape = data['common_uv'], data['shape_hw']
            a = grid_cells(uv[data['reliable_mask']], shape)
            b = grid_cells(uv[data['matched_random_indices']], shape)
            np.testing.assert_array_equal(np.bincount(a, minlength=16), np.bincount(b, minlength=16))
    dump(args.out / 'replay_qa.json', qa)
    for variant, report in reports.items():
        dump(args.out / (variant + '_summary.json'), report)
        dump(args.out / (variant + '_protocol.json'), json.loads((folders[variant] / 'protocol.json').read_text()))
    calibration = json.loads((args.bundle / 'outputs/calibration-contract-audit.json').read_text())
    validation = json.loads((args.bundle / 'outputs/validation-visual-geometry/summary.json').read_text())
    dump(args.out / 'calibration_audit.json', calibration)
    dump(args.out / 'validation_geometry.json', validation)
    lines = ['# 固定候选池的对应点替换回放', '',
        '目标：GLACE 为 LEADER 提供可靠的像素—世界三维坐标约束；成功标准是下游救回失败多于新增失败。', '',
        '本轮实现并运行第一阶段诊断，不宣称新模型训练或独立测试收益。真实缓存候选保留 v1、原始 LEADER 与所有 SC2 位姿；不生成 GT 扰动，不固定 GT 平移。此前旋转实验固定的是 LiDAR 预测平移，也没有固定 GT 平移，但它使用了另一套人工扩展候选，不能混用其 148/148 上限。', '',
        f"当前真实候选池 @1m/2° 可达 {reports['selected']['all']['reachable_1m_2deg']}/148，@0.5m/1° 可达 {reports['selected']['all']['reachable_05m_1deg']}/148；原始 LEADER 输出为 101/148，现用 v1-two-stage 基线为 117/148。", '',
        '## 相同像素、点数和覆盖的核心对照', '',
        '表内为 1m/2° 成功帧数，分母均为 148；联合优化可能产生池外新位姿，因此可超过仅重评分的池内上限。', '',
        '| 输入 | selected 重评分 | selected 优化后 | balanced 重评分 | balanced 优化后 |',
        '|---|---:|---:|---:|---:|']
    labels = dict(prediction_common='GLACE 预测', supervision_common='当前监督目标（循环构造诊断）',
        shuffled_common='打乱像素配对（负对照）', oracle_prediction='真值筛出的预测子集（oracle）',
        oracle_matched_random='与 oracle 同点数、同空间块数量的随机子集')
    for arm, label in labels.items():
        values = [reports[v]['all']['methods'][arm + '/' + stage]['success_1m_2deg'] for v in ('selected', 'balanced') for stage in ('joint_selected', 'joint_refined')]
        lines.append('| ' + label + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '前三行使用完全相同像素；oracle 与随机子集配对比较，不能把更少的 oracle 点直接与全点结果当作公平的单变量消融。独立参考三维点缺失，该行明确跳过，绝不以监督目标代替独立参考。', '',
        '## 结论边界', '',
        '- 监督替换后优化显著改善，提示预测点与现有目标之间存在重要差距；但这些目标由同一相机 GT 和内参生成，方向天然自洽，只能测试条件接口，不能证明标定、遮挡处理或监督表面正确。',
        '- 即使直接使用监督点，相机单独重评分也只有 116/148；候选的平移—旋转权衡、视角可观测性、两套参考位姿差异都可能影响排序，不能把问题全部归为置信度。',
        '- 真值筛点后的联合优化达到 120/148（selected）、124/148（balanced），但仍分别损害 5、4 个基线成功帧；存在可利用点不等于能可靠识别这些点，也不等于优化总能改善。',
        '- 打乱对应关系未出现稳定收益；额外候选与优化预算已固定。',
        '- 不依赖 GT 的全图预测分块验收，重评分回退后两组均为 117/148；优化后验收均为 116/148，各损害 1 帧，门控不能被宣称为安全或已有效。',
        '- 当前数据上旧评分与“出界点罚满”的最大分数差为 0，因此该边界规则需要保留，但没有证据表明它解释了本轮失败。', '',
        '## 几何与标定核查', '',
        '从 NCLT 官方修正版 cam_params.zip 读取参数，并按官方投影示例的 body→LB3→camera 链核对：原始 K、缓存缩放 K、相机到 body 外参及 8 像素输出网格中心均一致；148 帧均有同名扫描。这是参数链一致性检查，不能代替真实物理特征投影、去畸变映射和运动同步核验。', '',
        '[NCLT 官方勘误及参数](https://robots.engin.umich.edu/nclt/)说明表 6 相机中心 x、y 曾互换；本地参数与当前官方文件一致。本轮没有据此改动标定。', '',
        '303 帧验证日期的几何诊断（该日期已参与历史模型选择，不是新留出集）：', '',
        '| 分组 | 平均逐帧 10px 内点率 | 逐帧角误差中位数的中位数 |', '|---|---:|---:|']
    for label, name in [('全部', 'all'), ('纹理梯度较高', 'high_texture'), ('纹理梯度较低', 'low_texture'), ('预测较远', 'far_prediction'), ('预测较近', 'near_prediction')]:
        metric = validation['groups'][name]
        lines.append(f"| {label} | {metric['mean_frame_q10']:.2%} | {metric['median_frame_angle_deg']:.3f}° |")
    lines += ['', '纹理以当前帧 Sobel 梯度中位数分组；远近由预测距离分组。这里只报告关联，不据此在测试集选择权重，也不将其解释为静态语义或因果证据。有符号残差、沿视线误差、横向误差和角误差均已逐点保存。', '',
        '## 复现与产物', '',
        '在本地包根目录运行，输出目录必须不存在：', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe local.py replay --variant selected --out outputs/replay-new', '```', '',
        f'完整候选、固定像素、逐点残差、分数、选中位姿、优化位姿及 GT-only 评价存放在 `outputs/{args.prefix}-{{selected,balanced}}/frames`，逐帧记录为 `records.jsonl`。`summary.json` 含两个成功门限、95% 分位误差、救回/损害及连续 60 秒轨迹块的净变化；相邻帧没有当作独立样本做显著性宣称。', '',
        '本目录保存冻结协议、汇总和配对一致性检查。推理分支不读 GT；监督与 oracle 分支明确标为诊断。没有训练新坐标网络、可靠性头或按测试结果选择门限。独立静态参考点、独立日期的真实 LEADER 候选以及对应的参考质量资料是下一阶段仍需补齐的输入。']
    (args.out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    for path in args.out.glob('*.json'):
        path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    print(str(args.out / 'REPORT.md'))


if __name__ == '__main__':
    main()
