import argparse
import csv
import json
from pathlib import Path

from .fixed_origin_refine import success, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summaries, records = {}, {}
    for variant in ('selected', 'balanced'):
        source = args.bundle / 'outputs' / ('fixed-origin-final-' + variant)
        records[variant] = [json.loads(line) for line in (source / 'records.jsonl').read_text().splitlines()]
        summaries[variant] = json.loads((source / 'summary.json').read_text())
        if len(records[variant]) != 148 or any(not arm['solver_called'] for r in records[variant] for arm in r['solver'].values()):
            raise ValueError('Report requires all 148 frames with every optimizer actually called')
        for name in ('summary.json', 'protocol.json', 'oracle_harmed_frames.json', 'complete.json'):
            write_json(args.out / (variant + '_' + name), json.loads((source / name).read_text()))
    write_json(args.out / 'qa.json', json.loads((args.bundle / 'outputs/fixed-origin-qa.json').read_text()))
    frame_rows = []
    harm_records = {}
    for variant, rows in records.items():
        harm_records[variant] = []
        for row in rows:
            base, lidar = row['errors']['baseline'], row['errors']['lidar_only']
            if success(base) and any(not success(e) for e in row['errors'].values()):
                harm_records[variant].append(row)
            for arm, diagnostics in row['solver'].items():
                error = row['errors'][arm]
                frame_rows.append(dict(variant=variant, image=row['image'], time_block=row['time_block'], arm=arm,
                    v1_t_m=base[0], v1_r_deg=base[1], lidar_only_t_m=lidar[0], lidar_only_r_deg=lidar[1],
                    result_t_m=error[0], result_r_deg=error[1],
                    v1_success=bool(success(base)), lidar_only_success=bool(success(lidar)), result_success=bool(success(error)),
                    solver_called=diagnostics['solver_called'], solver_converged=diagnostics['success'],
                    nfev=diagnostics['nfev'], residual_calls=diagnostics['residual_calls'],
                    lidar_support=diagnostics['lidar_support'], camera_support=diagnostics['camera_support'],
                    lidar_objective_before=diagnostics['fixed_objective_before']['lidar'],
                    lidar_objective_after=diagnostics['fixed_objective_after']['lidar'],
                    camera_scan_gt_t_m=row['camera_scan_gt_difference'][0],
                    camera_scan_gt_r_deg=row['camera_scan_gt_difference'][1],
                    camera_reference_result_t_m=row['camera_reference_errors'][arm][0],
                    camera_reference_result_r_deg=row['camera_reference_errors'][arm][1]))
    with (args.out / 'per_frame.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(frame_rows)
    write_json(args.out / 'harmed_frame_details.json', harm_records)
    lines = ['# 同一 v1 起点的视觉开关回放', '',
        '本轮只隔离共享精修的影响，没有重新选择候选、训练模型、调整门限或改变监督。', '',
        '## 实验约束', '',
        '- 所有组的起点均为缓存 v1_two_stage 完整六自由度位姿，不使用真值起点。',
        '- LiDAR 池、可靠性权重、0.3m 残差尺度、v1 初始支持集完全一致；支持集在优化中固定。',
        '- 使用同一个 JointProblem.refine、Cauchy 目标和 LM，max_nfev=20；视觉系数为 0 或原值 1，LiDAR 系数不重归一化。逐帧保存实际残差调用次数、nfev、收敛状态及位姿更新。',
        '- LiDAR-only 真正执行优化。诊断模式对视觉支持不足 3 点也继续运行同一 LiDAR 目标并加入现有 0–2 个视觉点，避免以整帧跳过优化产生假对照；正常后端默认保护条件不变。',
        '- prediction_full 沿用无 GT 的全图固定采样；另保留 prediction_common、旧 oracle 及其同点数同空间块随机控制。oracle 仍只表示相机参考位姿下正深度、重投影小于 10px，不保证三维正确。',
        '- 不加更新接受、候选回退或重新筛支持，直接记录原始优化结果。所有组、所有 148 帧均实际求解并报告收敛。', '',
        '## 成功数及视觉增量', '',
        '两个门限分别为 1m/2° 与 0.5m/1°，分母均为 148；“新增损害/救回”以同帧 LiDAR-only 输出为对照，不以 v1 为对照。', '',
        '| 模型 / 输入 | 1m/2° | 0.5m/1° | 相对 v1 救回 / 损害 | 相对 LiDAR-only 新增损害 / 救回 |',
        '|---|---:|---:|---:|---:|', '| v1 | 117 | 42 | 0 / 0 | — |']
    for variant in ('selected', 'balanced'):
        for name in ('lidar_only', 'prediction_full', 'prediction_common', 'oracle_prediction', 'oracle_matched_random'):
            item = summaries[variant]['all']['methods'][name]
            lines.append(f"| {variant} / {name} | {item['success_1m_2deg']} | {item['success_05m_1deg']} | {item['rescued_vs_v1']} / {item['harmed_vs_v1']} | {item['new_harm_vs_lidar_only']} / {item['rescue_vs_lidar_only']} |")
    lines += ['', '## 后端损害的逐帧归属', '',
        '两组模型的 LiDAR-only 输出逐元素完全一致；它救回 1 帧，损害 3 帧，117→115。三帧损害均是旋转越过 2°，且优化的 LiDAR Cauchy 目标都降低：', '',
        '| 帧 | v1 旋转误差 | LiDAR-only 旋转误差 | LiDAR 目标：之前 → 之后 |', '|---|---:|---:|---:|']
    for row in records['selected']:
        if success(row['errors']['baseline']) and not success(row['errors']['lidar_only']):
            info = row['solver']['lidar_only']
            lines.append(f"| {row['image']} | {row['errors']['baseline'][1]:.4f}° | {row['errors']['lidar_only'][1]:.4f}° | {info['fixed_objective_before']['lidar']:.6f} → {info['fixed_objective_after']['lidar']:.6f} |")
    lines += ['', '这 3 帧属于同一个连续 60 秒块，不能当作独立重复证据。结果说明“优化本身收敛、所优化的目标下降”不等于定位更准，当前应先保护 v1 已有解；本实验尚未区分权重、支持集与目标形式各自的贡献。', '',
        'selected prediction_full 相对 LiDAR-only 救回 2 帧且不新增失败，最后剩下的 1 帧损害已属于后端损害；不过严格门限仍为 36/148，低于 v1 的 42/148，不能宣称稳定融合收益。', '',
        'balanced prediction_full 相对 LiDAR-only 新增 3 个失败、救回 1 帧：其中 2 帧原本 v1 成功且 LiDAR-only 也成功，另 1 帧是取消了 LiDAR-only 对 v1 失败帧的救回。因此相对 v1 的最终 4 帧损害由 2 帧共同后端损害和 2 帧新增视觉损害组成，不能把 3 个“相对 LiDAR-only 新失败”全部叫作“新增 v1 损害”。', '',
        '## oracle 与两套参考位姿', '',
        'selected 的 oracle 与匹配随机控制为 122 对 116；balanced 为 123 对 117。两组 oracle 相对 LiDAR-only 均没有新增失败，分别救回 7、8 帧。oracle 剩余的 2 帧、1 帧 v1 损害全部已出现在 LiDAR-only 中，不能据此把它们归为视觉新增损害。', '',
        '逐帧检查相机与 scan 参考位姿：oracle 损害帧的参考平移差最大约 1.25mm、旋转差最大约 0.024°；这些帧在相机参考下仍然构成损害，没有因更换参考而翻转为成功。这排除了本批 oracle 损害仅由这两套参考的门限翻转造成，但不证明任一参考物理上无误。', '',
        '## 验证和后续边界', '',
        '30 项相关测试通过。148 帧中两组 LiDAR-only 的起点、初始支持掩码与输出逐元素一致；对 10 个真实帧比较 91ccbb8 的原 refine 实现与当前默认调用，输出逐元素一致。新增诊断开关没有静默改变默认视觉优化路径。', '',
        '下一步优先处理共享后端对已有 v1 解的更新接受与目标不一致；不启动新监督或置信度训练，不让 GLACE 学习抵消后端退化。当前数据均已用于开发，本轮归因不构成独立日期上的最终收益证明。', '',
        '## 复现', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe local.py refine-ablation --variant selected --out outputs/fixed-origin-new',
        '```', '',
        '完整结果在 outputs/fixed-origin-final-{selected,balanced}；本目录保存冻结协议、汇总、全部 1480 行逐帧方法对照、损害帧详情和一致性验证。逐帧二进制文件保存初值、输出位姿与固定支持掩码。']
    (args.out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
