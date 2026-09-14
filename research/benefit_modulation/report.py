import json
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = Path('/home/zhang/benefit-channel-modulation')


def read(path):
    return json.loads(path.read_text())


def metrics(records, key='standard'):
    a = np.asarray([r[key] for r in records])
    result = dict(mean=a.mean(0).tolist(), p95=np.percentile(a, 95, axis=0).tolist(),
                  successes=int(((a[:, 0] < 1) & (a[:, 1] < 5)).sum()))
    if 'protected' in records[0]:
        count = sum(r['protected']['count'] for r in records)
        result['protected'] = dict(count=count, delta_m=sum(r['protected']['delta_sum'] for r in records) / count,
                                   harm_rate=sum(r['protected']['harmed'] for r in records) / count,
                                   help_rate=sum(r['protected']['helped'] for r in records) / count)
    return result


def interval(a):
    rng = np.random.default_rng(271828)
    blocks = a.reshape(8, 4).mean(1)
    samples = blocks[rng.integers(0, 8, size=(10000, 8))].mean(1)
    return np.percentile(samples, [2.5, 97.5]).tolist()


def main():
    baseline = read(OUT / 'baseline.json')
    reference = read(OUT / 'reference_line1.json')
    summary = dict(baseline=metrics(baseline), line1=metrics(reference), line1_fixed=metrics(reference, 'fixed'), runs={})
    values = {}
    for seed in [2089, 2090, 2091]:
        for arm in ['aligned', 'shuffled']:
            name = f'{arm}_{seed}'
            records = read(OUT / name / 'development.json')
            training = read(OUT / name / 'training.json')
            assert [r['epoch'] for r in training] == list(range(1, 101))
            expected = min([r for r in training if 'internal' in r], key=lambda r: r['internal']['mean'][0])
            assert [r['frame_id'] for r in records] == [r['frame_id'] for r in baseline]
            result = dict(selection=read(OUT / name / 'selection.json'), standard=metrics(records), fixed=metrics(records, 'fixed'))
            assert result['selection']['epoch'] == expected['epoch']
            result['gate_bins'] = []
            for b in range(5):
                gate = np.concatenate([r['gate_bins'][b]['gate'] for r in records])
                benefit = np.concatenate([r['gate_bins'][b]['benefit'] for r in records])
                correlation = float(spearmanr(gate, benefit).correlation)
                result['gate_bins'].append(dict(count=len(gate), spearman=correlation if np.isfinite(correlation) else None,
                                                mean_gate=float(gate.mean()), mean_benefit_m=float(benefit.mean())))
            values[name] = np.asarray([r['standard'] for r in records])
            summary['runs'][name] = result
    base = np.asarray([r['standard'] for r in baseline])
    aligned = np.mean([values[f'aligned_{s}'] for s in [2089, 2090, 2091]], axis=0)
    shuffled = np.mean([values[f'shuffled_{s}'] for s in [2089, 2090, 2091]], axis=0)
    summary['paired_seed_mean_translation'] = dict(aligned_minus_baseline=float((aligned[:, 0] - base[:, 0]).mean()),
         baseline_difference_block95=interval(aligned[:, 0] - base[:, 0]),
         aligned_minus_shuffled=float((aligned[:, 0] - shuffled[:, 0]).mean()),
         shuffled_difference_block95=interval(aligned[:, 0] - shuffled[:, 0]))
    passed = True
    for seed in [2089, 2090, 2091]:
        a, s = [summary['runs'][f'{arm}_{seed}'] for arm in ['aligned', 'shuffled']]
        b = summary['baseline']
        passed &= a['standard']['mean'][0] < min(b['mean'][0], s['standard']['mean'][0])
        passed &= a['standard']['mean'][1] <= b['mean'][1] and a['standard']['p95'][0] <= b['p95'][0]
        passed &= a['fixed']['mean'][0] <= b['mean'][0]
        passed &= a['standard']['protected']['harm_rate'] < summary['line1']['protected']['harm_rate']
    summary['hypothesis_passed'] = bool(passed)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# 收益监督的视觉条件通道调制实验', '',
             '本轮使用可配对的本地缓存：51 帧训练、连续末段 13 帧内部选模、32 帧已接触开发评估。没有使用缺失原始扫描的 907 帧集合。', '',
             '六组均训练满 100 轮、每组 1300 次参数更新；每 10 轮用内部最终位置误差选检查点，平局取较早轮次。第 0 轮仅用于恒等性核验，不参与选模；没有选最好种子，也没有在 32 帧上调整参数。', '',
             '## 结构与验证', '',
             '冻结原 LEADER 编码器、MMRegressor 和 DeDoDe/PCA；37,473 参数小模块执行有界乘法调制。原可靠度帧内排名限制幅度为 2%–10%；输入采用无参数 LayerNorm，原 LiDAR 主路径不归一化。前 10 轮门控固定 0.5，其后加入停止梯度的尝试收益 BCE；保留原 TRR 和原入选可见对应的退化惩罚。', '',
             '96 帧初始融合与原特征严格一致，重算原回归输出与缓存最大差为 0。单元检查覆盖无图像恒等、调制上限、排名并列、门控预热、收益标签停止梯度、梯度穿过冻结回归头。投影、mask、定位坐标和监督目标未修改。', '',
             '## 开发集结果', '',
             '| 方法 | 内部选中轮次 | 平均位置 m | 平均旋转 ° | 位置 P95 m | 固定原候选位置 m |',
             '|---|---:|---:|---:|---:|---:|']
    for name, m, fixed, epoch in [('纯 LEADER', summary['baseline'], summary['baseline'], '—'), ('① 修复投影的简单门控', summary['line1'], summary['line1_fixed'], '历史训练')]:
        lines.append(f'| {name} | {epoch} | {m["mean"][0]:.6f} | {m["mean"][1]:.6f} | {m["p95"][0]:.6f} | {fixed["mean"][0]:.6f} |')
    for name, r in summary['runs'].items():
        m = r['standard']
        lines.append(f'| {name} | {r["selection"]["epoch"]} | {m["mean"][0]:.6f} | {m["mean"][1]:.6f} | {m["p95"][0]:.6f} | {r["fixed"]["mean"][0]:.6f} |')
    lines += ['', '## 原入选可见对应的保护', '', '| 方法 | 平均坐标误差变化 m | 伤害超过 1 cm | 改善超过 1 cm |', '|---|---:|---:|---:|']
    for name, r in [('①', summary['line1'])] + [(name, r['standard']) for name, r in summary['runs'].items()]:
        p = r['protected']
        lines.append(f'| {name} | {p["delta_m"]:+.6f} | {p["harm_rate"]:.2%} | {p["help_rate"]:.2%} |')
    lines += ['', '## 门控是否识别收益', '', '按原可靠度排名分成五个区间，下表是门控与令门控为 1 的视觉尝试收益的 Spearman 相关；正值表示更高门控倾向于更有益的尝试。只作开发集机制诊断，不能证明因果。', '', '| 方法 | q 0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1 |', '|---|---:|---:|---:|---:|---:|']
    for name, r in summary['runs'].items():
        cells = ['NA' if b['spearman'] is None else f'{b["spearman"]:.3f}' for b in r['gate_bins']]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    p = summary['paired_seed_mean_translation']
    lines += ['', '## 判定与边界', '', f'预先固定的机制判定：**{"通过" if passed else "未通过"}**。要求三个正确图像种子均优于纯 LEADER 与对应置乱种子，平均旋转和位置 P95 不退化，固定候选位置不退化，并减少相对于①的有害对应。', '',
              f'三个种子的逐帧均值：正确图像减 baseline 的位置差 {p["aligned_minus_baseline"]:+.6f} m，连续 4 帧分块 bootstrap 95% 区间 {p["baseline_difference_block95"]}；正确图像减置乱 {p["aligned_minus_shuffled"]:+.6f} m，区间 {p["shuffled_difference_block95"]}。仅 8 个轨迹块，此区间是探索性不确定性读数。', '',
              '标准定位按各自新可靠度排序选原比例对应；固定候选诊断严格保持原 LEADER 的索引与顺序，仅替换坐标。两者使用相同 Matcher 和配对随机状态。①为历史训练参考，训练数据量与优化预算不同，不能作为同预算结构消融。', '',
              '当前只有 51 帧用于梯度更新，13 帧用于选择检查点；32 帧不是独立盲测。此前重新编码的特征差异尚未完全解释，本轮全部条件严格复用同一份缓存，只能支持缓存条件下的局部研究结论。在线实现还需原预测与融合预测两次回归以及视觉编码，未测完整系统延迟。', '',
              '## 复现', '', '`check.py` 执行机制核验；`experiment.py` 执行六组训练与开发评估；`reference.py` 重算①及固定候选对照；`report.py` 汇总。运行环境为现有 WSL egonn118，数据根目录与原权重路径见 experiment.py。results 保存协议、逐轮内部误差、选择依据、逐帧开发结果和最佳/末轮小模块权重。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    shutil.copytree(OUT, HERE / 'results', dirs_exist_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
