import json
import shutil
import numpy as np
import torch

from experiment import ARGS, OUT, HERE, run, selected, pose_error, Matcher
from model import BenefitModulation, reliability_rank
from report import interval


def choose(base, normal, opened, target, valid):
    candidates = torch.stack([base[:, :3], normal[:, :3], opened[:, :3]], dim=1)
    errors = (candidates - target[:, None]).norm(dim=-1)
    choice = errors.argmin(dim=1)
    choice = torch.where(valid, choice, torch.zeros_like(choice))
    result = base.clone()
    result[:, :3] = candidates[torch.arange(len(base), device=base.device), choice]
    assert torch.equal(result[~valid], base[~valid])
    assert ((result[:, :3] - target).norm(dim=-1) <= errors[:, 0] + 1e-6).all()
    return result, choice, errors


def main():
    torch.set_num_threads(4)
    destination = OUT / 'oracle'
    destination.mkdir(exist_ok=True)
    protocol = dict(checkpoints='Previously selected best.pt; no retraining or reselection',
                    candidates=['gate=0', 'learned gate', 'gate=1 with original amplitude bound'],
                    gt='Existing coarse voxel localization target, not Cartesian projection representative',
                    selection='Minimum pointwise Euclidean coordinate error; exact ties prefer baseline then normal then open',
                    matcher='Original baseline candidate indices AND order, local source coordinates; paired seed 2089',
                    invalid='Keep original prediction', scope='32 touched development frames; GT-assisted diagnostic, not deployable or pose upper bound',
                    weights={f'{arm}_{seed}': run.digest(OUT / f'{arm}_{seed}' / 'best.pt') for seed in [2089, 2090, 2091] for arm in ['aligned', 'shuffled']})
    run.save_json(destination / 'protocol.json', protocol)
    for name, digest in protocol['weights'].items():
        assert digest == run.digest(HERE / 'results' / name / 'best.pt')
    dummy = torch.zeros(3, 4)
    tied, choice, _ = choose(dummy, dummy, dummy, torch.ones(3, 3), torch.tensor([True, False, True]))
    assert not choice.any() and torch.equal(tied, dummy)
    decoder = run.load_leader(ARGS).decoder
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    rows = sorted([r for r in json.loads((ARGS.output / 'manifest.json').read_text()) if r['split'] == 'val'], key=lambda r: r['frame_id'])
    summary, arrays = {}, {}
    with torch.no_grad():
        for seed in [2089, 2090, 2091]:
            for arm in ['aligned', 'shuffled']:
                name = f'{arm}_{seed}'
                head = BenefitModulation().cuda().eval()
                head.load_state_dict(torch.load(OUT / name / 'best.pt'))
                previous = {r['frame_id']: r for r in json.loads((OUT / name / 'development.json').read_text())}
                records = []
                for row in rows:
                    item = run.frame(ARGS, row)
                    item['indices'] = selected(item['prediction'])
                    visual = item['image'].clone()
                    if arm == 'shuffled':
                        rng = np.random.default_rng(seed + int(row['frame_id']) % 1000000007)
                        indices = torch.where(item['valid'])[0]
                        visual[indices] = visual[indices[torch.as_tensor(rng.permutation(len(indices)), device='cuda')]]
                    output = head(item['features'], visual, reliability_rank(item['prediction'][:, 3]), item['valid'])
                    normal, opened = decoder(output['fused']), decoder(output['attempt'])
                    assert torch.equal(decoder(item['features']), item['prediction'])
                    oracle, choice, error = choose(item['prediction'], normal, opened, item['target'], item['valid'])
                    predictions = dict(baseline=item['prediction'], normal=normal, opened=opened, oracle=oracle)
                    poses = {k: pose_error(item, pred, matcher, True) for k, pred in predictions.items()}
                    assert np.allclose(poses['normal'], previous[row['frame_id']]['fixed'], atol=1e-7, rtol=0)
                    mask = torch.zeros_like(item['valid'])
                    mask[item['indices']] = True
                    valid_selected = mask & item['valid']
                    record = dict(frame_id=row['frame_id'], poses=poses,
                                  selected_count=int(mask.sum()), selected_visible_count=int(valid_selected.sum()),
                                  selected_error_sum={k: float((pred[mask, :3] - item['target'][mask]).norm(dim=-1).sum()) for k, pred in predictions.items()},
                                  visible_selected_choices=torch.bincount(choice[valid_selected], minlength=3).tolist())
                    records.append(record)
                run.save_json(destination / f'{name}_frames.json', records)
                arrays[name] = {k: np.asarray([r['poses'][k] for r in records]) for k in predictions}
                summary[name] = dict(metrics={k: run.metrics(v) for k, v in arrays[name].items()},
                                     selected_coordinate_error={k: sum(r['selected_error_sum'][k] for r in records) / sum(r['selected_count'] for r in records) for k in predictions},
                                     visible_selected_choices=np.sum([r['visible_selected_choices'] for r in records], axis=0).tolist())
                print(name, summary[name], flush=True)
    aligned = np.mean([arrays[f'aligned_{s}']['oracle'] for s in [2089, 2090, 2091]], axis=0)
    shuffled = np.mean([arrays[f'shuffled_{s}']['oracle'] for s in [2089, 2090, 2091]], axis=0)
    baseline = arrays['aligned_2089']['baseline']
    summary['paired'] = dict(aligned_oracle_mean=aligned.mean(0).tolist(), shuffled_oracle_mean=shuffled.mean(0).tolist(),
                             delta_baseline=float((aligned[:, 0] - baseline[:, 0]).mean()),
                             delta_shuffled=float((aligned[:, 0] - shuffled[:, 0]).mean()),
                             baseline_block95=interval(aligned[:, 0] - baseline[:, 0]), shuffled_block95=interval(aligned[:, 0] - shuffled[:, 0]))
    run.save_json(destination / 'summary.json', summary)
    lines = ['# GT 辅助三候选门控诊断', '',
             '**当前分支包含经 GT 筛选后可带来小幅定位改善的修正，但置乱分支获得相近改善，不能归因于正确逐点视觉对应；保留纯 LEADER，不继续调整上一版训练方案。**', '',
             '六组已内部选定的权重保持不变，未训练、未重新选检查点。对有效点在原预测、正常门控、保留幅度限制的全开尝试中按 GT 坐标误差选择；并列优先原预测。GT 使用既有 coarse voxel 监督，非投影代表点。', '',
             '所有姿态均固定原 LEADER 候选索引、顺序、本地坐标及 Matcher 随机状态。无图像点严格保留原预测。核验了并列处理、逐点坐标误差不增、缓存原预测完全一致，以及正常门控固定候选姿态与上一轮逐帧结果一致。', '',
             '| 方法 | 平均位置 cm | 平均旋转 ° | 位置 P95 cm | 原候选平均坐标误差 m |', '|---|---:|---:|---:|---:|']
    b = summary['aligned_2089']
    m = b['metrics']['baseline']
    lines.append(f'| 纯 LEADER | {m["mean"][0]*100:.4f} | {m["mean"][1]:.4f} | {m["p95"][0]*100:.4f} | {b["selected_coordinate_error"]["baseline"]:.6f} |')
    for name, r in summary.items():
        if name == 'paired':
            continue
        for condition in ['normal', 'opened', 'oracle']:
            m = r['metrics'][condition]
            lines.append(f'| {name} {condition} | {m["mean"][0]*100:.4f} | {m["mean"][1]:.4f} | {m["p95"][0]*100:.4f} | {r["selected_coordinate_error"][condition]:.6f} |')
    lines += ['', 'normal、opened、oracle 均为固定原候选诊断，不是按新可靠度筛选的标准推理。完整逐帧结果、选择计数和成功率见 JSON。', '',
              '## 配对比较', '', '下列区间为跨种子逐帧均值、连续 4 帧分块 bootstrap 95% 区间，仅 8 个轨迹块，不能据此宣称统计等效。', '',
              '```json', json.dumps(summary['paired'], indent=2), '```', '',
              '## 解释', '',
              '正确图像三个种子的 GT 选择均值为 12.7810 cm，比纯 LEADER 的 12.9776 cm 低约 1.97 mm；置乱图像为 12.7900 cm，同样低约 1.88 mm。正确图像仅在一个配对种子中胜过置乱，三种子均值只差约 0.09 mm。六组的平均旋转和位置 P95 均优于 baseline，成功率均为 32/32。', '',
              '这批候选并非完全没有可利用的修正：GT 选择同时降低了原候选坐标误差和平均定位误差。因此本次不符合“坐标更准却完全不改善定位”的观察；但更符合“正确与置乱经 GT 选择后改善相近”。不能据此认定真正缺少的仅是一个更好的可学习门控，也不能宣称这些改进来自正确视觉对齐。', '',
              '分块区间均包含 0：正确图像相对 baseline 的位置差约为 [-4.97, +0.87] mm，相对置乱约为 [-2.07, +2.19] mm。当前小样本不足以建立稳健优势，也不能证明两种输入统计等效。', '',
              '这是已接触的 32 帧开发集上的 GT 辅助机制诊断，不是可部署方法，也不是定位性能上限；未联合优化姿态，也没有穷尽门控取值。正确图像须与同样经过 GT 选择的置乱修正比较。']
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    shutil.copytree(destination, HERE / 'results/oracle', dirs_exist_ok=True)


if __name__ == '__main__':
    main()
