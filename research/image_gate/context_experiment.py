import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import run
from fusion import ImageGate


class ContextGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ImageGate()
        with torch.random.fork_rng(devices=[]):
            self.adapter = nn.Sequential(nn.LayerNorm(774), nn.Linear(774, 128))

    def forward(self, lidar, image, valid):
        image = torch.where(valid[:, None], image, torch.zeros_like(image))
        local = image[:, :128] + self.adapter(image[:, 128:])
        return self.gate(lidar, local, valid)


def prepare(args):
    root = Path('/home/zhang/rscore-l-local')
    original = Path('/home/zhang/leader-image-gate-raw')
    package = run.WORKSPACE / 'LEADER/research/rscore_l'
    sys.path[:0] = [str(package / 'vendor'), str(package.parent), str(run.WORKSPACE / 'rscore-assets/hloc')]
    from rscore_l.evaluate import load_model
    from scrstudio.data.samplers import PQKNN
    model, checkpoint = load_model(root, args.scene_variant)
    model.requires_grad_(False)
    rows = json.loads((original / 'manifest.json').read_text())
    reference = json.loads((root / 'data/manifest.json').read_text())
    assert not ({r['frame_id'] for r in rows} & {r['frame_id'] for r in reference['train']})
    query_index = {r['frame_id']: i for i, r in enumerate(reference['val'])}
    queries = np.load(root / 'data/val/netvlad_feats.npy').astype(np.float32)
    embeddings = torch.load(root / 'data/train/pose_n2c.pt', weights_only=True)['model.embedding.weight'].cuda().float()
    with (root / 'data/train/netvlad_feats_pq.pkl').open('rb') as f:
        pq, codes = pickle.load(f)
    retriever = PQKNN(pq, codes, n_neighbors=1)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Global context and coarse/refined scene representations can disambiguate locally similar descriptors and reduce final LEADER pose error.',
        limitation='Wrong retrieval or reprojection depth ambiguity can inject incorrect scene context; no theoretical guarantee of improvement.',
        mechanism='Conditional scene-coordinate MLP: local128 + retrieved Node2Vec256 -> hidden768 + sc0/sc6; train adapter774->128, add local128, original gate.',
        retrieval='Frozen NetVLAD PQ top1 from 907 other-date training images; no query GT or pose used.',
        scene_head=args.scene_variant,
        supervision='persistent coarse/final LiDAR geometry supervision' if args.scene_variant == 'geometry' else 'reprojection only',
        extraction='Apply pointwise scene head to the same bilinearly sampled PCA descriptors as line 1; coordinates are context only, never a separate pose.',
        frozen=['LEADER encoder', 'LEADER decoder', 'DeDoDe/PCA', 'NetVLAD', 'Node2Vec', 'coarse/refinement'],
        train_frames=64, validation_frames=32, seed=2089, steps_per_arm=600, learning_rate=.0001,
        arms=['aligned', 'shuffled', 'aligned_wrong', 'aligned_missing'],
        pass_rule='mean translation >=5% better than line1 AND LEADER; mean rotation and p95 translation <=105% of each; no damage; aligned beats shuffled.',
        scope='Previously touched local development data, one seed; not blind/full NCLT.',
        head_sha256=run.digest(checkpoint), embedding_sha256=run.digest(root / 'data/train/pose_n2c.pt'),
        baseline_protocol_sha256=run.digest(original / 'protocol.json'),
        query_sha256=run.digest(root / 'data/val/netvlad_feats.npy'),
        retrieval_sha256=run.digest(root / 'data/train/netvlad_feats_pq.pkl'),
        trainable_parameters=sum(p.numel() for p in ContextGate().parameters()),
        frozen_scene_parameters=sum(p.numel() for p in model.parameters()))
    if args.scene_variant == 'scrfacto':
        protocol.pop('supervision')
        protocol['scene_head'] = 'scrfacto reprojection-trained with pose Node2Vec, no additional LiDAR 3D supervision (line 4 excluded)'
    path = args.output / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    run.save_json(path, protocol)
    run.save_json(args.output / 'manifest.json', rows)
    target = args.output / 'lidar'
    if not target.exists():
        target.symlink_to((original / 'lidar').resolve(), target_is_directory=True)
    (args.output / 'visual').mkdir(exist_ok=True)
    records = []
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        with np.load(original / 'visual' / (row['frame_id'] + '.npz')) as data:
            local = torch.tensor(data['image'], device='cuda')
            valid = torch.tensor(data['valid'], device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        index = retriever.kneighbors(queries[query_index[row['frame_id']]])[0]
        context = local.new_zeros((len(local), 774))
        with torch.inference_mode(), torch.autocast('cuda'):
            out = model({'features': torch.cat([embeddings[index][None].expand(int(valid.sum()), -1), local[valid]], -1)})
        context[valid] = torch.cat([out['features'].float(), out['sc0'].float() / 100, out['sc'].float() / 100], -1)
        assert context.shape[1] == 774 and torch.isfinite(context).all()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        combined = torch.cat([local, context], -1).cpu().numpy()
        np.savez(args.output / 'visual' / (row['frame_id'] + '.npz'), image=combined, valid=valid.cpu().numpy(), seconds=elapsed)
        records.append(dict(frame_id=row['frame_id'], reference_frame=reference['train'][int(index)]['frame_id'],
            valid=int(valid.sum()), extra_seconds=elapsed, visual_sha256=run.digest(original / 'visual' / (row['frame_id'] + '.npz'))))
    run.save_json(args.output / 'extraction.json', dict(records=records, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,
        timing_excludes_cached_DeDoDe_and_NetVLAD=True))
    print(json.dumps(protocol, indent=2), flush=True)


def evaluate(args, rows):
    run.evaluate(args, rows)
    result = json.loads((args.output / 'result.json').read_text())
    line1 = json.loads(Path('/home/zhang/leader-image-gate-raw/result.json').read_text())
    current = result['metrics']['aligned']
    references = [result['metrics']['baseline'], line1['metrics']['aligned']]
    passed = all(current['mean'][0] <= .95 * b['mean'][0] and current['mean'][1] <= 1.05 * b['mean'][1]
        and current['p95'][0] <= 1.05 * b['p95'][0] for b in references)
    passed &= result['damage'] == 0 and current['mean'][0] < result['metrics']['shuffled']['mean'][0]
    old = json.loads(Path('/home/zhang/leader-image-gate-raw/validation_frames.json').read_text())
    new = json.loads((args.output / 'validation_frames.json').read_text())
    assert [r['frame_id'] for r in old] == [r['frame_id'] for r in new]
    assert np.allclose([r['errors']['baseline'] for r in old], [r['errors']['baseline'] for r in new], atol=1e-6)
    result.update(passed=bool(passed), line1=line1['metrics']['aligned'],
        next_action='expand frozen development evaluation' if passed else 'stop this fixed configuration; no validation tuning')
    run.save_json(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))


def report(args, rows):
    import shutil
    gate = ContextGate().cuda().eval()
    item = run.frame(args, rows[0])
    with torch.no_grad():
        assert torch.equal(gate(item['features'], item['image'], item['valid']), item['features'])
    gate.load_state_dict(torch.load(args.output / 'aligned.pt', map_location='cuda'))
    missing = torch.zeros_like(item['valid'])
    poisoned = torch.full_like(item['image'], float('nan'))
    with torch.no_grad():
        assert torch.equal(gate(item['features'], poisoned, missing), item['features'])
    gate(item['features'], item['image'], item['valid']).square().mean().backward()
    adapter_gradient = float(gate.adapter[1].weight.grad.norm())
    assert np.isfinite(adapter_gradient) and adapter_gradient > 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(20):
            gate(item['features'], item['image'], item['valid'])
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            gate(item['features'], item['image'], item['valid'])
        torch.cuda.synchronize()
    diagnostic = dict(gate_ms=(time.perf_counter()-start)*5, voxels=len(item['features']),
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20, adapter_gradient_norm=adapter_gradient,
        zero_initialization_identity=True, missing_nonfinite_identity=True, baseline_pose_parity=True)
    run.save_json(args.output / 'checks.json', diagnostic)
    if json.loads((args.output / 'protocol.json').read_text())['scene_head'] == 'geometry':
        destination = run.HERE / 'results/geometry_retrain'
        destination.mkdir(parents=True, exist_ok=True)
        for path in args.output.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
        print(json.dumps(diagnostic, indent=2))
        return
    result = json.loads((args.output / 'result.json').read_text())
    extraction = json.loads((args.output / 'extraction.json').read_text())
    table = []
    for name, value in [('LEADER', result['metrics']['baseline']), ('① 简单门控', result['line1']),
        ('③ 全局与坐标上下文门控', result['metrics']['aligned']), ('③ 打乱训练对照', result['metrics']['shuffled']),
        ('③ 正常模型、打乱输入', result['metrics']['aligned_wrong']), ('③ 缺图', result['metrics']['aligned_missing'])]:
        table.append(f"| {name} | {value['mean'][0]:.6f} | {value['mean'][1]:.6f} | {value['median'][0]:.6f} | {value['median'][1]:.6f} | {value['p95'][0]:.6f} | {value['p95'][1]:.6f} | {value['successes']}/32 |")
    text = '\n'.join([
        '# ③ Node2Vec + coarse/refinement 与①对比', '',
        '结论：③未通过预设开发验证。平均位置误差比①降低约1.83%，但比LEADER增加约1.86%；打乱训练对照的平均位置误差还低于正常③，不能归因于正确视觉对应带来的稳健收益。', '',
        '| 方法 | 平移均值m | 旋转均值° | 平移中位m | 旋转中位° | 平移P95m | 旋转P95° | 1m/5°成功 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|', *table, '',
        '数据：沿用①投影修复后的64帧训练、32帧验证，均为2012-02-18；同一LiDAR缓存、代表点、有效mask、512D特征、GT、原MMRegressor/TRR/Matcher。随机种子2089，每个训练臂600步，Adam 1e-4，每步最多1024点。没有重新编码LiDAR，先前缓存重编码差异仍未解释，但不影响本次共用缓存的比较。', '',
        '③实现：NetVLAD只从907张其他日期训练图像检索top1，取对应pose Node2Vec 256D，与同一128D局部采样特征送入既有scrfacto模型。取768D末级隐藏特征和coarse/final各3D坐标，坐标固定除100，拼接774D，经LayerNorm+Linear映射128D，加到局部特征，再进入①原门控。场景坐标仅作上下文，未用于视觉PnP或替代LEADER定位。', '',
        '冻结：DeDoDe/PCA、NetVLAD、Node2Vec、场景回归、LEADER编码器和回归头。本轮重训适配器与门控，没有联合微调场景回归。使用原有仅重投影监督scrfacto+pose Node2Vec，未加入④的额外LiDAR三维监督，也未加可靠性头。', '',
        '公平性边界：③有184141个可训练参数，①为83393；额外冻结场景模型21050938参数，且③复用额外场景预训练。预算仅对齐本轮训练步数，不代表总计算量或模型容量相等。没有将所有新增模块逐一消融，本次只能判断这一组合配置。', '',
        '通过条件在评估前固定：平均平移相对①和LEADER均降低至少5%，旋转均值与平移P95相对两者恶化不超过5%，没有新增失败，且优于打乱训练。实际rescue=0、damage=0，所有方法32/32；③的平移P95也比①更差。失败后停止，不根据验证分数调参。', '',
        f"检查：零初始化严格等于LiDAR输入；缺图且图像含NaN时严格回退；适配器梯度非零；32帧缺图预测与基线误差一致。门控+适配器 {diagnostic['gate_ms']:.3f} ms/{diagnostic['voxels']} voxels；测量峰值显存 {diagnostic['peak_memory_mb']:.1f} MiB。",
        f"额外检索+场景头缓存提取平均 {np.mean([r['extra_seconds'] for r in extraction['records']])*1000:.3f} ms/帧，峰值显存 {extraction['peak_memory_mb']:.1f} MiB；不含缓存DeDoDe/NetVLAD提取，不是完整推理延迟。", '',
        '适用范围：已接触的本地开发数据、一个种子，不是盲测或完整NCLT结论；不能据此否定联合微调或多检索假设版本。', '',
        '复现：context_experiment.py prepare 使用rscore-l环境；train、evaluate、report使用egonn118环境。保留protocol、缓存来源哈希、模型权重、逐帧结果和检查。',
    ])
    (args.output / 'REPORT.md').write_text(text)
    destination = run.HERE / 'results/context_retrain'
    destination.mkdir(parents=True, exist_ok=True)
    for path in args.output.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    print(json.dumps(diagnostic, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'report'])
    parser.add_argument('--output', type=Path, default=Path('/home/zhang/leader-image-gate-context'))
    parser.add_argument('--scene-variant', choices=['scrfacto', 'geometry'], default='scrfacto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.checkpoint = run.WORKSPACE / 'research/image_gate_checkpoint'
    if args.stage == 'prepare':
        prepare(args)
        return
    run.ImageGate = ContextGate
    rows = json.loads((args.output / 'manifest.json').read_text())
    if args.stage == 'train':
        run.train(args, rows)
    elif args.stage == 'evaluate':
        evaluate(args, rows)
    else:
        report(args, rows)


if __name__ == '__main__':
    main()
