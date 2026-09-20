import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from model import QueryFusion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(HERE.parent / 'dinov2_fusion'))
spec = importlib.util.spec_from_file_location('dino_baseline', HERE.parent / 'dinov2_fusion/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(WORKSPACE / 'glace-local/code/tools'))
from full_pool_robust_v1 import full_pool_refine

DINO = Path('/home/zhang/dinov2-leader')
ARMS = ['lidar_finetune', 'query', 'gaussian']


def offsets():
    y, x = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing='ij')
    return torch.stack([x, y], -1).reshape(25, 2).float()


def prepare(args):
    rows = json.loads((DINO / 'manifest.json').read_text())
    assert len(rows) == 96 and sum(r['split'] == 'train' for r in rows) == 64
    old_protocol = json.loads((DINO / 'protocol.json').read_text())
    for row in rows:
        stem = row['frame_id']
        for name, path in dict(lidar=base.SOURCE / 'lidar' / (stem + '.npz'),
                mapping=base.SOURCE / 'projection_audit/mapping' / (stem + '.npz'), image=Path(row['image'])).items():
            if base.sha(path) != old_protocol['input_hashes'][stem][name]:
                raise ValueError(f'Frozen input changed: {stem}/{name}')
    protocol = dict(seed=2089, steps=600, points_per_step=1024, learning_rate_fusion=.0001,
        learning_rate_decoder=.0001, modality_dropout=.1, training_frames=64, validation_frames=32,
        source_protocol_sha256=base.sha(DINO / 'protocol.json'), pca_sha256=base.sha(DINO / 'pca.pt'),
        checkpoint_sha256=base.sha(WORKSPACE / 'research/image_gate_checkpoint/model.safetensors'),
        refinement_source_sha256=base.sha(WORKSPACE / 'glace-local/code/tools/full_pool_robust_v1.py'),
        matcher_source_sha256=base.sha(REPO / 'models/sc2pcr.py'),
        frozen=['DINOv2 ViT-B/14', 'PCA', 'LEADER encoder', 'LEADER decoder except pred_out'],
        trainable=['decoder.pred_out in every trained arm', 'query attention and residual gate in visual arms'],
        arms=ARMS, visual_grid=dict(size=[5, 5], spacing_pixels=14, query_dim=64, gaussian_sigma_pixels=14),
        query='512D LEADER voxel feature; no GT/world pose as input; 25 nearby image descriptors as keys/values',
        gaussian='Fixed isotropic image-plane log prior -0.5*(dx^2+dy^2) in patch units; not a 3D Gaussian map',
        coordinates='Original voxel centers preserved; original LEADER head predicts world coordinates, no new points.',
        visibility='Audited raw representative center visibility; neighbor raster bounds and undistortion mask; no invented depth at neighbor pixels',
        pose_solver='Same upstream SC2 top50% followed by original full_pool_refine thresholds [1.2,0.6] for all arms',
        sampling='Frame schedule numpy seed2089; each step point schedule independent seed2089+step shared by all arms',
        evaluation=['baseline', *ARMS, 'gaussian_wrong', 'gaussian_missing', 'query_missing'],
        scope='Existing 64/32 development split, not original907/303/148. 2012-02-18 participated in LEADER pretraining.',
        success_thresholds=['1m/2deg', '1m/5deg'],
        pass_rule='Visual arm must improve both MPE and MOE over original two-stage AND same-budget lidar_finetune; no loss of 1m/2deg successes; no validation tuning.')
    path = args.root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a fresh output directory')
    base.save(path, protocol)
    base.save(args.root / 'manifest.json', rows)


def extract(args, rows):
    model = torch.hub.load(str(DINO / 'dinov2-main'), 'dinov2_vitb14', source='local', pretrained=False)
    model.load_state_dict(torch.load(DINO / 'weights/dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True))
    model = model.cuda().eval().requires_grad_(False)
    pca = torch.load(DINO / 'pca.pt', weights_only=True)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    mask = torch.tensor(np.load(WORKSPACE / 'glace-local/data/valid_mask.npy'), device='cuda').float()[None, None]
    mask = F.interpolate(mask, size=(476, 630), mode='nearest')
    off = offsets().cuda()
    destination = args.root / 'visual'
    destination.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        target = destination / (row['frame_id'] + '.npz')
        if target.exists():
            continue
        start = time.perf_counter()
        image = Image.open(row['image']).convert('RGB').resize((630, 476), Image.Resampling.BILINEAR)
        image = torch.tensor(np.array(image), device='cuda').permute(2, 0, 1)[None].float() / 255
        with np.load(base.SOURCE / 'projection_audit/mapping' / (row['frame_id'] + '.npz')) as mapping:
            uv = torch.tensor(mapping['uv'], device='cuda')
            h, w = mapping['image_hw']
            center_valid = torch.tensor(mapping['valid'], device='cuda')
            with np.load(base.SOURCE / 'lidar' / (row['frame_id'] + '.npz')) as lidar:
                assert np.array_equal(mapping['localization_xyz'], lidar['source'])
        uv = (uv + .5) * uv.new_tensor([630 / w, 476 / h]) - .5
        locations = uv[:, None] + 14 * off[None]
        flat = locations.reshape(-1, 2)
        valid = (locations[..., 0] >= 0) & (locations[..., 0] <= 629)
        valid &= (locations[..., 1] >= 0) & (locations[..., 1] <= 475)
        valid &= base.sample_patches(mask, flat, (476, 630)).reshape(-1, 25) > .999
        valid &= center_valid[:, None]
        with torch.inference_mode(), torch.autocast('cuda'):
            tokens = model.forward_features((image - mean) / std)['x_norm_patchtokens']
        with torch.inference_mode():
            dense = tokens.reshape(1, 34, 45, 768).permute(0, 3, 1, 2).float()
            compressed = F.conv2d(dense, pca['weight'].cuda()[:, :, None, None], pca['bias'].cuda())
            sampled = base.sample_patches(compressed, flat, (476, 630)).reshape(-1, 25, 128)
            sampled = torch.where(valid[..., None], sampled, torch.zeros_like(sampled))
        if not torch.isfinite(sampled).all():
            raise ValueError('Nonfinite neighborhood features')
        torch.cuda.synchronize()
        np.savez(target, image=sampled.half().cpu().numpy(), mask=valid.cpu().numpy(), seconds=time.perf_counter()-start)
        print(f'neighborhoods {index+1}/96 visible={int(valid.any(-1).sum())}/{len(uv)}', flush=True)


def frame(args, row):
    with np.load(base.SOURCE / 'lidar' / (row['frame_id'] + '.npz')) as data:
        item = {k: torch.tensor(data[k], device='cuda', dtype=torch.float32)
                for k in ['features', 'source', 'prediction', 'target', 'GT', 'center']}
    with np.load(args.root / 'visual' / (row['frame_id'] + '.npz')) as data:
        item['image'] = torch.tensor(data['image'], device='cuda', dtype=torch.float32)
        item['mask'] = torch.tensor(data['mask'], device='cuda')
    return item


def train(args, rows):
    training = [r for r in rows if r['split'] == 'train']
    schedule = np.random.default_rng(2089).integers(len(training), size=600)
    trr = base.source_class(REPO / 'run_mink.py', 'TRR')(scale=10.)
    off = offsets().cuda()
    for arm in ARMS:
        path = args.root / (arm + '.pt')
        if path.exists():
            continue
        torch.manual_seed(2089)
        decoder = base.decoder()
        decoder.pred_out.requires_grad_(True)
        fusion = None if arm == 'lidar_finetune' else QueryFusion(gaussian=arm == 'gaussian').cuda()
        params = [dict(params=decoder.pred_out.parameters(), lr=.0001)]
        if fusion is not None:
            params.append(dict(params=fusion.parameters(), lr=.0001))
        optimizer = torch.optim.Adam(params)
        logs = []
        start = time.perf_counter()
        for step, index in enumerate(schedule):
            item = frame(args, training[index])
            rng = np.random.default_rng(2089+step)
            indices = rng.permutation(len(item['features']))[:1024]
            lidar, image, mask, target = [item[k][indices] for k in ['features', 'image', 'mask', 'target']]
            if rng.random() < .1:
                mask = torch.zeros_like(mask)
            optimizer.zero_grad(set_to_none=True)
            feature = lidar if fusion is None else fusion(lidar, image, mask, off)
            pred = decoder(feature)
            loss = trr(target, pred[:, :3], pred[:, 3], torch.zeros(len(pred), dtype=torch.long, device='cuda'))[0].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm} step{step}: nonfinite loss')
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == 599:
                logs.append(dict(step=step+1, loss=float(loss.detach()), seconds=time.perf_counter()-start))
                base.save(args.root / (arm+'_training.json'), logs)
                print(f'{arm} {step+1}/600 loss={loss.item():.5f}', flush=True)
        torch.save(dict(decoder=decoder.cpu().state_dict(), fusion=None if fusion is None else fusion.cpu().state_dict()), path)


def error(pose, gt):
    trans = float((pose[:3, 3] - gt[:3, 3]).norm())
    cosine = ((pose[:3, :3].T @ gt[:3, :3]).trace() - 1) / 2
    return [trans, float(torch.rad2deg(cosine.clamp(-1, 1).acos()))]


def verify_refinement(args):
    path = next((WORKSPACE / 'glace-local/cache/lidar_pools').glob('*.npz'))
    with np.load(path) as data:
        pose = data['leader'] @ np.linalg.inv(data['T_corr'])
        pose[:3, 3] -= data['center_t']
        refined, _ = full_pool_refine(torch.tensor(pose, dtype=torch.float32, device='cuda'),
                torch.tensor(data['c_local_all'], dtype=torch.float32, device='cuda'),
                torch.tensor(data['c_pred_all'], dtype=torch.float32, device='cuda'))
        world = refined.cpu().numpy().astype(np.float64)
        world[:3, 3] += data['center_t']
        world = world @ data['T_corr']
        t = float(np.linalg.norm(world[:3, 3] - data['v1_two_stage'][:3, 3]))
        r = float(np.max(np.abs(world[:3, :3] - data['v1_two_stage'][:3, :3])))
        if t > .001 or r > 1e-4:
            raise ValueError(f'Original two-stage did not reproduce: {t}, {r}')
    base.save(args.root / 'refinement_check.json', dict(frame=path.stem, translation_difference_m=t, rotation_matrix_max_difference=r))


def evaluate(args, rows):
    from models.sc2pcr import Matcher
    verify_refinement(args)
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    models = {}
    off = offsets().cuda()
    for arm in ARMS:
        state = torch.load(args.root / (arm+'.pt'), map_location='cuda')
        decoder = base.decoder()
        decoder.load_state_dict(state['decoder'])
        fusion = None if arm == 'lidar_finetune' else QueryFusion(gaussian=arm == 'gaussian').cuda().eval()
        if fusion is not None:
            fusion.load_state_dict(state['fusion'])
        models[arm] = (decoder, fusion)
    records = []
    for index, row in enumerate(r for r in rows if r['split'] == 'val'):
        item = frame(args, row)
        with torch.inference_mode():
            predictions = dict(baseline=item['prediction'])
            attention_stats = {}
            for arm, (decoder, fusion) in models.items():
                if fusion is None:
                    predictions[arm] = decoder(item['features'])
                    continue
                feature, attention = fusion(item['features'], item['image'], item['mask'], off, return_attention=True)
                predictions[arm] = decoder(feature)
                predictions[arm+'_missing'] = decoder(fusion(item['features'], item['image'], torch.zeros_like(item['mask']), off))
                if not torch.equal(predictions[arm+'_missing'], decoder(item['features'])):
                    raise ValueError('Missing image must exactly equal this arm\'s adapted decoder')
                visible = item['mask'].any(-1)
                entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
                attention_stats[arm] = dict(entropy=float(entropy[visible].mean()), center_weight=float(attention[visible, 12].mean()))
                if arm == 'gaussian':
                    image = item['image'].clone()
                    selected = torch.where(visible)[0]
                    image[selected] = image[selected.roll(1)]
                    predictions['gaussian_wrong'] = decoder(fusion(item['features'], image, item['mask'], off))
            errors, initial_errors, supports = {}, {}, {}
            for arm, prediction in predictions.items():
                torch.manual_seed(2089+index)
                selected = prediction[:, 3].topk(max(min(50, len(prediction)), int(.5*len(prediction)))).indices
                initial = matcher.estimator(item['source'][selected][None], prediction[selected, :3][None])[0]
                refined, support = full_pool_refine(initial, item['source'], prediction[:, :3])
                initial[:3, 3] += item['center']
                refined[:3, 3] += item['center']
                errors[arm] = error(refined, item['GT'])
                initial_errors[arm] = error(initial, item['GT'])
                supports[arm] = support
        records.append(dict(frame_id=row['frame_id'], errors=errors, initial_errors=initial_errors,
                            refinement_support=supports, attention=attention_stats))
        base.save(args.root / 'records.json', records)
        print(f'validation {index+1}/32 baseline={errors["baseline"]} query={errors["query"]} gaussian={errors["gaussian"]}', flush=True)
    metrics = {arm: base.metrics([r['errors'][arm] for r in records]) for arm in records[0]['errors']}
    passed = {}
    for arm in ['query', 'gaussian']:
        passed[arm] = all(metrics[arm]['MPE'] < metrics[control]['MPE'] and metrics[arm]['MOE'] < metrics[control]['MOE']
            and metrics[arm]['success_1m_2deg'] >= metrics[control]['success_1m_2deg'] for control in ['baseline', 'lidar_finetune'])
    base.save(args.root / 'result.json', dict(metrics=metrics, passed=passed))


def report(args):
    import shutil
    result = json.loads((args.root / 'result.json').read_text())
    records = json.loads((args.root / 'records.json').read_text())
    samples = np.random.default_rng(2089).integers(0, len(records), size=(10000, len(records)))
    paired = {}
    for arm in ['query', 'gaussian']:
        delta = np.array([r['errors'][arm] for r in records]) - np.array([r['errors']['lidar_finetune'] for r in records])
        paired[arm] = dict(delta_MPE_MOE=delta.mean(0).tolist(), bootstrap_95ci=np.percentile(delta[samples].mean(1), [2.5, 97.5], axis=0).tolist())
    base.save(args.root / 'paired.json', paired)
    labels = dict(baseline='原始 LEADER＋two-stage', lidar_finetune='同预算纯 LiDAR 回归头微调',
        query='Query 融合', gaussian='Query＋高斯先验', query_missing='Query 模型：去掉图像',
        gaussian_missing='高斯模型：去掉图像', gaussian_wrong='高斯模型：打乱图像对应')
    lines = ['# LEADER Query / Gaussian 特征融合结果', '',
        '64 帧训练、32 帧开发验证；冻结 DINOv2 与 LEADER 编码器，每个训练配置固定 600 步。',
        '所有结果均使用相同的原始 SC2＋full-pool two-stage 精化；不是原来的 148 帧测试结果。', '',
        '| 方法 | MPE (m) | MOE (°) | 成功 <1m/2° |', '|---|---:|---:|---:|']
    for arm, m in result['metrics'].items():
        lines.append(f'| {labels[arm]} | {m["MPE"]:.6f} | {m["MOE"]:.6f} | {m["success_1m_2deg"]}/32 |')
    lines += ['', '## 判断', '',
        'Query 与高斯版本均未满足训练前冻结的通过条件：MPE 和 MOE 同时优于原始 two-stage 及同预算 LiDAR 微调，且不损失 1m/2° 成功帧。',
        f'通过状态：{result["passed"]}',
        '相对原始模型的平移改善大部分也能由纯 LiDAR 回归头微调得到；视觉版本相对该对照仅约 1mm 的平均平移差别，同时旋转误差增加。',
        '去掉或打乱视觉输入没有造成明显退化；高斯模型打乱对应后两个均值甚至略低，当前没有证据证明网络有效利用了正确的视觉对应关系。',
        '高斯版本与普通 Query 很接近，不能据此宣称高斯关联有效，也不能据此否定其他高斯表示或更充分训练。', '',
        '## 实现与验证边界', '',
        '缺失图像实验使用各自已微调的回归器，只保证融合特征退回原几何特征，不保证恢复原始预训练模型位姿。',
        '高斯仅为固定图像平面空间先验，不是 3D Gaussian 地图，也不是校准后的不确定性。原有局部点坐标与 correspondence 数量不变。',
        'GT 仅用于训练标签和误差统计，不参与图像关联或位姿求解。四项针对性实现检查通过；原始 two-stage 精化与历史缓存的差异见 refinement_check.json。',
        '本日期参与过 LEADER 预训练且已有开发接触；这是单种子小规模诊断，不能证明跨日期泛化。',
        'paired.json 给出逐帧配对 bootstrap 区间；邻近帧存在相关性，该区间仅作描述，不作为独立样本显著性结论。']
    (args.root / 'report.md').write_text('\n'.join(lines)+'\n')
    destination = HERE / 'results'
    destination.mkdir(exist_ok=True)
    for name in ['protocol.json', 'manifest.json', 'records.json', 'result.json', 'paired.json', 'refinement_check.json', 'report.md', *[a+'_training.json' for a in ARMS]]:
        shutil.copy2(args.root / name, destination / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extract', 'train', 'evaluate', 'report', 'check'])
    parser.add_argument('--root', type=Path, default=Path('/home/zhang/leader-query-fusion'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'check':
        import unittest
        from test_model import QueryTests
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueryTests))
        if not result.wasSuccessful():
            raise RuntimeError('Checks failed')
        return
    if args.stage in ['prepare', 'report']:
        globals()[args.stage](args)
    else:
        globals()[args.stage](args, json.loads((args.root / 'manifest.json').read_text()))


if __name__ == '__main__':
    main()
