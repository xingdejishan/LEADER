import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import run
from fusion import sample_map
from lpgrf import LoFTRLocal, LPGRF, Distillation


ORIGINAL = Path('/home/zhang/leader-image-gate-raw')
MAPPING = Path('/home/zhang/leader-image-gate/projection_audit/mapping')
ARMS = ['aligned', 'no_distill', 'shuffled', 'lidar_only']


def prepare(args, rows):
    from PIL import Image
    import urllib.request
    checkpoint = args.output / 'assets/loftr_outdoor.ckpt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        pending = checkpoint.with_suffix('.pending')
        urllib.request.urlretrieve('https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt', pending)
        pending.replace(checkpoint)
    model = LoFTRLocal().load_pretrained(args.output / 'assets/loftr_outdoor.ckpt').cuda().eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['stem', 'visual']:
        (args.output / name).mkdir(exist_ok=True)
    if not (args.output / 'lidar').exists():
        (args.output / 'lidar').symlink_to((ORIGINAL / 'lidar').resolve(), target_is_directory=True)
    run.save_json(args.output / 'manifest.json', rows)
    protocol = dict(method='② LP-GRF with LoFTR layer3/PCA128 and differentiable final image block',
        hypothesis='LiDAR teacher feature alignment can adapt image representation to localization while bounded residual protects geometric features.',
        falsification='Correct-image fusion must improve final pose beyond both original LEADER and decoder-only finetuning; intermediate loss reduction alone is insufficient.',
        stages=[dict(steps=320, epochs=5, train=['fusion', 'two projectors'], lr=.001),
                dict(steps=600, lr_fusion=.001, lr_projectors=.001, lr_decoder=.0001, lr_image_layer3=.00001)],
        distillation_weight=.05, modality_dropout=.1, alpha_initial=.1, seed=2089,
        train_frames=64, val_frames=32, points_per_step=1024, arms=ARMS,
        teacher='Frozen original LEADER predictions, confidence top50% intersect image valid mask; stopgrad LiDAR features',
        image='LoFTR outdoor grayscale at 632x480; frozen stem through layer2 cached float16, train layer3 online; BatchNorm running statistics frozen',
        pca='Fit only on the 64 training images, 512 valid dense cells each; fixed centered linear PCA256->128',
        visibility='Keep audited representative raw Cartesian mapping, 4px raw-scan zbuffer and 0.5m tolerance; override obsolete document voxel centers/8px zbuffer',
        missing='Exact feature fallback; finetuned decoder changes pose fallback relative to original checkpoint. Evaluate both explicitly.',
        pass_rule='aligned mean translation <=95% of original and decoder-only; mean rotation and p95 translation <=105% of both; zero damage and aligned better than shuffled.',
        scope='Local previously touched development, single seed. Not full NCLT.',
        checkpoint_sha256=run.digest(args.output / 'assets/loftr_outdoor.ckpt'),
        lidar_protocol_sha256=run.digest(ORIGINAL / 'protocol.json'))
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    run.save_json(protocol_path, protocol)
    sums = torch.zeros(256, dtype=torch.float64, device='cuda')
    squares = torch.zeros(256, 256, dtype=torch.float64, device='cuda')
    count = 0
    generator = torch.Generator(device='cuda').manual_seed(2089)
    mask = torch.tensor(np.load(run.WORKSPACE / 'glace-local/data/valid_mask.npy'), device='cuda')[None, None].float()
    for index, row in enumerate(rows):
        image = Image.open(row['image']).convert('L').resize((632, 480), Image.Resampling.BILINEAR)
        image = torch.tensor(np.array(image), device='cuda')[None, None].float()/255
        with torch.no_grad():
            stem = model.stem(image).half()
            dense = model.layer3(stem.float())
        np.save(args.output / 'stem' / (row['frame_id'] + '.npy'), stem.cpu().numpy())
        if row['split'] == 'train':
            valid = F.interpolate(mask, size=dense.shape[-2:], mode='nearest').flatten() > .999
            features = dense[0].permute(1, 2, 0).reshape(-1, 256)[valid]
            features = features[torch.randperm(len(features), generator=generator, device='cuda')[:512]].double()
            sums += features.sum(0)
            squares += features.T @ features
            count += len(features)
        if index % 20 == 0:
            print(f'LoFTR stem {index+1}/{len(rows)}', flush=True)
    mean = sums/count
    covariance = squares/count - mean[:, None]*mean[None]
    values, vectors = torch.linalg.eigh(covariance)
    weight = vectors[:, -128:].T.flip(0).float()
    pca = dict(weight=weight[:, :, None, None].cpu(), bias=(-(weight@mean.float())).cpu())
    torch.save(pca, args.output / 'pca.pt')
    run.save_json(args.output / 'pca.json', dict(samples=count, training_only=True, retained_variance=float(values[-128:].sum()/values.sum())))
    for row in rows:
        with torch.no_grad():
            stem = torch.tensor(np.load(args.output / 'stem' / (row['frame_id'] + '.npy')), device='cuda').float()
            dense = F.conv2d(model.layer3(stem), pca['weight'].cuda(), pca['bias'].cuda())
            with np.load(MAPPING / (row['frame_id'] + '.npz')) as mapping:
                assert np.array_equal(mapping['image_hw'], [480, 632])
                uv = torch.tensor(mapping['uv'], device='cuda')
                valid = mapping['valid']
                sample = sample_map(dense, uv, (480, 632)).cpu().numpy()
            with np.load(ORIGINAL / 'visual' / (row['frame_id'] + '.npz')) as old:
                assert np.array_equal(valid, old['valid'])
            sample[~valid] = 0
            assert np.isfinite(sample).all()
            np.savez(args.output / 'visual' / (row['frame_id'] + '.npz'), image=sample, valid=valid)
    print('LoFTR/PCA preparation complete', flush=True)


def load_frame(args, row):
    item = run.frame(args, row)
    with np.load(MAPPING / (row['frame_id'] + '.npz')) as mapping:
        assert np.array_equal(mapping['localization_xyz'], item['source'].cpu().numpy())
        item['uv'] = torch.tensor(mapping['uv'], device='cuda')
        item['distance'] = torch.tensor(np.linalg.norm(mapping['projection_xyz'], axis=-1), device='cuda')
    reliable = torch.zeros_like(item['valid'])
    reliable[item['prediction'][:, 3].topk(max(1, len(reliable)//2)).indices] = True
    item['reliable'] = reliable
    return item


def image_features(args, row, item, image_model, pca):
    stem = torch.tensor(np.load(args.output / 'stem' / (row['frame_id'] + '.npy')), device='cuda').float()
    dense = F.conv2d(image_model.layer3(stem), pca['weight'], pca['bias'])
    return sample_map(dense, item['uv'], (480, 632))


def components(args):
    torch.manual_seed(2089)
    leader = run.load_leader(args)
    image = LoFTRLocal().load_pretrained(args.output / 'assets/loftr_outdoor.ckpt').cuda().eval().requires_grad_(False)
    fusion = LPGRF().cuda()
    distill = Distillation().cuda()
    pca = {k: v.cuda() for k, v in torch.load(args.output / 'pca.pt').items()}
    return leader, image, fusion, distill, pca


def train(args, rows):
    training = [r for r in rows if r['split'] == 'train']
    for arm in ARMS:
        leader, image, fusion, distill, pca = components(args)
        trr = run.official_trr()
        generator = np.random.default_rng(2089)
        stage1 = np.concatenate([generator.permutation(len(training)) for _ in range(5)])
        stage2 = generator.integers(len(training), size=600)
        logs = []
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        gradient_seen = dict(trr_image=0., distillation_image=0.)
        for stage, schedule in [(1, stage1), (2, stage2)]:
            torch.cuda.manual_seed(2089 + stage)
            if arm == 'lidar_only' and stage == 1:
                continue
            if stage == 2:
                leader.decoder.requires_grad_(True)
                image.layer3.requires_grad_(arm != 'lidar_only')
            groups = [dict(params=fusion.parameters(), lr=.001), dict(params=distill.parameters(), lr=.001)]
            if stage == 2:
                groups += [dict(params=leader.decoder.parameters(), lr=.0001)]
                if arm != 'lidar_only':
                    groups += [dict(params=image.layer3.parameters(), lr=.00001)]
            optimizer = torch.optim.Adam(groups)
            for step, index in enumerate(schedule):
                row = training[index]
                item = load_frame(args, row)
                indices = torch.randperm(len(item['features']), device='cuda')[:1024]
                if stage == 2 and arm != 'lidar_only':
                    visual = image_features(args, row, item, image, pca)
                else:
                    visual = item['image']
                visual = visual[indices]
                features, valid, distance, reliable, target = [item[k][indices] for k in ['features', 'valid', 'distance', 'reliable', 'target']]
                if arm == 'shuffled':
                    visual = visual.clone()
                    v = torch.where(valid)[0]
                    visual[v] = visual[v.roll(1)]
                if arm == 'lidar_only' or np.random.default_rng(2089 + stage*10000 + step).random() < .1:
                    valid = torch.zeros_like(valid)
                fused = features if arm == 'lidar_only' else fusion(features, visual, valid, distance)
                prediction = leader.decoder(fused)
                task_loss = trr(target, prediction[:, :3], prediction[:, 3], torch.zeros(len(indices), device='cuda', dtype=torch.long))[0].mean()
                auxiliary = distill(features, visual, valid, reliable) if arm not in ['no_distill', 'lidar_only'] else task_loss*0
                loss = task_loss + .05*auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{arm} {stage} {step}')
                if stage == 2 and arm == 'aligned' and valid.any() and gradient_seen['trr_image'] == 0:
                    parameter = image.layer3[0].conv1.weight
                    gradient_seen['trr_image'] = float(torch.autograd.grad(task_loss, parameter, retain_graph=True)[0].norm())
                    gradient_seen['distillation_image'] = float(torch.autograd.grad(auxiliary, parameter, retain_graph=True)[0].norm())
                    assert min(gradient_seen.values()) > 0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 100 == 0 or step == len(schedule)-1:
                    logs.append(dict(stage=stage, step=step, trr=float(task_loss.detach()), distill=float(auxiliary.detach()), alpha=float(fusion.alpha.detach())))
                    print(f'{arm} stage{stage} {step+1}/{len(schedule)} TRR={task_loss.item():.4f}', flush=True)
                    run.save_json(args.output / (arm + '_training.json'), logs)
            if stage == 1:
                torch.save(dict(fusion=fusion.state_dict(), distill=distill.state_dict()), args.output / (arm + '_stage1.pt'))
        torch.save(dict(fusion=fusion.state_dict(), distill=distill.state_dict(), decoder=leader.decoder.state_dict(), layer3=image.layer3.state_dict()), args.output / (arm+'.pt'))
        run.save_json(args.output / (arm+'_training_stats.json'), dict(seconds=time.perf_counter()-start, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,
            image_gradients=gradient_seen, fusion_parameters=sum(p.numel() for p in fusion.parameters()), distill_parameters=sum(p.numel() for p in distill.parameters()),
            decoder_parameters=sum(p.numel() for p in leader.decoder.parameters()), image_trainable_parameters=sum(p.numel() for p in image.layer3.parameters())))


def pose_error(matcher, item, prediction):
    torch.manual_seed(2089)
    n = len(prediction)
    selected = prediction[:, 3].topk(max(min(50, n), int(.5*n))).indices
    pose = matcher.estimator(item['source'][selected][None], prediction[selected, :3][None])[0]
    pose[:3, 3] += item['center']
    translation = (pose[:3, 3]-item['GT'][:3, 3]).norm().item()
    cosine = ((pose[:3, :3].T@item['GT'][:3, :3]).trace()-1)/2
    return [translation, torch.rad2deg(cosine.clamp(-1, 1).acos()).item()]


def evaluate(args, rows):
    from models.sc2pcr import Matcher
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    errors = {k: [] for k in ['baseline', *ARMS, 'aligned_wrong', 'aligned_missing']}
    timings = {}
    records = {r['frame_id']: dict(frame_id=r['frame_id'], errors={}) for r in rows if r['split']=='val'}
    for arm in ARMS:
        leader, image, fusion, _, pca = components(args)
        saved = torch.load(args.output / (arm+'.pt'))
        leader.decoder.load_state_dict(saved['decoder'])
        image.layer3.load_state_dict(saved['layer3'])
        fusion.load_state_dict(saved['fusion'])
        timings[arm] = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for row in [r for r in rows if r['split']=='val']:
                item = load_frame(args, row)
                torch.cuda.synchronize()
                started = time.perf_counter()
                visual = image_features(args, row, item, image, pca)
                valid = item['valid']
                if arm == 'shuffled':
                    visual = visual.clone()
                    v = torch.where(valid)[0]
                    visual[v] = visual[v.roll(1)]
                fused = item['features'] if arm == 'lidar_only' else fusion(item['features'], visual, valid, item['distance'])
                predictions = {arm: leader.decoder(fused)}
                torch.cuda.synchronize()
                timings[arm].append(time.perf_counter()-started)
                if arm == 'aligned':
                    wrong = visual.clone()
                    v = torch.where(valid)[0]
                    wrong[v] = wrong[v.roll(1)]
                    missing = fusion(item['features'], torch.full_like(visual, float('nan')), torch.zeros_like(valid), item['distance'])
                    assert torch.equal(missing, item['features'])
                    predictions.update(baseline=item['prediction'], aligned_wrong=leader.decoder(fusion(item['features'], wrong, valid, item['distance'])), aligned_missing=leader.decoder(missing))
                for key, prediction in predictions.items():
                    error = pose_error(matcher, item, prediction)
                    errors[key].append(error)
                    records[row['frame_id']]['errors'][key] = error
        print('evaluated '+arm, flush=True)
        run.save_json(args.output/(arm+'_inference_stats.json'), dict(mean_seconds=float(np.mean(timings[arm])),
            peak_memory_mb=torch.cuda.max_memory_allocated()/2**20, includes='cached stem loading, image layer3/PCA/sampling, fusion and decoder',
            excludes='LiDAR encoder, image stem, matcher; lidar_only timing also includes unnecessary image feature computation'))
    metrics = {k: run.metrics(v) for k, v in errors.items()}
    a, b = np.asarray(errors['aligned']), np.asarray(errors['baseline'])
    sa, sb = (a[:, 0]<1)&(a[:, 1]<5), (b[:, 0]<1)&(b[:, 1]<5)
    damage, rescue = int((~sa&sb).sum()), int((sa&~sb).sum())
    current = metrics['aligned']
    passed = all(current['mean'][0]<=.95*metrics[k]['mean'][0] and current['mean'][1]<=1.05*metrics[k]['mean'][1]
        and current['p95'][0]<=1.05*metrics[k]['p95'][0] for k in ['baseline', 'lidar_only'])
    passed &= damage == 0 and current['mean'][0]<metrics['shuffled']['mean'][0]
    old = json.loads((ORIGINAL/'validation_frames.json').read_text())
    assert np.allclose([r['errors']['baseline'] for r in old], errors['baseline'], atol=1e-6)
    run.save_json(args.output/'validation_frames.json', list(records.values()))
    run.save_json(args.output/'result.json', dict(metrics=metrics, damage=damage, rescue=rescue, passed=bool(passed),
        scope='32 local development frames; stage2 finetunes image layer3 and decoder; single seed'))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--output', type=Path, default=Path('/home/zhang/leader-image-gate-lpgrf'))
    args = parser.parse_args()
    args.checkpoint = run.WORKSPACE/'research/image_gate_checkpoint'
    torch.set_num_threads(4)
    rows = json.loads((ORIGINAL/'manifest.json').read_text())
    globals()[args.stage](args, rows)


if __name__ == '__main__':
    main()
