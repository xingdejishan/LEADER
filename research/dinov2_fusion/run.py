import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn, Tensor
from PIL import Image
from fusion import FeatureFusion, sample_patches

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
SOURCE = Path('/home/zhang/leader-image-gate')
sys.path.insert(0, str(REPO))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def source_class(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    scope = dict(torch=torch, nn=nn, np=np, Tensor=Tensor)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def decoder():
    from safetensors.torch import load_file
    model = source_class(REPO / 'models/model_mink.py', 'MMRegressor')().cuda().eval()
    checkpoint = load_file(str(WORKSPACE / 'research/image_gate_checkpoint/model.safetensors'))
    model.load_state_dict({k[len('decoder.'):]: v for k, v in checkpoint.items() if k.startswith('decoder.')}, strict=True)
    return model.requires_grad_(False)


def prepare(args):
    rows = json.loads((SOURCE / 'manifest.json').read_text())
    assert sum(r['split'] == 'train' for r in rows) == 64
    assert sum(r['split'] == 'val' for r in rows) == 32
    assert len({r['frame_id'] for r in rows}) == 96
    weights = args.root / 'weights/dinov2_vitb14_pretrain.pth'
    protocol = dict(seed=2089, steps=600, learning_rate=.0001, points_per_step=1024,
        train_frames=64, val_frames=32, image_size_hw=[476, 630], model='dinov2_vitb14',
        weights_sha256=sha(weights), dinov2_source_zip_sha256=sha(args.root / 'dinov2.zip'),
        pretrained_leader_sha256=sha(WORKSPACE / 'research/image_gate_checkpoint/model.safetensors'),
        frozen=['DINOv2', 'LEADER encoder', 'LEADER decoder', 'training-only PCA'],
        trainable='83393-parameter feature residual gate before original LEADER MMRegressor',
        projection='Reuse audited real scan representative per output voxel; existing 4px zbuffer, 0.5m tolerance and undistortion mask. No GT in feature projection.',
        pca='64 training frames only, at most 512 visible point descriptors per frame, centered 768->128',
        pose_solver='Unmodified upstream LEADER SC2 Matcher / confidence top50%; NOT v1 two-stage.',
        arms=['baseline', 'aligned', 'shuffled', 'aligned_wrong', 'aligned_missing'],
        scope='64/32 local development diagnostic on 2012-02-18, a LEADER pretraining date; NOT the 907/303/148 split or blind test.',
        pass_rule='Aligned MPE <=95% baseline, MOE and translation p95 <=105% baseline, no lost 1m/5deg successes, aligned beats shuffled MPE.',
        input_hashes={r['frame_id']: {name: sha(path) for name, path in {
            'lidar': SOURCE / 'lidar' / (r['frame_id'] + '.npz'),
            'mapping': SOURCE / 'projection_audit/mapping' / (r['frame_id'] + '.npz'),
            'image': Path(r['image'])}.items()} for r in rows})
    existing = args.root / 'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError('Frozen protocol changed; use another output directory')
    save(existing, protocol)
    save(args.root / 'manifest.json', rows)
    return rows


def extract(args, rows):
    model = torch.hub.load(str(args.root / 'dinov2-main'), 'dinov2_vitb14', source='local', pretrained=False)
    model.load_state_dict(torch.load(args.root / 'weights/dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    raw_dir = args.root / 'raw'
    raw_dir.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        target = raw_dir / (row['frame_id'] + '.npz')
        if target.exists():
            continue
        started = time.perf_counter()
        image = Image.open(row['image']).convert('RGB').resize((630, 476), Image.Resampling.BILINEAR)
        image = torch.as_tensor(np.array(image), device='cuda').permute(2, 0, 1)[None].float() / 255
        with np.load(SOURCE / 'projection_audit/mapping' / (row['frame_id'] + '.npz')) as mapping:
            uv = torch.tensor(mapping['uv'], device='cuda')
            old_hw = mapping['image_hw']
            valid = mapping['valid'].astype(bool)
            with np.load(SOURCE / 'lidar' / (row['frame_id'] + '.npz')) as lidar:
                if not np.array_equal(mapping['localization_xyz'], lidar['source']):
                    raise ValueError('Projection/LEADER row mismatch')
        uv = (uv + .5) * uv.new_tensor([630 / old_hw[1], 476 / old_hw[0]]) - .5
        with torch.inference_mode(), torch.autocast('cuda'):
            tokens = model.forward_features((image - mean) / std)['x_norm_patchtokens']
        patches = tokens.reshape(1, 34, 45, 768).permute(0, 3, 1, 2)
        with torch.inference_mode():
            features = sample_patches(patches, uv[valid], (476, 630))
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite DINOv2 features')
        torch.cuda.synchronize()
        np.savez(target, features=features.half().cpu().numpy(), valid=valid,
                 seconds=time.perf_counter()-started)
        print(f'DINOv2 {i+1}/{len(rows)} visible={valid.sum()}/{len(valid)}', flush=True)
    del model
    torch.cuda.empty_cache()
    pca_path = args.root / 'pca.pt'
    if not pca_path.exists():
        rng = np.random.default_rng(2089)
        total = np.zeros(768, np.float64)
        gram = np.zeros((768, 768), np.float64)
        count = 0
        for row in rows:
            if row['split'] != 'train':
                continue
            with np.load(raw_dir / (row['frame_id'] + '.npz')) as data:
                x = data['features'].astype(np.float64)
            x = x[rng.choice(len(x), min(512, len(x)), replace=False)]
            total += x.sum(0)
            gram += x.T @ x
            count += len(x)
        if count <= 128:
            raise ValueError('Insufficient visible training points for PCA')
        center = total / count
        values, vectors = np.linalg.eigh((gram - count * np.outer(center, center)) / (count - 1))
        weight = vectors[:, -128:][:, ::-1].T.copy().astype(np.float32)
        torch.save(dict(weight=torch.from_numpy(weight), bias=torch.from_numpy(-(weight @ center).astype(np.float32))), pca_path)
        save(args.root / 'pca.json', dict(training_samples=count, fitting_frames=64, dimensions=128,
             explained_variance=float(values[-128:].sum() / values.sum())))
    pca = torch.load(pca_path, weights_only=True)
    (args.root / 'visual').mkdir(exist_ok=True)
    for row in rows:
        with np.load(raw_dir / (row['frame_id'] + '.npz')) as data:
            valid = data['valid']
            x = data['features'].astype(np.float32)
            seconds = float(data['seconds'])
        image = np.zeros((len(valid), 128), np.float16)
        image[valid] = x @ pca['weight'].numpy().T + pca['bias'].numpy()
        np.savez(args.root / 'visual' / (row['frame_id'] + '.npz'), image=image, valid=valid, seconds=seconds)


def frame(args, row):
    with np.load(SOURCE / 'lidar' / (row['frame_id'] + '.npz')) as data:
        item = {k: torch.tensor(data[k], device='cuda', dtype=torch.float32)
                for k in ['features', 'source', 'prediction', 'target', 'GT', 'center']}
    with np.load(args.root / 'visual' / (row['frame_id'] + '.npz')) as data:
        item['image'] = torch.tensor(data['image'], device='cuda', dtype=torch.float32)
        item['valid'] = torch.tensor(data['valid'], device='cuda')
    return item


def shuffle(image, valid):
    image = image.clone()
    indices = torch.where(valid)[0]
    image[indices] = image[indices.roll(1)]
    return image


def train(args, rows):
    model = decoder()
    loss_fn = source_class(REPO / 'run_mink.py', 'TRR')(scale=10.)
    training = [r for r in rows if r['split'] == 'train']
    schedule = np.random.default_rng(2089).integers(len(training), size=600)
    for arm in ['aligned', 'shuffled']:
        if (args.root / (arm + '.pt')).exists():
            continue
        torch.manual_seed(2089)
        fusion = FeatureFusion().cuda()
        optimizer = torch.optim.Adam(fusion.parameters(), lr=.0001)
        logs = []
        for step, index in enumerate(schedule):
            item = frame(args, training[index])
            indices = torch.randperm(len(item['features']), device='cuda')[:1024]
            lidar, image, valid, target = [item[k][indices] for k in ['features', 'image', 'valid', 'target']]
            if arm == 'shuffled':
                image = shuffle(image, valid)
            optimizer.zero_grad(set_to_none=True)
            pred = model(fusion(lidar, image, valid))
            loss = loss_fn(target, pred[:, :3], pred[:, 3], torch.zeros(len(pred), dtype=torch.long, device='cuda'))[0].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm} step {step}: nonfinite loss')
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == 599:
                logs.append(dict(step=step+1, loss=float(loss.detach())))
                save(args.root / (arm + '_training.json'), logs)
                print(f'{arm} {step+1}/600 loss={loss.item():.5f}', flush=True)
        torch.save(fusion.cpu().state_dict(), args.root / (arm + '.pt'))


def metrics(values):
    x = np.asarray(values)
    return dict(MPE=float(x[:, 0].mean()), MOE=float(x[:, 1].mean()), median=np.median(x, axis=0).tolist(),
                p95=np.percentile(x, 95, axis=0).tolist(), frames=len(x),
                success_1m_2deg=int(((x[:, 0] < 1) & (x[:, 1] < 2)).sum()),
                success_1m_5deg=int(((x[:, 0] < 1) & (x[:, 1] < 5)).sum()))


def evaluate(args, rows):
    from models.sc2pcr import Matcher
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    model = decoder()
    gates = {}
    for arm in ['aligned', 'shuffled']:
        gates[arm] = FeatureFusion().cuda().eval()
        gates[arm].load_state_dict(torch.load(args.root / (arm + '.pt'), weights_only=True))
    records = []
    for row in [r for r in rows if r['split'] == 'val']:
        item = frame(args, row)
        with torch.inference_mode():
            image, valid = item['image'], item['valid']
            predictions = dict(baseline=item['prediction'])
            for name, arm, visual, mask in [('aligned', 'aligned', image, valid),
                    ('shuffled', 'shuffled', shuffle(image, valid), valid),
                    ('aligned_wrong', 'aligned', shuffle(image, valid), valid),
                    ('aligned_missing', 'aligned', image, torch.zeros_like(valid))]:
                predictions[name] = model(gates[arm](item['features'], visual, mask))
            parity = float((predictions['aligned_missing'] - predictions['baseline']).abs().max())
            if parity > 1e-4:
                raise ValueError(f'Cached LEADER/decoder parity failed: {parity}')
            errors = {}
            for arm, pred in predictions.items():
                torch.manual_seed(2089)
                selected = pred[:, 3].topk(max(min(50, len(pred)), int(.5*len(pred)))).indices
                pose = matcher.estimator(item['source'][selected][None], pred[selected, :3][None])[0]
                pose[:3, 3] += item['center']
                trans = float((pose[:3, 3] - item['GT'][:3, 3]).norm())
                cosine = ((pose[:3, :3].T @ item['GT'][:3, :3]).trace() - 1) / 2
                errors[arm] = [trans, float(torch.rad2deg(cosine.clamp(-1, 1).acos()))]
        records.append(dict(frame_id=row['frame_id'], errors=errors, missing_parity=parity,
                            visible_fraction=float(valid.float().mean())))
        save(args.root / 'records.json', records)
        print(f'validation {len(records)}/32 baseline={errors["baseline"]} aligned={errors["aligned"]}', flush=True)
    summary = {arm: metrics([r['errors'][arm] for r in records]) for arm in records[0]['errors']}
    a, b, s = [summary[k] for k in ['aligned', 'baseline', 'shuffled']]
    passed = a['MPE'] <= .95*b['MPE'] and a['MOE'] <= 1.05*b['MOE'] and a['p95'][0] <= 1.05*b['p95'][0]
    passed &= a['success_1m_5deg'] >= b['success_1m_5deg'] and a['MPE'] < s['MPE']
    save(args.root / 'result.json', dict(metrics=summary, passed=bool(passed)))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extract', 'train', 'evaluate', 'all', 'check'])
    parser.add_argument('--root', type=Path, default=Path('/home/zhang/dinov2-leader'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'check':
        import unittest
        from test_fusion import FusionTests
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FusionTests))
        if not result.wasSuccessful():
            raise RuntimeError('Fusion checks failed')
        return
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage in ['prepare', 'all']:
        rows = prepare(args)
    else:
        rows = json.loads((args.root / 'manifest.json').read_text())
    for stage in ['extract', 'train', 'evaluate']:
        if args.stage in [stage, 'all']:
            globals()[stage](args, rows)


if __name__ == '__main__':
    main()
