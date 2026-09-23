import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.train_local905 import batch_loss, digest


def load_model(base, fusion_init):
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=True).cuda()
    missing, unexpected = model.load_state_dict(base['model'], strict=False)
    if unexpected or set(missing) != {
            name for name in model.state_dict() if name.startswith('magic_fusion.')}:
        raise ValueError('Base checkpoint is not a pure LiDAR LEADER state')
    model.magic_fusion.load_state_dict(fusion_init['fusion'])
    return model


def check_identity(model, loader, voxel_size):
    from models.magic_fusion import polar_voxel_centers
    import MinkowskiEngine as ME

    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        sparse = ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda())
        encoded, stages = model.encoder(sparse, return_stages=True)
        stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
        points = polar_voxel_centers(encoded.C, stride, voxel_size, 1024)
        original = model.decoder(encoded.F)
        count = len(batch['T'])
        fused = model.magic_fusion(
            encoded.F, points, encoded.C, stride, batch['sam_features'].cuda(),
            batch['intrinsics'].cuda(), batch['camera_from_lidar'].cuda(),
            torch.eye(4, device='cuda')[None].expand(count, -1, -1),
            batch['image_bounds'].cuda(), stages=stages, voxel_size=voxel_size,
            horizontal=1024, image_valid_mask=batch['image_valid_mask'].cuda())
        difference = (model.decoder(fused) - original).abs().max().item()
    if difference > 1e-5:
        raise ValueError(f'Initial multimodal model differs from LiDAR baseline: {difference}')
    return difference


def evaluate(model, loader, center, loss_fn, voxel_size):
    total = 0.0
    batches = 0.0
    frames = 0
    with torch.no_grad():
        for batch in loader:
            weighted, _ = batch_loss(model, batch, center, loss_fn, True, voxel_size, 1024)
            value = weighted.item()
            total += value * len(batch['T'])
            batches += value
            frames += len(batch['T'])
    return total / frames, batches / len(loader)


def assert_frozen(model, base):
    current = model.state_dict()
    for name, tensor in base['model'].items():
        if not torch.equal(current[name].cpu(), tensor):
            raise ValueError(f'Frozen LEADER state changed: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('A', 'B'), required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--fusion_init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=120)
    parser.add_argument('--min_epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--fusion_lr', type=float, default=0.001)
    parser.add_argument('--leader_lr_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--smoke_only', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    split_hash = digest(args.split)
    sam_hash = digest(args.sam_manifest)
    base_hash = digest(args.base_checkpoint)
    init_hash = digest(args.fusion_init)
    base = torch.load(args.base_checkpoint, map_location='cpu')
    initial = torch.load(args.fusion_init, map_location='cpu')
    source = base['settings']
    if source['magic'] or source['split_sha256'] != split_hash:
        raise ValueError('Base checkpoint has wrong modality or split')
    if initial['base_checkpoint_sha256'] != base_hash or initial['seed'] != args.seed:
        raise ValueError('Fusion initialization differs from frozen protocol')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    flags = SimpleNamespace(
        dataset='Local905', dataset_folder=str(args.data_root.resolve()),
        local905_split=str(args.split.resolve()), local905_max_points=source['max_points'],
        voxel_size=source['voxel_size'], horizontal_res=1024,
        batch_size=args.batch_size, val_batch_size=args.batch_size,
        num_workers=0, mode='train', magic_manifest=str(args.sam_manifest.resolve()),
    )
    train_loader, val_loader = get_data_loader(flags)
    center = torch.tensor(base['center_t'], dtype=torch.float32, device='cuda')
    expected_center = train_loader.dataset.get_center_t() + np.array([0, 0, -200])
    if not np.allclose(center.cpu().numpy(), expected_center, atol=1e-4):
        raise ValueError('Base checkpoint has a different coordinate center')
    model = load_model(base, initial)
    identity_difference = check_identity(model, val_loader, source['voxel_size'])
    if args.variant == 'B':
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        groups = [{'params': model.magic_fusion.parameters(), 'lr': args.fusion_lr,
                   'name': 'fusion'}]
        minimum_lrs = [1e-6]
    else:
        groups = [
            {'params': model.encoder.parameters(),
             'lr': args.fusion_lr * args.leader_lr_ratio, 'name': 'encoder'},
            {'params': model.magic_fusion.parameters(), 'lr': args.fusion_lr,
             'name': 'fusion'},
            {'params': model.decoder.parameters(),
             'lr': args.fusion_lr * args.leader_lr_ratio, 'name': 'decoder'},
        ]
        minimum_lrs = [1e-7, 1e-6, 1e-7]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=minimum_lrs)
    loss_fn = TRR(scale=10)
    if args.smoke_only:
        model.train()
        if args.variant == 'B':
            model.encoder.eval()
            model.decoder.eval()
        optimizer.zero_grad(set_to_none=True)
        weighted, _ = batch_loss(model, next(iter(train_loader)), center, loss_fn,
                                 True, source['voxel_size'], 1024)
        weighted.backward()
        gradient = model.magic_fusion.aggregate.output.weight.grad.norm().item()
        if args.variant == 'B':
            assert_frozen(model, base)
        print(json.dumps({'variant': args.variant, 'identity_max_difference': identity_difference,
                          'loss': weighted.item(), 'fusion_gradient_norm': gradient,
                          'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}),
              flush=True)
        return
    settings = {
        'variant': args.variant, 'magic': True, 'split_sha256': split_hash,
        'sam_manifest_sha256': sam_hash, 'base_checkpoint_sha256': base_hash,
        'fusion_init_sha256': init_hash, 'max_points': source['max_points'],
        'voxel_size': source['voxel_size'], 'batch_size': args.batch_size,
        'seed': args.seed, 'fusion_lr': args.fusion_lr,
        'leader_lr_ratio': args.leader_lr_ratio, 'max_epochs': args.max_epochs,
        'min_epochs': args.min_epochs, 'patience': args.patience,
        'train_count': len(train_loader.dataset), 'val_count': len(val_loader.dataset),
        'center_t': center.tolist(), 'metric_reduction': 'frame',
        'identity_max_difference': identity_difference,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.out / 'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding='utf-8')) != settings:
            raise ValueError('Existing A/B training settings differ')
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    best = float('inf')
    best_epoch = -1
    stale = 0
    start_epoch = 0
    last_path = args.out / 'last.pt'
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best = state['best']
        best_epoch = state['best_epoch']
        stale = state['stale']
        start_epoch = state['epoch'] + 1
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        if args.variant == 'B':
            model.encoder.eval()
            model.decoder.eval()
        train_total = 0.0
        train_frames = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            weighted, _ = batch_loss(model, batch, center, loss_fn, True,
                                     source['voxel_size'], 1024)
            if not torch.isfinite(weighted):
                raise FloatingPointError(f'Nonfinite train loss at epoch {epoch}')
            weighted.backward()
            optimizer.step()
            train_total += weighted.item() * len(batch['T'])
            train_frames += len(batch['T'])
        model.eval()
        val_trr, val_batch_trr = evaluate(
            model, val_loader, center, loss_fn, source['voxel_size'])
        improved = val_trr < best * 0.998
        if improved:
            best = val_trr
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        scheduler.step(val_trr)
        if args.variant == 'B':
            assert_frozen(model, base)
        record = {
            'epoch': epoch, 'train_trr_frame_weighted': train_total / train_frames,
            'val_trr_frame_weighted': val_trr, 'val_trr_batch_mean': val_batch_trr,
            'best_epoch': best_epoch, 'stale': stale,
            'lrs': {group['name']: group['lr'] for group in optimizer.param_groups},
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
        }
        with (args.out / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best,
            'best_epoch': best_epoch, 'stale': stale, 'settings': settings,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        }
        temporary = args.out / 'last.tmp'
        torch.save(state, temporary)
        os.replace(temporary, last_path)
        if improved:
            temporary = args.out / 'best.tmp'
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_trr': val_trr, 'center_t': center.tolist(),
                        'settings': settings}, temporary)
            os.replace(temporary, args.out / 'best.pt')
        if epoch + 1 >= args.min_epochs and stale >= args.patience:
            (args.out / 'converged.json').write_text(json.dumps({
                'criterion': 'frame-weighted validation TRR improves <0.2% for 8 epochs',
                'stopped_epoch': epoch, 'best_epoch': best_epoch,
                'best_val_trr': best,
            }, indent=2) + '\n', encoding='utf-8')
            break


if __name__ == '__main__':
    main()
