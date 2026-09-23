import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import MinkowskiEngine as ME
import numpy as np
import torch

from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER
from run_mink import TRR, get_data_loader


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def batch_loss(model, batch, center, loss_fn, magic, voxel_size, horizontal_res):
    sparse = ME.SparseTensor(batch['feats'].cuda(non_blocking=True),
                             batch['coords'].cuda(non_blocking=True))
    if magic:
        encoded, stages = model.encoder(sparse, return_stages=True)
    else:
        encoded = model.encoder(sparse)
    stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
    points = polar_voxel_centers(encoded.C, stride, voxel_size, horizontal_res)
    batch_index = encoded.C[:, 0].long()
    poses = batch['T'].cuda(non_blocking=True)
    targets = torch.bmm(poses[batch_index, :3, :3], points.unsqueeze(-1)).squeeze(-1)
    targets = targets + poses[batch_index, :3, 3] - center
    features = encoded.F
    if magic:
        sam = batch['sam_features'].cuda(non_blocking=True)
        intrinsics = batch['intrinsics'].cuda(non_blocking=True)
        extrinsics = batch['camera_from_lidar'].cuda(non_blocking=True)
        bounds = batch['image_bounds'].cuda(non_blocking=True)
        recovery = torch.eye(4, device='cuda')[None].expand(poses.shape[0], -1, -1)
        features = model.magic_fusion(features, points, encoded.C, stride, sam,
                                       intrinsics, extrinsics, recovery, bounds,
                                       stages=stages, voxel_size=voxel_size,
                                       horizontal=horizontal_res,
                                       image_valid_mask=batch['image_valid_mask'].cuda(non_blocking=True)
                                       if 'image_valid_mask' in batch else None)
    predictions = model.decoder(features)
    weighted, raw = loss_fn(targets, predictions[:, :3], predictions[:, 3], batch_index)
    return weighted.mean(), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--magic', action='store_true')
    parser.add_argument('--max_points', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--voxel_size', type=float, default=0.2)
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--min_epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--smoke_only', action='store_true')
    args = parser.parse_args()
    if args.magic and args.sam_manifest is None:
        raise ValueError('--sam_manifest is required with --magic')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    flags = SimpleNamespace(
        dataset='Local905', dataset_folder=str(args.data_root.resolve()),
        local905_split=str(args.split.resolve()), local905_max_points=args.max_points,
        voxel_size=args.voxel_size, horizontal_res=1024, batch_size=args.batch_size,
        val_batch_size=args.batch_size, num_workers=0, mode='train',
        magic_manifest=str(args.sam_manifest.resolve()) if args.magic else '',
    )
    train_loader, val_loader = get_data_loader(flags)
    center = train_loader.dataset.get_center_t() + np.array([0, 0, -200], dtype=np.float32)
    center = torch.tensor(center, dtype=torch.float32, device='cuda')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=args.magic).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = TRR(scale=10)
    if args.smoke_only:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        weighted, raw = batch_loss(model, next(iter(train_loader)), center, loss_fn,
                                   args.magic, args.voxel_size, 1024)
        weighted.backward()
        torch.cuda.synchronize()
        print(json.dumps({
            'mode': 'magic' if args.magic else 'lidar',
            'loss': weighted.item(), 'raw_loss': raw.item(),
            'elapsed_s': time.perf_counter() - start,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'fusion_grad': (float(model.magic_fusion.aggregate.output.weight.grad.norm().item())
                            if args.magic else None),
        }), flush=True)
        return
    split_hash = digest(args.split)
    sam_hash = digest(args.sam_manifest) if args.magic else None
    settings = {
        'magic': args.magic, 'split_sha256': split_hash, 'sam_manifest_sha256': sam_hash,
        'max_points': args.max_points, 'voxel_size': args.voxel_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'seed': args.seed,
        'max_epochs': args.max_epochs, 'min_epochs': args.min_epochs,
        'patience': args.patience, 'train_count': len(train_loader.dataset),
        'val_count': len(val_loader.dataset), 'center_t': center.tolist(),
    }
    settings_path = args.out / 'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding='utf-8')) != settings:
            raise ValueError('Existing training settings differ')
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
    history = args.out / 'history.jsonl'
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        train_loss = 0.0
        train_l2 = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            weighted, raw = batch_loss(model, batch, center, loss_fn, args.magic,
                                       args.voxel_size, 1024)
            if not torch.isfinite(weighted):
                raise FloatingPointError(f'Nonfinite train loss at epoch {epoch}')
            weighted.backward()
            optimizer.step()
            train_loss += weighted.item()
            train_l2 += raw.item()
        model.eval()
        val_loss = 0.0
        val_l2 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                weighted, raw = batch_loss(model, batch, center, loss_fn, args.magic,
                                           args.voxel_size, 1024)
                val_loss += weighted.item()
                val_l2 += raw.item()
        train_loss /= len(train_loader)
        train_l2 /= len(train_loader)
        val_loss /= len(val_loader)
        val_l2 /= len(val_loader)
        improved = val_loss < best * 0.998
        if improved:
            best = val_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        scheduler.step(val_loss)
        record = {
            'epoch': epoch, 'train_trr': train_loss, 'train_l2': train_l2,
            'val_trr': val_loss, 'val_l2': val_l2,
            'best_epoch': best_epoch, 'stale': stale,
            'lr': optimizer.param_groups[0]['lr'],
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
        with history.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best,
            'best_epoch': best_epoch, 'stale': stale,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'center_t': center.tolist(), 'settings': settings,
        }
        temporary = args.out / 'last.tmp'
        torch.save(state, temporary)
        os.replace(temporary, last_path)
        if improved:
            best_path = args.out / 'best.pt'
            temporary = args.out / 'best.tmp'
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_trr': val_loss, 'center_t': center.tolist(),
                        'settings': settings}, temporary)
            os.replace(temporary, best_path)
        if epoch + 1 >= args.min_epochs and stale >= args.patience:
            (args.out / 'converged.json').write_text(json.dumps({
                'criterion': 'validation TRR improves <0.2% for 8 epochs after minimum 12',
                'stopped_epoch': epoch, 'best_epoch': best_epoch, 'best_val_trr': best,
            }, indent=2) + '\n', encoding='utf-8')
            break


if __name__ == '__main__':
    main()
