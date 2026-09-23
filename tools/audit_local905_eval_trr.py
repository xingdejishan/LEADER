import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from data.local905_mink import Local905_mink
from data.magic_data import CalibratedImageDataset
from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.train_local905 import batch_loss, digest


def evaluate(loader, model, center, magic, voxel_size):
    batch_total = 0.0
    frame_total = 0.0
    frames = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            weighted, _ = batch_loss(model, batch, center, TRR(scale=10),
                                     magic, voxel_size, 1024)
            value = weighted.item()
            count = len(batch['T'])
            batch_total += value
            frame_total += value * count
            frames += count
            batches += 1
    return {'frames': frames, 'batches': batches,
            'trr_batch_mean': batch_total / batches,
            'trr_frame_weighted': frame_total / frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    settings = checkpoint['settings']
    magic = settings['magic']
    if magic != (args.sam_manifest is not None):
        raise ValueError('Checkpoint modality and SAM manifest differ')
    if digest(args.split) != settings['split_sha256']:
        raise ValueError('Split differs from checkpoint')
    if magic and digest(args.sam_manifest) != settings['sam_manifest_sha256']:
        raise ValueError('SAM manifest differs from checkpoint')
    split = json.loads(args.split.read_text(encoding='utf-8'))
    flags = SimpleNamespace(
        dataset='Local905', dataset_folder=str(args.data_root.resolve()),
        local905_split=str(args.split.resolve()),
        local905_max_points=settings['max_points'],
        voxel_size=settings['voxel_size'], horizontal_res=1024,
        batch_size=settings['batch_size'], val_batch_size=settings['batch_size'],
        num_workers=0, mode='train',
        magic_manifest=str(args.sam_manifest.resolve()) if magic else '',
    )
    torch.manual_seed(settings['seed'])
    train_loader, val_loader = get_data_loader(flags)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=magic).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    center = torch.tensor(checkpoint['center_t'], dtype=torch.float32, device='cuda')
    result = {
        'protocol': 'local905_same_eval_train_val_v1',
        'checkpoint_sha256': digest(args.checkpoint),
        'split_sha256': digest(args.split),
        'checkpoint_epoch': checkpoint['epoch'],
        'train': evaluate(train_loader, model, center, magic, settings['voxel_size']),
        'val': evaluate(val_loader, model, center, magic, settings['voxel_size']),
    }
    if 'same_domain_val' in split['splits']:
        same = Local905_mink(args.data_root, args.split, 'same_domain_val',
                             voxel_size=settings['voxel_size'],
                             max_points=settings['max_points'])
        if magic:
            same = CalibratedImageDataset(same, args.data_root, args.sam_manifest,
                                          same.valid_mask_sha256)
        same_loader = DataLoader(same, batch_size=settings['batch_size'],
                                 shuffle=False, num_workers=0,
                                 collate_fn=val_loader.collate_fn)
        result['same_domain_val'] = evaluate(
            same_loader, model, center, magic, settings['voxel_size'])
        if result['same_domain_val']['frames'] != len(split['splits']['same_domain_val']):
            raise ValueError('Incomplete same-domain denominator')
    if result['train']['frames'] != settings['train_count'] or result['val']['frames'] != settings['val_count']:
        raise ValueError('Incomplete train or validation denominator')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
