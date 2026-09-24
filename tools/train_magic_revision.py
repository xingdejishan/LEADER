import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.train_local905 import batch_loss, digest


def camera_key(extrinsic):
    value = extrinsic.to(dtype=torch.float64).cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()[:16]


def sample_batches(frame_count, batch_size, total_steps, seed):
    random = np.random.default_rng(seed)
    order = []
    while len(order) < batch_size * total_steps:
        order.extend(random.permutation(frame_count).tolist())
    values = np.asarray(order[:batch_size * total_steps], dtype=np.int32)
    return values.reshape(total_steps, batch_size), hashlib.sha256(values.tobytes()).hexdigest()


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def set_mode(model, stage):
    model.train()
    if stage in ('warmup', 'B'):
        model.encoder.eval()
        model.decoder.eval()
    freeze_bn_stats(model)


def apply_null(batch, templates):
    if templates is None:
        return batch
    batch['sam_features'] = torch.cat([
        templates[camera_key(extrinsic)]
        for extrinsic in batch['camera_from_lidar']
    ], dim=0)
    return batch


def checkpoint(model, center, settings, step):
    return {'model': model.state_dict(), 'center_t': center.tolist(),
            'settings': settings, 'optimizer_step': step}


def write_torch(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_pose(model, center, settings, step, args, out):
    candidate = out / 'candidate.pt'
    write_torch(candidate, checkpoint(model, center, settings, step))
    evaluation_dir = out / f'validation_step_{step:05d}'
    online = [sys.executable, '-m', 'tools.eval_local905_online',
              '--data_root', str(args.data_root), '--split', str(args.split),
              '--checkpoint', str(candidate), '--subset', 'val',
              '--out', str(evaluation_dir)]
    if settings['magic']:
        online += ['--sam_manifest', str(args.sam_manifest)]
    if settings.get('null_template_sha256'):
        online += ['--null_template', str(args.null_template)]
    with (evaluation_dir.parent / f'validation_step_{step:05d}.log').open('w') as stream:
        subprocess.run(online, check=True, stdout=stream)
    subprocess.run([sys.executable, '-m', 'tools.eval_local905_gt',
                    '--data_root', str(args.data_root), '--split', str(args.split),
                    '--predictions', str(evaluation_dir / 'predictions.json'),
                    '--out', str(evaluation_dir / 'evaluation.json')],
                   check=True, stdout=subprocess.DEVNULL)
    result = json.loads((evaluation_dir / 'evaluation.json').read_text(encoding='utf-8'))
    if result['successful_frames'] != result['frames']:
        score = float('inf')
    else:
        score = 0.5 * (
            result['all_frame_mpe_mean_m'] / args.l0_mpe +
            result['all_frame_moe_mean_deg'] / args.l0_moe)
    return candidate, result, score


def save_state(out, model, optimizer, center, settings, step, best_j, best_step):
    state = checkpoint(model, center, settings, step)
    state.update({'optimizer': optimizer.state_dict(), 'best_j': best_j,
                  'best_step': best_step, 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state()})
    write_torch(out / 'last.pt', state)


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('warmup', 'A', 'B', 'LFT'), required=True)
    parser.add_argument('--null', action='store_true')
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path, required=True)
    parser.add_argument('--null_template', type=Path)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--fusion_init', type=Path)
    parser.add_argument('--warmup_checkpoint', type=Path)
    parser.add_argument('--l0_val_report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fusion_lr', type=float, required=True)
    parser.add_argument('--total_steps', type=int, default=690)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_interval', type=int, default=69)
    args = parser.parse_args()
    if args.stage == 'LFT' and args.null:
        raise ValueError('LFT has no image input')
    if args.stage != 'LFT' and args.fusion_init is None:
        raise ValueError('Fusion conditions require the paired initialization')
    if args.null != bool(args.null_template) and args.stage != 'LFT':
        raise ValueError('Null condition requires exactly one fixed template')
    if args.stage in ('A', 'B') and args.warmup_checkpoint is None:
        raise ValueError('A/B must start from their shared warmup checkpoint')
    if args.total_steps < 10 or args.batch_size < 1 or args.eval_interval < 1:
        raise ValueError('Invalid optimizer-step budget')
    if args.fusion_lr not in (1e-4, 3e-4, 1e-3):
        raise ValueError('Fusion LR must come from the locked grid')
    warmup_steps = args.total_steps // 10
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = torch.load(args.base_checkpoint, map_location='cpu')
    if base['settings']['magic'] or base['settings']['max_points'] != 0:
        raise ValueError('Base must be original full-point LiDAR LEADER')
    if base['settings']['split_sha256'] != digest(args.split):
        raise ValueError('Training split differs from base')
    original_val = json.loads(args.l0_val_report.read_text(encoding='utf-8'))
    if original_val['subset'] != 'val' or original_val['successful_frames'] != 40:
        raise ValueError('Invalid L0 validation denominator')
    args.l0_mpe = original_val['all_frame_mpe_mean_m']
    args.l0_moe = original_val['all_frame_moe_mean_deg']
    if args.l0_mpe <= 0 or args.l0_moe <= 0:
        raise ValueError('Invalid L0 pose normalization')
    magic = args.stage != 'LFT'
    init = torch.load(args.fusion_init, map_location='cpu') if magic else None
    if magic and (init['base_checkpoint_sha256'] != digest(args.base_checkpoint)
                  or init['seed'] != args.seed):
        raise ValueError('Paired fusion initialization mismatch')
    with np.load(args.null_template, allow_pickle=False) if args.null else _empty_archive() as archive:
        templates = ({name: torch.from_numpy(archive[name].copy())[None]
                      for name in archive.files} if args.null else None)
    flags = SimpleNamespace(
        dataset='Local905', dataset_folder=str(args.data_root),
        local905_split=str(args.split), local905_max_points=0,
        voxel_size=0.2, horizontal_res=1024, batch_size=args.batch_size,
        val_batch_size=args.batch_size, num_workers=0, mode='train',
        magic_manifest=str(args.sam_manifest) if magic else '')
    train_loader, _ = get_data_loader(flags)
    frame_count = len(train_loader.dataset)
    order, order_hash = sample_batches(frame_count, args.batch_size,
                                       args.total_steps, 100000 + args.seed)
    settings = {
        'protocol': 'magic_revision_six_condition_v1',
        'variant': args.stage + ('-null' if args.null else ''),
        'magic': magic, 'split_sha256': digest(args.split),
        'sam_manifest_sha256': digest(args.sam_manifest) if magic else None,
        'null_template_sha256': digest(args.null_template) if args.null else None,
        'base_checkpoint_sha256': digest(args.base_checkpoint),
        'fusion_init_sha256': digest(args.fusion_init) if magic else None,
        'model_code_sha256': digest(Path('models/model_mink.py')),
        'fusion_code_sha256': digest(Path('models/magic_fusion.py')),
        'max_points': 0, 'voxel_size': 0.2, 'batch_size': args.batch_size,
        'seed': args.seed, 'data_seed': 100000 + args.seed,
        'sample_order_sha256': order_hash, 'fusion_lr': args.fusion_lr,
        'leader_lr': args.fusion_lr * 0.1, 'total_steps': args.total_steps,
        'warmup_steps': warmup_steps, 'eval_interval': args.eval_interval,
        'l0_val_mpe_m': args.l0_mpe, 'l0_val_moe_deg': args.l0_moe,
        'bn_running_stats_frozen': True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.out / 'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding='utf-8')) != settings:
            raise ValueError('Resume settings differ')
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=magic).cuda()
    model.load_state_dict(base['model'], strict=not magic)
    if magic:
        model.magic_fusion.load_state_dict(init['fusion'])
    center = torch.tensor(base['center_t'], dtype=torch.float32, device='cuda')
    if args.stage in ('warmup', 'B'):
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
    if args.stage == 'LFT':
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters(), 'lr': args.fusion_lr * 0.1},
            {'params': model.decoder.parameters(), 'lr': args.fusion_lr * 0.1}])
    else:
        optimizer = torch.optim.Adam([{'params': model.magic_fusion.parameters(),
                                       'lr': args.fusion_lr}])
    if args.stage in ('A', 'B'):
        warm = torch.load(args.warmup_checkpoint, map_location='cpu')
        prior = warm['settings']
        if (prior['variant'] != 'warmup' + ('-null' if args.null else '')
                or prior['fusion_lr'] != args.fusion_lr
                or prior['seed'] != args.seed
                or prior['sample_order_sha256'] != order_hash
                or prior['split_sha256'] != settings['split_sha256']
                or prior['sam_manifest_sha256'] != settings['sam_manifest_sha256']
                or prior['null_template_sha256'] != settings['null_template_sha256']
                or prior['model_code_sha256'] != settings['model_code_sha256']
                or prior['fusion_code_sha256'] != settings['fusion_code_sha256']):
            raise ValueError('Wrong warmup pairing')
        if warm['optimizer_step'] != warmup_steps:
            raise ValueError('Warmup has wrong number of optimizer steps')
        model.load_state_dict(warm['model'])
        optimizer.load_state_dict(warm['optimizer'])
        torch.set_rng_state(warm['torch_rng'])
        torch.cuda.set_rng_state(warm['cuda_rng'])
        if args.stage == 'A':
            model.encoder.requires_grad_(True)
            model.decoder.requires_grad_(True)
            optimizer.add_param_group({'params': model.encoder.parameters(),
                                       'lr': args.fusion_lr * 0.1})
            optimizer.add_param_group({'params': model.decoder.parameters(),
                                       'lr': args.fusion_lr * 0.1})
        if not (args.out / 'best.pt').exists():
            warm_best = torch.load(args.warmup_checkpoint.parent / 'best.pt', map_location='cpu')
            write_torch(args.out / 'best.pt', {
                'model': warm_best['model'], 'center_t': warm_best['center_t'],
                'settings': settings, 'optimizer_step': warm_best['optimizer_step']})
        best_j, best_step = warm['best_j'], warm['best_step']
        start_step = warmup_steps
    else:
        best_j, best_step = 1.0, 0
        start_step = 0 if args.stage == 'warmup' else warmup_steps
        if not (args.out / 'best.pt').exists():
            write_torch(args.out / 'best.pt', checkpoint(model, center, settings, 0))
    last = args.out / 'last.pt'
    if last.exists() and args.stage != 'warmup':
        saved = torch.load(last, map_location='cpu')
        if saved['settings'] != settings:
            raise ValueError('Resume checkpoint settings differ')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best_j, best_step = saved['best_j'], saved['best_step']
        start_step = saved['optimizer_step']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
    elif last.exists() and args.stage == 'warmup':
        saved = torch.load(last, map_location='cpu')
        if saved['settings'] != settings:
            raise ValueError('Resume checkpoint settings differ')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best_j, best_step = saved['best_j'], saved['best_step']
        start_step = saved['optimizer_step']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
    end_step = warmup_steps if args.stage == 'warmup' else args.total_steps
    original_state = {name: value.detach().cpu().clone() for name, value in
                      base['model'].items()} if args.stage in ('warmup', 'B') else None
    for step in range(start_step, end_step):
        set_mode(model, args.stage)
        sample = [train_loader.dataset[int(index)] for index in order[step]]
        batch = apply_null(train_loader.collate_fn(sample), templates)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, batch, center, TRR(scale=10), magic, 0.2, 1024)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite TRR at optimizer step {step}')
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach())
        del batch, loss, _
        completed = step + 1
        if completed % args.eval_interval == 0 or completed == end_step:
            model.eval()
            model.cpu()
            move_optimizer_state(optimizer, 'cpu')
            torch.cuda.empty_cache()
            try:
                candidate, validation, score = evaluate_pose(
                    model, center, settings, completed, args, args.out)
            finally:
                model.cuda()
                move_optimizer_state(optimizer, 'cuda')
            if score < best_j:
                os.replace(candidate, args.out / 'best.pt')
                best_j, best_step = score, completed
            else:
                candidate.unlink()
            if original_state is not None:
                for name, old in original_state.items():
                    if not torch.equal(model.state_dict()[name].cpu(), old):
                        raise ValueError(f'Frozen LEADER changed: {name}')
            record = {'step': completed, 'train_trr_last_batch': train_loss,
                      'val_mpe_m': validation['all_frame_mpe_mean_m'],
                      'val_moe_deg': validation['all_frame_moe_mean_deg'],
                      'val_failures': validation['failed_frames'],
                      'val_j': score, 'best_j': best_j, 'best_step': best_step,
                      'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
            with (args.out / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            save_state(args.out, model, optimizer, center, settings,
                       completed, best_j, best_step)
            print(json.dumps(record), flush=True)
    (args.out / 'complete.json').write_text(json.dumps({
        'stage': args.stage, 'null': args.null, 'seed': args.seed,
        'fusion_lr': args.fusion_lr, 'optimizer_steps': end_step,
        'best_step': best_step, 'best_j': best_j,
    }, indent=2) + '\n', encoding='utf-8')


class _empty_archive:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == '__main__':
    main()
