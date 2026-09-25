import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.train_local905 import batch_loss, digest
from tools.train_magic_revision import sample_batches, write_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.variant = 'spatial'
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    np.random.seed(37)
    torch.backends.cudnn.deterministic = True
    split = args.assets / 'split_masked.json'
    sam = args.assets / 'sam_cache/manifest_905.json'
    base_path = args.assets / 'official_l0.pt'
    init_path = args.assets / 'fusion_init_official_s37.pt'
    base = torch.load(base_path, map_location='cpu')
    init = torch.load(init_path, map_location='cpu')
    if (base['settings']['max_points'] != 0 or base['settings']['magic'] or
            base['settings']['split_sha256'] != digest(split) or
            init['base_checkpoint_sha256'] != digest(base_path) or init['seed'] != 37):
        raise ValueError('Wrong full-point base, split, or common initialization')
    model = LEADER(in_channels=3, out_channels=4, magic=True,
                   fusion_variant=args.variant).cuda()
    missing, extra = model.load_state_dict(base['model'], strict=False)
    if extra or any(not key.startswith('magic_fusion.') for key in missing):
        raise ValueError('Unexpected LEADER state mismatch')
    common = {key: value for key, value in init['fusion'].items()
              if not key.startswith('aggregate.')}
    missing, extra = model.magic_fusion.load_state_dict(common, strict=False)
    expected = {key for key in model.magic_fusion.state_dict() if key.startswith('aggregate.')
                and not key.endswith('num_batches_tracked')}
    missing = [key for key in missing if not key.endswith('num_batches_tracked')]
    if extra or set(missing) != expected:
        raise ValueError('Unexpected fusion state mismatch')
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.eval()
    model.magic_fusion.train()
    center = torch.tensor(base['center_t'], device='cuda')
    flags = SimpleNamespace(dataset='Local905', dataset_folder=str(args.data_root),
                            local905_split=str(split), local905_max_points=0,
                            voxel_size=0.2, horizontal_res=1024, batch_size=2,
                            val_batch_size=1, num_workers=0, mode='train', magic_manifest=str(sam))
    loader, _ = get_data_loader(flags)
    if len(loader.dataset) != 552:
        raise ValueError('Wrong training denominator')
    order, order_hash = sample_batches(552, 8, 138, 100037)
    settings = dict(base['settings'])
    settings.update(protocol='spatial_decoder_trial_v1', variant='spatial', magic=True,
                    fusion_variant=args.variant, spatial_hidden=128,
                    sam_manifest_sha256=digest(sam), base_checkpoint_sha256=digest(base_path),
                    fusion_init_sha256=digest(init_path), seed=37, data_seed=100037,
                    sample_order_sha256=order_hash, effective_batch=8, microbatch=2,
                    optimizer_steps=138, fusion_lr=1e-4, selection='fixed_final_step',
                    leader_bn_running_stats_frozen=True, fusion_bn_running_stats_frozen=False,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    source_sha256={name: digest(Path(name)) for name in (
                        'models/model_mink.py', 'models/magic_fusion.py',
                        'models/spatial_magic_fusion.py', 'tools/train_spatial_trial.py',
                        'tools/train_local905.py', 'tools/eval_local905_online.py',
                        'data/local905_query.py', 'experiments/spatial_decoder/PLAN.md')})
    settings.pop('bn_running_stats_frozen', None)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    optimizer = torch.optim.Adam(model.magic_fusion.parameters(), lr=1e-4)
    loss_fn = TRR(scale=10)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = 1 if args.smoke else 138
    initial_differences = []
    gradient_evidence = {}
    def record_initial(module, inputs, output):
        initial_differences.append(float((output - inputs[0]).abs().max()))
    hook = model.magic_fusion.register_forward_hook(record_initial) if args.smoke else None
    with (args.out / 'history.jsonl').open('w') as history:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            step_start = time.perf_counter()
            for micro in range(4):
                sample = [loader.dataset[int(i)] for i in order[step, micro * 2:micro * 2 + 2]]
                batch = loader.collate_fn(sample)
                loss, _ = batch_loss(model, batch, center, loss_fn, True, 0.2, 1024)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}, micro {micro}')
                (loss / 4).backward()
                loss_sum += float(loss.detach()) / 4
                del loss, batch, sample, _
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.magic_fusion.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            if step in (0, 1, steps - 1):
                gradient_evidence[str(step + 1)] = {
                    prefix: sum(float(p.grad.detach().abs().sum())
                                for name, p in model.magic_fusion.named_parameters()
                                if name.startswith(prefix) and p.grad is not None)
                    for prefix in ('image_encoder', 'attention', 'aggregate.native',
                                   'aggregate.decode', 'aggregate.output')}
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(step=step + 1, trr=loss_sum, seconds=time.perf_counter() - step_start,
                       elapsed_seconds=time.perf_counter() - started,
                       peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
            history.write(json.dumps(row) + '\n')
            history.flush()
            print(json.dumps(row), flush=True)
    current = model.state_dict()
    if hook is not None:
        hook.remove()
        if len(initial_differences) != 4 or max(initial_differences) != 0:
            raise RuntimeError('Real-batch zero-residual initialization differs from LEADER')
    unchanged = all(torch.equal(current[name].cpu(), value) for name, value in base['model'].items())
    if not unchanged:
        raise RuntimeError('Frozen LEADER parameters or buffers changed')
    report = dict(variant=args.variant, completed_steps=steps, frames=steps * 8,
                  frozen_leader_exactly_unchanged=unchanged, smoke=args.smoke,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
    report['gradient_l1'] = gradient_evidence
    if args.smoke:
        report['initial_feature_max_differences'] = initial_differences
    if not args.smoke:
        model.cpu()
        write_torch(args.out / 'final.pt', {'model': model.state_dict(),
                    'center_t': center.tolist(), 'settings': settings, 'optimizer_step': steps,
                    'optimizer': optimizer.state_dict()})
        report['checkpoint_sha256'] = digest(args.out / 'final.pt')
    (args.out / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
