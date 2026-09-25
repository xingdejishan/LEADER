import argparse
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.compare_local905 import compare
from tools.run_surface_trial import run
from tools.train_local905 import batch_loss, digest
from tools.train_magic_revision import freeze_bn_stats, write_torch
from tools.train_gravity_fusion import objective


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def setup(args):
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    np.random.seed(37)
    torch.backends.cudnn.deterministic = True
    base = torch.load(args.assets/'official_l0.pt', map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, magic=True, fusion_variant='interaction').cuda()
    missing, unexpected = model.load_state_dict(base['model'], strict=False)
    if unexpected or any(not name.startswith('interaction.') for name in missing):
        raise ValueError('Pretrained LEADER not fully loaded')
    if not all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in base['model'].items()):
        raise ValueError('Pretrained parameters differ')
    flags = SimpleNamespace(dataset='Local905', dataset_folder=str(args.data_root),
        local905_split=str(args.assets/'split_masked.json'), local905_max_points=0,
        voxel_size=.2, horizontal_res=1024, batch_size=1, val_batch_size=1,
        num_workers=0, mode='train', magic_manifest=str(args.assets/'sam_cache/manifest_905.json'))
    loader, _ = get_data_loader(flags)
    return model, base, loader


def evaluate(checkpoint, folder, subset, args):
    run(['tools.eval_local905_online', '--data_root', args.data_root,
         '--split', args.assets/'split_masked.json', '--sam_manifest', args.assets/'sam_cache/manifest_905.json',
         '--checkpoint', checkpoint, '--subset', subset, '--out', folder], folder.parent/(folder.name+'_online.log'))
    run(['tools.eval_local905_gt', '--data_root', args.data_root, '--split', args.assets/'split_masked.json',
         '--predictions', folder/'predictions.json', '--out', folder/'evaluation.json'], folder.parent/(folder.name+'_gt.log'))
    result = read(folder/'evaluation.json')
    if result['predictions_sha256'] != digest(folder/'predictions.json'):
        raise ValueError('Frozen prediction mismatch')
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('data_root', 'assets', 'previous', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source_commit', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    model, base, loader = setup(args)
    if len(loader.dataset) != 552:
        raise ValueError('Training split changed')
    center = torch.tensor(base['center_t'], device='cuda')
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder.parameters())+list(model.decoder.parameters()), 'lr': 1e-5},
        {'params': model.interaction.parameters(), 'lr': 1e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=.001, min_lr=1e-7)
    settings = dict(base['settings'], magic=True, fusion_variant='interaction',
                    sam_manifest_sha256=digest(args.assets/'sam_cache/manifest_905.json'),
                    base_checkpoint_sha256=digest(args.assets/'official_l0.pt'))
    sources = ('models/model_mink.py', 'models/interaction_backbone.py', 'models/surface_token_fusion.py',
        'models/magic_fusion.py', 'tools/train_interaction.py', 'tools/train_local905.py',
        'tools/eval_local905_online.py', 'tools/eval_local905_gt.py', 'tools/verify_interaction.py',
        'data/local905_mink.py', 'data/local905_query.py', 'data/magic_data.py',
        'experiments/interaction/PLAN.md', 'experiments/interaction/audit.json')
    protocol = dict(source_commit=args.source_commit, source_sha256={name:digest(Path(name)) for name in sources},
        settings=settings, min_epochs=40, validation_every=5, patience_epochs=30, required_lr_reductions=2,
        optimizer='AdamW', backbone_lr=1e-5, new_lr=1e-4, effective_batch=8, microbatch=1,
        seed=37, bn_statistics='frozen', selection='40 only; eligible min ratio, otherwise mean ratios',
        test_policy='313 one selected checkpoint, development only')
    epoch, best_epoch, best_value, significant, last_significant, reductions = 0, 0, math.inf, math.inf, 0, 0
    history, records, gradients = [], [], {}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if read(args.out/'protocol.json') != protocol:
            raise ValueError('Resume source or protocol differs')
        state = torch.load(args.out/'latest.pt', map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for value in optimizer.state.values():
            for key, item in value.items():
                if torch.is_tensor(item):
                    value[key] = item.cuda()
        scheduler.load_state_dict(state['scheduler'])
        epoch, best_epoch, best_value, significant, last_significant, reductions = state['progress']
        history, records, gradients = state['history'], state['records'], state['gradients']
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    else:
        write(args.out/'protocol.json', protocol)
        with zipfile.ZipFile(args.out/'executed_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(name, name)
        for subset in ('val', 'test'):
            shutil.copytree(args.previous/f'L0_{subset}', args.out/f'L0_{subset}')
    baseline = {subset:read(args.out/f'L0_{subset}/evaluation.json') for subset in ('val', 'test')}
    for subset in baseline:
        if baseline[subset]['predictions_sha256'] != digest(args.out/f'L0_{subset}/predictions.json'):
            raise ValueError('Baseline fingerprint mismatch')
    loss_fn = TRR(scale=10)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def save(path):
        write_torch(path, dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            center_t=base['center_t'], settings=settings, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            progress=(epoch,best_epoch,best_value,significant,last_significant,reductions),
            history=history, records=records, gradients=gradients, numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

    with (args.out/'history.jsonl').open('a') as stream:
        while True:
            epoch += 1
            tick = time.perf_counter()
            model.train()
            freeze_bn_stats(model)
            order = np.random.permutation(552)
            total = 0.
            for offset in range(0, 552, 8):
                optimizer.zero_grad(set_to_none=True)
                for index in order[offset:offset+8]:
                    batch = loader.collate_fn([loader.dataset[int(index)]])
                    loss, _ = batch_loss(model, batch, center, loss_fn, True, .2, 1024)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss/8).backward()
                    total += float(loss.detach())
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite training gradient')
                if epoch == 1 and offset == 8:
                    gradients = {name:float(p.grad.abs().sum()) for name,p in model.named_parameters()
                                 if p.grad is not None and (name.startswith('interaction.') or name in
                                 ('encoder.stem.0.0.linear.weight', 'decoder.head.0.weight'))}
                optimizer.step()
                if offset % 160 == 0:
                    print(json.dumps(dict(epoch=epoch, frames=offset+8, mean_loss=total/(offset+8))), flush=True)
            torch.cuda.synchronize()
            row = dict(epoch=epoch, loss=total/552, seconds=time.perf_counter()-tick,
                       lr=[group['lr'] for group in optimizer.param_groups], peak_mib=torch.cuda.max_memory_allocated()/1024**2)
            history.append(row)
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            print(json.dumps(row), flush=True)
            if epoch % 5:
                save(args.out/'latest.pt')
                continue
            candidate = args.out/'candidate.pt'
            save(candidate)
            result = evaluate(candidate, args.out/f'val_{epoch:04d}', 'val', args)
            value, eligible = objective(result, baseline['val'])
            old_lr = optimizer.param_groups[1]['lr']
            scheduler.step(value)
            reductions += int(optimizer.param_groups[1]['lr'] < old_lr)
            if value < best_value:
                best_epoch, best_value = epoch, value
                shutil.copy2(candidate, args.out/'best.pt')
            if value < significant*.999:
                significant, last_significant = value, epoch
            records.append(dict(epoch=epoch, objective=value, eligible=eligible, best_epoch=best_epoch,
                mpe=result['all_frame_mpe_mean_m'], moe=result['all_frame_moe_mean_deg'],
                checkpoint_sha256=digest(candidate), lr_reductions=reductions))
            write(args.out/'validation_selection.json', dict(records=records,best_epoch=best_epoch))
            print(json.dumps(records[-1]), flush=True)
            save(args.out/'latest.pt')
            if epoch >= 40 and epoch-last_significant >= 30 and reductions >= 2:
                break
    write(args.out/'selection_frozen.json', dict(epoch=best_epoch, checkpoint_sha256=digest(args.out/'best.pt'), before_313_evaluation=True))
    write(args.out/'training_report.json', dict(epochs=epoch,best_epoch=best_epoch, plateau_criterion_met=True,
        gradient_evidence=gradients,training_seconds=sum(row['seconds'] for row in history),
        peak_memory_mib=max(row['peak_mib'] for row in history),pretrained_LEADER_initialization=True,
        joint_backbone_finetuning=True, real_images=True,lr_reductions=reductions))
    test = evaluate(args.out/'best.pt', args.out/'selected_test', 'test', args)
    value, eligible = objective(test, baseline['test'])
    report = dict(goal_met=eligible and value<=.9, best_epoch=best_epoch,
        metrics={k:v for k,v in test.items() if k!='rows'}, selected_minus_L0=compare(test, baseline['test']),
        mpe_reduction=1-test['all_frame_mpe_mean_m']/baseline['test']['all_frame_mpe_mean_m'],
        moe_reduction=1-test['all_frame_moe_mean_deg']/baseline['test']['all_frame_moe_mean_deg'],
        development_only=True, elapsed_seconds=time.perf_counter()-started)
    write(args.out/'comparison.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
