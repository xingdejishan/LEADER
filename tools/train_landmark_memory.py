import argparse
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.landmark_memory import LandmarkMemory
from tools.compare_local905 import compare
from tools.eval_landmark_memory import predict, reference_tensors
from tools.run_surface_trial import run
from tools.train_local905 import digest
from tools.train_magic_revision import write_torch


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def evaluate(checkpoint, directory, subset, args):
    run(['tools.eval_landmark_memory', '--cache', args.cache, '--checkpoint', checkpoint,
         '--subset', subset, '--out', directory], directory.parent / (directory.name + '_online.log'))
    run(['tools.eval_local905_gt', '--data_root', args.data_root,
         '--split', args.assets / 'split_masked.json', '--predictions', directory / 'predictions.json',
         '--out', directory / 'evaluation.json'], directory.parent / (directory.name + '_gt.log'))
    result = read(directory / 'evaluation.json')
    if result['predictions_sha256'] != digest(directory / 'predictions.json'):
        raise ValueError('Prediction/evaluator fingerprint mismatch')
    return result


def objective(result, baseline):
    keys = ('all_frame_mpe_mean_m', 'all_frame_moe_mean_deg')
    if result['failed_frames'] or any(result[key] is None for key in keys):
        return 1e6, False
    ratios = [result[key] / baseline[key] for key in keys]
    eligible = all(ratio <= 1 for ratio in ratios)
    return (min(ratios) if eligible else 1 + sum(ratios) / 2), eligible


def main():
    parser = argparse.ArgumentParser()
    for name in ('data_root', 'assets', 'cache', 'previous', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source_commit', required=True)
    args = parser.parse_args()
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    np.random.seed(37)
    torch.backends.cudnn.deterministic = True
    manifest = read(args.cache / 'manifest.json')
    if (manifest['split_sha256'] != digest(args.assets / 'split_masked.json') or
            manifest['training_pairs_sha256'] != digest(args.cache / 'training_pairs.npz')):
        raise ValueError('Training cache identity mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_names = ('models/landmark_memory.py', 'tools/build_landmark_memory.py',
                    'tools/eval_landmark_memory.py', 'tools/train_landmark_memory.py',
                    'tests/test_landmark_memory.py', 'tools/eval_local905_gt.py',
                    'models/sc2pcr.py', 'utils/full_pool_robust_v1.py',
                    'experiments/landmark_memory/PLAN.md', 'experiments/landmark_memory/audit.json')
    protocol = dict(name='landmark_memory_v1', source_commit=args.source_commit,
                    source_sha256={name: digest(Path(name)) for name in source_names},
                    cache_manifest_sha256=digest(args.cache / 'manifest.json'),
                    min_epochs=40, initial_epoch_block=160, patience_epochs=30,
                    validation_every=5, minimum_lr_reductions=2,
                    optimizer='AdamW', initial_lr=3e-4, weight_decay=1e-4, batch_size=1024,
                    lr_scheduler='ReduceLROnPlateau factor=.5 patience=2 evaluations threshold=.001',
                    seed=37, selection='Eligible both means <= L0: minimum normalized MPE or MOE',
                    test_policy='One frozen validation-selected checkpoint only; development data',
                    baseline='L0 official + SC2-PCR + two-stage full pool')
    write(args.out / 'protocol.json', protocol)
    with zipfile.ZipFile(args.out / 'executed_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source_names:
            archive.write(name, name)
    baseline = {}
    for subset in ('val', 'test'):
        folder = args.previous / f'L0_{subset}'
        baseline[subset] = read(folder / 'evaluation.json')
        if baseline[subset]['predictions_sha256'] != digest(folder / 'predictions.json'):
            raise ValueError('Historical baseline fingerprint mismatch')
        shutil.copytree(folder, args.out / f'L0_{subset}')
    model = LandmarkMemory().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2,
                                                        threshold=.001, min_lr=1e-6)
    reference = reference_tensors(args.cache)
    with np.load(args.cache / 'training_pairs.npz', allow_pickle=False) as data:
        training = {key: torch.as_tensor(data[key], device='cuda') for key in
                    ('lidar', 'image', 'predicted', 'error', 'candidate_index', 'candidate_valid')}
    count = len(training['lidar'])
    records, history = [], []
    best_epoch, best_value, best_significant, last_significant = 0, math.inf, math.inf, 0
    reductions = 0
    gradient_evidence = {}
    torch.cuda.reset_peak_memory_stats()

    def save(path, epoch):
        write_torch(path, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                    cache_manifest_sha256=protocol['cache_manifest_sha256'],
                    protocol_sha256=digest(args.out / 'protocol.json'),
                    rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state()))

    initial = args.out / 'epoch_0000.pt'
    save(initial, 0)
    initial_result = evaluate(initial, args.out / 'val_0000', 'val', args)
    old = read(args.previous / 'L0_val/predictions.json')['predictions']
    new = read(args.out / 'val_0000/predictions.json')['predictions']
    if any(a['scan'] != b['scan'] or b['status'] != 'ok' for a, b in zip(old, new)):
        raise RuntimeError('Initial validation identifiers or success differ')
    differences = np.abs(np.asarray([r['T_world_body'] for r in old]) -
                         np.asarray([r['T_world_body'] for r in new]))
    parity = dict(max_rotation_component=float(differences[:, :3, :3].max()),
                  max_translation_component_m=float(differences[:, :3, 3].max()))
    write(args.out / 'initial_parity.json', parity)
    if parity['max_rotation_component'] > 1e-5 or parity['max_translation_component_m'] > 2e-4:
        raise RuntimeError('Zero correction baseline replay exceeds numeric tolerance')
    epoch = 0
    with (args.out / 'history.jsonl').open('w') as stream:
        while True:
            epoch += 1
            tick = time.perf_counter()
            model.train()
            order = torch.randperm(count, device='cuda')
            total = 0.0
            for offset in range(0, count, 1024):
                index = order[offset:offset + 1024]
                batch = {key: value[index] for key, value in training.items()}
                optimizer.zero_grad(set_to_none=True)
                delta = predict(model, reference, batch)
                loss = F.smooth_l1_loss(delta, batch['error'], beta=.1)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                if epoch == 1 and offset == 1024:
                    gradient_evidence = {name: float(parameter.grad.abs().sum())
                                         for name, parameter in model.named_parameters() if parameter.grad is not None}
                optimizer.step()
                total += float(loss.detach()) * len(index)
            torch.cuda.synchronize()
            row = dict(epoch=epoch, loss=total / count, seconds=time.perf_counter() - tick,
                       lr=optimizer.param_groups[0]['lr'], points=count,
                       peak_memory_mib=torch.cuda.max_memory_allocated() / 1024**2)
            history.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(json.dumps(row), flush=True)
            if epoch % 5:
                continue
            candidate = args.out / f'epoch_{epoch:04d}.pt'
            save(candidate, epoch)
            result = evaluate(candidate, args.out / f'val_{epoch:04d}', 'val', args)
            value, eligible = objective(result, baseline['val'])
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(value)
            if optimizer.param_groups[0]['lr'] < lr_before:
                reductions += 1
            if value < best_value:
                best_epoch, best_value = epoch, value
                shutil.copy2(candidate, args.out / 'best.pt')
            if value < best_significant * .999:
                best_significant, last_significant = value, epoch
            record = dict(epoch=epoch, objective=value, eligible=eligible,
                          mpe=result['all_frame_mpe_mean_m'], moe=result['all_frame_moe_mean_deg'],
                          checkpoint_sha256=digest(candidate), best_epoch=best_epoch,
                          lr_reductions=reductions, last_significant_epoch=last_significant)
            records.append(record)
            write(args.out / 'validation_selection.json', dict(records=records, best_epoch=best_epoch,
                                                              best_objective=best_value))
            save(args.out / 'latest.pt', epoch)
            print(json.dumps(record), flush=True)
            if epoch >= 40 and epoch - last_significant >= 30 and reductions >= 2:
                break
            if epoch % 160 == 0:
                write(args.out / f'continuation_{epoch:04d}.json', dict(
                    reason='Validation plateau criterion not met; continue same architecture and schedule',
                    last_significant_epoch=last_significant, best_epoch=best_epoch))
    selected = args.out / 'best.pt'
    write(args.out / 'selection_frozen.json', dict(epoch=best_epoch, checkpoint_sha256=digest(selected),
                                                  before_313_evaluation=True))
    write(args.out / 'training_report.json', dict(epochs=epoch, best_epoch=best_epoch,
          plateau_criterion_met=True, lr_reductions=reductions, gradient_evidence=gradient_evidence,
          training_seconds=sum(r['seconds'] for r in history), real_images=True,
          peak_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
          query_GT_not_in_online_cache=True))
    test = evaluate(selected, args.out / 'selected_test', 'test', args)
    score, eligible = objective(test, baseline['test'])
    report = dict(goal_met=eligible and score <= .9, best_epoch=best_epoch,
                  metrics={key: value for key, value in test.items() if key != 'rows'},
                  baseline_metrics={key: value for key, value in baseline['test'].items() if key != 'rows'},
                  selected_minus_L0=compare(test, baseline['test']),
                  mpe_reduction=1-test['all_frame_mpe_mean_m']/baseline['test']['all_frame_mpe_mean_m']
                  if test['all_frame_mpe_mean_m'] is not None else None,
                  moe_reduction=1-test['all_frame_moe_mean_deg']/baseline['test']['all_frame_moe_mean_deg']
                  if test['all_frame_moe_mean_deg'] is not None else None,
                  elapsed_seconds=time.perf_counter()-started, development_only=True)
    write(args.out / 'comparison.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
