import argparse
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_mink import LEADER
from run_mink import TRR, get_data_loader
from tools.compare_local905 import compare
from tools.run_surface_trial import run
from tools.train_local905 import batch_loss, digest
from tools.train_magic_revision import move_optimizer_state, sample_batches, write_torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def evaluate(checkpoint, directory, subset, args, magic=True):
    command = ['tools.eval_local905_online', '--data_root', args.data_root,
         '--split', args.assets / 'split_masked.json', '--checkpoint', checkpoint,
         '--subset', subset, '--out', directory]
    if magic:
        command += ['--sam_manifest', args.assets / 'sam_cache/manifest_905.json']
    run(command, directory.parent / (directory.name + '_online.log'))
    frozen_hash = digest(directory / 'predictions.json')
    run(['tools.eval_local905_gt', '--data_root', args.data_root,
         '--split', args.assets / 'split_masked.json',
         '--predictions', directory / 'predictions.json',
         '--out', directory / 'evaluation.json'],
        directory.parent / (directory.name + '_gt.log'))
    result = read(directory / 'evaluation.json')
    if result['predictions_sha256'] != frozen_hash or digest(directory / 'predictions.json') != frozen_hash:
        raise RuntimeError('Frozen prediction integrity failed')
    expected = 40 if subset == 'val' else 313
    if result['frames'] != expected or len(result['rows']) != expected:
        raise RuntimeError('Incomplete evaluation denominator')
    return result


def score(result, baseline):
    keys = ('all_frame_mpe_mean_m', 'all_frame_moe_mean_deg',
            'all_frame_mpe_p90_m', 'all_frame_moe_p90_deg')
    eligible = result['failed_frames'] == 0 and all(
        result[k] is not None and math.isfinite(result[k]) and result[k] <= baseline[k]
        for k in keys)
    objective = sum(result[k] / baseline[k] for k in keys[:2]) / 2 if eligible else None
    return eligible, objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parity_reference', type=Path)
    parser.add_argument('--initial_run', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if shutil.disk_usage(args.out.parent).free < 5 * 1024 ** 3:
        raise RuntimeError('Less than 5GiB free')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = args.prior / 'box/final.pt'
    source_hash = digest(source)
    if source_hash != '1f2157fb405af8c7fa5112f44a0cd0c7f0bb59ef48fcb0af63d0a62ae4df1e3f':
        raise ValueError('Unexpected starting checkpoint')
    for subset in ('val', 'test'):
        for variant in ('box', 'L0'):
            folder = args.prior / f'{variant}_{subset}'
            if read(folder / 'evaluation.json')['predictions_sha256'] != digest(folder / 'predictions.json'):
                raise ValueError('Historical predictions/evaluation hash mismatch')
    order, order_hash = sample_batches(552, 8, 690, 100037)
    protocol = dict(name='magic_real_training_refinement_v3', seed=37, data_seed=100037,
                    initial_step=138, additional_steps=552, optimizer='new Adam state',
                    lr='cosine 1e-4 to 1e-5', effective_batch=8, microbatch=2,
                    source_checkpoint_sha256=source_hash, sample_order_sha256=order_hash,
                    selection_steps=[138, 276, 414, 552, 690],
                    selection='val40 minimum normalized MPE/MOE mean; both means and P90 <= L0',
                    real_images_only=True, development_only=True, smoke=args.smoke,
                    reference_policy='Recompute L0 and old box in current fixed runtime',
                    initial_run=str(args.initial_run) if args.initial_run else None,
                    parity_tolerances=dict(rotation_component=1e-5, translation_component_m=2e-4),
                    parity_reference_sha256=digest(args.parity_reference) if args.parity_reference else None,
                    historical_evaluation_sha256={
                        f'{v}_{s}': digest(args.prior / f'{v}_{s}/evaluation.json')
                        for v in ('box', 'L0') for s in ('val', 'test')},
                    source_sha256={name: digest(Path(name)) for name in (
                        'tools/refine_magic_training.py', 'tools/train_local905.py',
                        'tools/train_magic_revision.py', 'models/model_mink.py',
                        'models/magic_fusion.py', 'run_mink.py',
                        'tools/eval_local905_online.py', 'tools/eval_local905_gt.py',
                        'data/local905_query.py', 'experiments/magic_training/PLAN.md')})
    write(args.out / 'protocol.json', protocol)
    if not args.smoke:
        if args.initial_run is None:
            raise ValueError('Initial same-runtime evaluations are required')
        for name, checkpoint_path in (('L0_val', args.assets / 'official_l0.pt'), ('val_00138', source)):
            folder = args.initial_run / name
            predictions = read(folder / 'predictions.json')
            if (predictions['checkpoint_sha256'] != digest(checkpoint_path) or
                    read(folder / 'evaluation.json')['predictions_sha256'] != digest(folder / 'predictions.json')):
                raise ValueError('Initial evaluation checkpoint or output hash mismatch')
            shutil.copytree(folder, args.out / name)
        baseline_val = read(args.out / 'L0_val/evaluation.json')
        initial_val = read(args.out / 'val_00138/evaluation.json')
        if args.parity_reference is None:
            raise ValueError('Same-runtime starting prediction reference is required')
        old = read(args.parity_reference)
        new = read(args.out / 'val_00138/predictions.json')
        old_rows = old.get('rows', old.get('predictions'))
        new_rows = new.get('rows', new.get('predictions'))
        if (old_rows is None or new_rows is None or len(old_rows) != 40 or len(new_rows) != 40 or
                any(any(a.get(k) != b.get(k) for k in ('scan', 'status'))
                    for a, b in zip(old_rows, new_rows))):
            raise RuntimeError('Starting prediction identifiers differ')
        differences = np.abs(np.asarray([a['T_world_body'] for a in old_rows]) -
                             np.asarray([a['T_world_body'] for a in new_rows]))
        rotation_difference = float(differences[:, :3, :3].max())
        translation_difference = float(differences[:, :3, 3].max())
        if rotation_difference > 1e-5 or translation_difference > 2e-4:
            raise RuntimeError('Starting pose repeat exceeds pre-training numeric tolerances')
        write(args.out / 'initial_parity.json', dict(frames=len(old_rows), exact_pose_parity=False,
              numeric_tolerance_passed=True, max_rotation_component=rotation_difference,
              max_translation_component_m=translation_difference,
              old_predictions_sha256=digest(args.parity_reference),
              new_predictions_sha256=digest(args.out / 'val_00138/predictions.json')))
    else:
        baseline_val = read(args.prior / 'L0_val/evaluation.json')
        initial_val = read(args.prior / 'box_val/evaluation.json')
    eligible, best_score = score(initial_val, baseline_val)
    if not eligible:
        raise RuntimeError('Starting model fails frozen validation admission')
    best_step, best_path = 138, source
    validation_records = [dict(step=138, eligible=True, score=best_score)]
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    np.random.seed(37)
    torch.backends.cudnn.deterministic = True
    original = torch.load(source, map_location='cpu')
    if (original['settings']['split_sha256'] != digest(args.assets / 'split_masked.json') or
            original['settings']['sam_manifest_sha256'] !=
            digest(args.assets / 'sam_cache/manifest_905.json')):
        raise ValueError('Training split or SAM cache differs from starting checkpoint')
    model = LEADER(in_channels=3, out_channels=4, magic=True, fusion_variant='box').cuda()
    model.load_state_dict(original['model'], strict=True)
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.eval()
    model.magic_fusion.train()
    center = torch.tensor(original['center_t'], device='cuda')
    settings = dict(original['settings'])
    settings.update(protocol=protocol['name'], variant='B_real_refined', fusion_variant='box',
                    parent_checkpoint_sha256=source_hash, optimizer_state_restarted=True,
                    additional_steps=552, optimizer_steps=690,
                    training_protocol_sha256=digest(args.out / 'protocol.json'))
    flags = SimpleNamespace(dataset='Local905', dataset_folder=str(args.data_root),
                            local905_split=str(args.assets / 'split_masked.json'),
                            local905_max_points=0, voxel_size=0.2, horizontal_res=1024,
                            batch_size=2, val_batch_size=1, num_workers=0, mode='train',
                            magic_manifest=str(args.assets / 'sam_cache/manifest_905.json'))
    loader, _ = get_data_loader(flags)
    if len(loader.dataset) != 552:
        raise ValueError('Wrong training denominator')
    optimizer = torch.optim.Adam(model.magic_fusion.parameters(), lr=1e-4)
    loss_fn = TRR(scale=10)
    torch.cuda.reset_peak_memory_stats()
    updates = 1 if args.smoke else 552
    training_seconds = 0.0
    with (args.out / 'history.jsonl').open('w', encoding='utf-8') as stream:
        for update in range(updates):
            step = 138 + update
            lr = 1e-5 + 0.5 * 9e-5 * (1 + math.cos(math.pi * update / 551))
            for group in optimizer.param_groups:
                group['lr'] = lr
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(4):
                samples = [loader.dataset[int(i)] for i in order[step, micro * 2:micro * 2 + 2]]
                batch = loader.collate_fn(samples)
                loss, raw = batch_loss(model, batch, center, loss_fn, True, 0.2, 1024)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss / 4).backward()
                total += float(loss.detach()) / 4
                del loss, raw, batch, samples
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.magic_fusion.parameters()):
                raise FloatingPointError('Nonfinite fusion gradient')
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            training_seconds += duration
            completed = step + 1
            row = dict(step=completed, phase_step=update + 1, trr=total, lr=lr,
                       seconds=duration, peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if (update + 1) % 23 == 0 or args.smoke:
                print(json.dumps(row), flush=True)
            if (update + 1) % 138 == 0 and not args.smoke:
                model.cpu()
                move_optimizer_state(optimizer, 'cpu')
                torch.cuda.empty_cache()
                unchanged = all(torch.equal(model.state_dict()[k], v)
                                for k, v in original['model'].items() if not k.startswith('magic_fusion.'))
                if not unchanged:
                    raise RuntimeError('Frozen LEADER changed')
                candidate = args.out / f'step_{completed:05d}.pt'
                write_torch(candidate, dict(model=model.state_dict(), center_t=center.tolist(),
                            settings=settings, optimizer_step=completed, optimizer=optimizer.state_dict()))
                result = evaluate(candidate, args.out / f'val_{completed:05d}', 'val', args)
                eligible, value = score(result, baseline_val)
                record = dict(step=completed, eligible=eligible, score=value,
                              mpe=result['all_frame_mpe_mean_m'], moe=result['all_frame_moe_mean_deg'],
                              checkpoint_sha256=digest(candidate))
                validation_records.append(record)
                if eligible and value < best_score:
                    best_score, best_step, best_path = value, completed, candidate
                write(args.out / 'validation_selection.json', dict(
                    candidates=validation_records, best_step=best_step, best_score=best_score))
                print(json.dumps(record), flush=True)
                model.cuda()
                move_optimizer_state(optimizer, 'cuda')
                model.eval()
                model.magic_fusion.train()
    model.cpu()
    unchanged = all(torch.equal(model.state_dict()[k], v)
                    for k, v in original['model'].items() if not k.startswith('magic_fusion.'))
    changed = any(not torch.equal(model.state_dict()[k], v)
                  for k, v in original['model'].items() if k.startswith('magic_fusion.'))
    if not unchanged or not changed:
        raise RuntimeError('Frozen state or trained fusion invariant failed')
    report = dict(completed_updates=updates, training_seconds=training_seconds,
                  frozen_leader_exactly_unchanged=unchanged, fusion_changed=changed,
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
                  best_step=best_step, best_validation_score=best_score, smoke=args.smoke)
    write(args.out / 'training_report.json', report)
    if args.smoke:
        print(json.dumps(report), flush=True)
        return
    move_optimizer_state(optimizer, 'cpu')
    del model, optimizer
    torch.cuda.empty_cache()
    selected = args.out / 'selected.pt'
    shutil.copy2(best_path, selected)
    write(args.out / 'selection_frozen.json', dict(step=best_step, score=best_score,
          checkpoint_sha256=digest(selected), source=str(best_path), before_test_evaluation=True))
    test = evaluate(selected, args.out / 'selected_test', 'test', args)
    old_test = evaluate(source, args.out / 'old_box_test', 'test', args)
    baseline_test = evaluate(args.assets / 'official_l0.pt', args.out / 'L0_test', 'test', args, False)
    summary = dict(selected_step=best_step, development_only=True, metrics={
                   name: {k: v for k, v in data.items() if k != 'rows'}
                   for name, data in [('selected', test), ('old_box', old_test), ('L0', baseline_test)]},
                   selected_minus_old_box=compare(test, old_test),
                   selected_minus_L0=compare(test, baseline_test),
                   elapsed_seconds=time.perf_counter() - started)
    write(args.out / 'comparison.json', summary)
    print(json.dumps(dict(completed=True, selected_step=best_step,
                         mpe=test['all_frame_mpe_mean_m'], moe=test['all_frame_moe_mean_deg'])), flush=True)


if __name__ == '__main__':
    main()
