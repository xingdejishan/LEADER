import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (37, 53, 71)
FUSION_LRS = (0.0001, 0.0003, 0.001)
TOTAL_STEPS = 690
BATCH_SIZE = 8
EVAL_INTERVAL = 69


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(command, out):
    complete = out / 'complete.json'
    if complete.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'process.log').open('a', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not complete.exists():
        raise RuntimeError(f'Condition did not finish: {out}')


def retire_resume_state(out):
    if not (out / 'complete.json').exists():
        raise ValueError(f'Cannot retire incomplete condition: {out}')
    path = out / 'last.pt'
    if path.exists():
        path.unlink()


def retire_best(out):
    if not (out / 'complete.json').exists():
        raise ValueError(f'Cannot retire incomplete condition: {out}')
    path = out / 'best.pt'
    if path.exists():
        path.unlink()


def selected_learning_rates(calibration):
    selected = {}
    for condition in ('LFT', 'A', 'B', 'A-null', 'B-null'):
        choices = []
        for lr in FUSION_LRS:
            values = [calibration['runs'][f's{seed}_lr{lr:g}'][condition]['best_j']
                      for seed in SEEDS]
            choices.append((sum(values) / len(values), lr))
        selected[condition] = min(choices)[1]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path, required=True)
    parser.add_argument('--null_template', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--seed37_init', type=Path, required=True)
    parser.add_argument('--l0_val_report', type=Path, required=True)
    parser.add_argument('--gate_report', type=Path, required=True)
    parser.add_argument('--confirmation_protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('calibration', 'final'), required=True)
    parser.add_argument('--calibration_report', type=Path)
    args = parser.parse_args()
    if args.phase == 'final' and not args.calibration_report:
        raise ValueError('Final refit requires completed calibration results')
    gate = json.loads(args.gate_report.read_text(encoding='utf-8'))
    if not gate['passed'] or gate['max_points'] != 0:
        raise ValueError('Pretraining gates failed')
    if gate['base_checkpoint_sha256'] != digest(args.base_checkpoint):
        raise ValueError('Gated checkpoint differs')
    if gate['split_sha256'] != digest(args.split):
        raise ValueError('Gated split differs')
    if gate['sam_manifest_sha256'] != digest(args.sam_manifest):
        raise ValueError('Gated SAM cache differs')
    confirmation = json.loads(args.confirmation_protocol.read_text(encoding='utf-8'))
    if not confirmation['date_selected_before_formal_training']:
        raise ValueError('Final confirmation interval was not predesignated')
    protocol = {
        'protocol': 'magic_revision_full_grid_v2',
        'phase': args.phase,
        'seeds': SEEDS, 'fusion_lr_grid': FUSION_LRS,
        'total_steps_per_condition': TOTAL_STEPS,
        'warmup_steps': TOTAL_STEPS // 10,
        'leader_branch_steps': TOTAL_STEPS - TOTAL_STEPS // 10,
        'batch_size': BATCH_SIZE, 'eval_interval': EVAL_INTERVAL,
        'training_conditions': ['LFT', 'A', 'B', 'A-null', 'B-null'],
        'selection': 'For each condition and each seed, choose minimum validation J over fixed checkpoints including epoch 0. For each condition choose LR by lowest mean seed J; break ties with lower LR. Report all same-LR paired A/B and A-null/B-null results and the selected-LR results. Do not use diagnostic313 or final confirmation to choose LR/checkpoint.',
        'base_checkpoint_sha256': digest(args.base_checkpoint),
        'seed37_init_sha256': digest(args.seed37_init),
        'split_sha256': digest(args.split),
        'sam_manifest_sha256': digest(args.sam_manifest),
        'null_template_sha256': digest(args.null_template),
        'l0_val_report_sha256': digest(args.l0_val_report),
        'gate_report_sha256': digest(args.gate_report),
        'confirmation_protocol_sha256': digest(args.confirmation_protocol),
        'model_code_sha256': digest(Path('models/model_mink.py')),
        'fusion_code_sha256': digest(Path('models/magic_fusion.py')),
        'trainer_code_sha256': digest(Path('tools/train_magic_revision.py')),
        'calibration_report_sha256': digest(args.calibration_report)
        if args.calibration_report else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Formal protocol changed during resume')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    calibration = (json.loads(args.calibration_report.read_text(encoding='utf-8'))
                   if args.calibration_report else None)
    selected = selected_learning_rates(calibration) if calibration else None
    if calibration and calibration['protocol_sha256'] != digest(
            args.calibration_report.parent / 'protocol.json'):
        raise ValueError('Calibration protocol hash differs')
    for seed in SEEDS:
        init = args.seed37_init if seed == 37 else args.out / f'fusion_init_s{seed}.pt'
        if seed != 37 and not init.exists():
            subprocess.run([sys.executable, '-m', 'tools.prepare_local905_ab_init',
                            '--base_checkpoint', str(args.base_checkpoint),
                            '--out', str(init), '--seed', str(seed)], check=True)
        if seed == 37 and digest(init) != gate['fusion_init_sha256']:
            raise ValueError('Formal seed-37 initialization differs from gate')
        for lr in FUSION_LRS:
            if args.phase == 'final' and lr not in set(selected.values()):
                continue
            name = f's{seed}_lr{lr:g}'
            common = [sys.executable, '-m', 'tools.train_magic_revision',
                      '--data_root', str(args.data_root), '--split', str(args.split),
                      '--sam_manifest', str(args.sam_manifest),
                      '--base_checkpoint', str(args.base_checkpoint),
                      '--fusion_init', str(init),
                      '--l0_val_report', str(args.l0_val_report),
                      '--seed', str(seed), '--fusion_lr', str(lr),
                      '--total_steps', str(TOTAL_STEPS),
                      '--batch_size', str(BATCH_SIZE),
                      '--eval_interval', str(EVAL_INTERVAL)]
            lft = args.out / name / 'LFT'
            if args.phase == 'calibration' or selected['LFT'] == lr:
                run(common + ['--stage', 'LFT', '--out', str(lft)], lft)
                retire_resume_state(lft)
                if args.phase == 'calibration':
                    retire_best(lft)
            for is_null in (False, True):
                suffix = '-null' if is_null else ''
                stages = [stage for stage in ('A', 'B')
                          if args.phase == 'calibration' or selected[stage + suffix] == lr]
                if not stages:
                    continue
                extra = ['--null', '--null_template', str(args.null_template)] if is_null else []
                warm = args.out / name / f'warmup{suffix}'
                run(common + ['--stage', 'warmup', '--out', str(warm)] + extra, warm)
                for stage in stages:
                    out = args.out / name / f'{stage}{suffix}'
                    run(common + ['--stage', stage, '--out', str(out),
                                  '--warmup_checkpoint', str(warm / 'last.pt')] + extra, out)
                    retire_resume_state(out)
                    if args.phase == 'calibration':
                        retire_best(out)
                retire_resume_state(warm)
                warm_best = warm / 'best.pt'
                if warm_best.exists():
                    warm_best.unlink()
            print(json.dumps({'finished_seed': seed, 'fusion_lr': lr}), flush=True)
    if args.phase == 'calibration':
        runs = {}
        for seed in SEEDS:
            for lr in FUSION_LRS:
                key = f's{seed}_lr{lr:g}'
                runs[key] = {}
                for condition in ('LFT', 'A', 'B', 'A-null', 'B-null'):
                    path = args.out / key / condition / 'complete.json'
                    runs[key][condition] = json.loads(path.read_text(encoding='utf-8'))
        report = {'protocol': 'magic_revision_calibration_v1',
                  'protocol_sha256': digest(protocol_path), 'runs': runs}
        report['selected_learning_rates'] = selected_learning_rates(report)
        (args.out / 'calibration_report.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
