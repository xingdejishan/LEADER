import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.magic_nclt_runtime import sha256_file, source_hashes


def execute(command, log, environment):
    with log.open('x', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-B', '-u', '-m', *command], env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    assert sys.prefix == str(Path.home() / '.conda/envs/leader')
    assert root.stat().st_uid == os.getuid() and not out.exists()
    out.mkdir(parents=True, mode=0o700)
    environment = dict(os.environ, LEADER_RESOURCE_GUARDS_DISABLED='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONDONTWRITEBYTECODE='1')
    for name in ('LEADER_DISK_GUARD_PATH', 'LEADER_DISK_MIN_FREE_BYTES'):
        environment.pop(name, None)
    frozen = source_hashes(root)
    (out / 'source_frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    state = {'utc': datetime.now(timezone.utc).isoformat(), 'pipeline_pid': os.getpid(),
             'gpu': environment.get('CUDA_VISIBLE_DEVICES'), 'resource_guards': False,
             'gpu_sharing': True, 'stage': 'same_code_l0_prediction'}
    (out / 'pipeline_state.json').write_text(json.dumps(state, indent=2) + '\n')
    l0 = out / 'l0'
    l0.mkdir()
    cache_sha = sha256_file(args.dev_sam)
    common = ['--data-root', str(args.data), '--manifest-root', str(args.data),
              '--sam-cache-manifest', str(args.dev_sam), '--sam-cache-sha256', cache_sha]
    execute(['tools.predict_magic_nclt', *common, '--checkpoint', str(args.l0),
             '--predictions', str(l0 / 'predictions.jsonl'), '--freeze', str(l0 / 'prediction_freeze.json'),
             '--seed', '20', '--device', 'cuda:0'], l0 / 'predict.log', environment)
    cpu_environment = dict(environment, CUDA_VISIBLE_DEVICES='-1')
    execute(['tools.evaluate_magic_nclt_dev', '--predictions', str(l0 / 'predictions.jsonl'),
             '--freeze', str(l0 / 'prediction_freeze.json'), '--checkpoint', str(args.l0),
             '--predictor', str(root / 'tools/predict_magic_nclt.py'), '--manifest-root', str(args.data),
             '--dev-gt-root', str(args.data / 'evaluator_only/test_gt'),
             '--role', 'l0_reference_development_validation', '--report', str(l0 / 'evaluation.json')],
            l0 / 'evaluate.log', cpu_environment)
    baseline = json.loads((l0 / 'evaluation.json').read_text())['overall']
    assert baseline['frames'] == 1200 and baseline['failure_frames'] == 0
    assert abs(baseline['mpe_mean_m'] - 0.20252352391525688) < 0.001
    assert abs(baseline['moe_mean_deg'] - 1.6661274113698579) < 0.01
    assert source_hashes(root) == frozen
    processes = {}
    streams = []
    for mode in ('mma_only', 'lidar_finetune'):
        arm = out / mode
        arm.mkdir()
        command = [sys.executable, '-B', '-u', '-m', 'tools.train_magic_nclt',
                   '--data-root', str(args.data), '--train-manifest-root', str(args.data),
                   '--dev-manifest-root', str(args.data), '--train-sam-cache-manifest', str(args.train_sam),
                   '--dev-sam-cache-manifest', str(args.dev_sam), '--dev-sam-cache-sha256', cache_sha,
                   '--dev-gt-root', str(args.data / 'evaluator_only/test_gt'), '--base-checkpoint', str(args.l0),
                   '--l0-report', str(l0 / 'evaluation.json'), '--out', str(arm / 'training'),
                   '--batch-size', '4', '--seed', '20', '--device', 'cuda:0', '--log-every', '10',
                   '--fusion-mode', mode, '--disable-resource-guards', '--min-free-gib', '0', '--start-free-gib', '0']
        stream = (arm / 'training.log').open('x')
        streams.append(stream)
        child = subprocess.Popen(command, cwd=str(root), env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        processes[mode] = child
        (arm / 'launch.json').write_text(json.dumps({'pid': child.pid, 'command': command,
                                                   'source': frozen, 'gpu_sharing': True,
                                                   'resource_guards': False}, indent=2) + '\n')
    state['stage'] = 'paired_training'
    state['training_pids'] = {mode: child.pid for mode, child in processes.items()}
    (out / 'pipeline_state.json').write_text(json.dumps(state, indent=2) + '\n')
    exits = {mode: child.wait() for mode, child in processes.items()}
    for stream in streams:
        stream.close()
    (out / 'exit.json').write_text(json.dumps(exits, indent=2) + '\n')
    if any(value != 0 for value in exits.values()):
        raise RuntimeError('An arm stopped; preserve all evidence and inspect exit.json')
    assert source_hashes(root) == frozen
    report = {'role': 'development_validation', 'independent_test': False, 'baseline': baseline, 'arms': {}}
    for mode in processes:
        training = out / mode / 'training'
        selection = json.loads((training / 'selection_frozen.json').read_text())
        history = [json.loads(line) for line in (training / 'history.jsonl').read_text().splitlines()]
        assert len(history) == 40 and sha256_file(training / 'best.pt') == selection['checkpoint_sha256']
        best = min(history, key=lambda row: (row['validation_failure_frames'], row['normalized_pose_objective'], row['epoch']))
        evaluation = next(path for path in (training / 'validation').glob('epoch_*_attempt_*/evaluation.json')
                          if sha256_file(path) == best['evaluation_sha256'])
        scored = json.loads(evaluation.read_text())
        assert scored['overall']['frames'] == 1200
        assert all(value['frames'] == 300 for value in scored['per_date'].values())
        report['arms'][mode] = {'best_epoch': best['epoch'], 'checkpoint_sha256': selection['checkpoint_sha256'],
                               'overall': scored['overall'], 'per_date': scored['per_date']}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    state['stage'] = 'completed_40_epochs_each'
    (out / 'pipeline_state.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--train-sam', type=Path, required=True)
    parser.add_argument('--dev-sam', type=Path, required=True)
    parser.add_argument('--l0', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
