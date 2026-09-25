import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.compare_local905 import compare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    print(json.dumps({'started': str(log), 'time': time.time()}), flush=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable, '-m', *map(str, command)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({'finished': str(log), 'time': time.time()}), flush=True)


def evaluate(checkpoint, variant, subset, args):
    directory = args.out / f'{variant}_{subset}'
    online = ['tools.eval_local905_online', '--data_root', args.data_root,
              '--split', args.assets / 'split_masked.json', '--checkpoint', checkpoint,
              '--subset', subset, '--out', directory]
    if variant != 'L0':
        online += ['--sam_manifest', args.assets / 'sam_cache/manifest_905.json']
    run(online, args.out / f'{variant}_{subset}_online.log')
    run(['tools.eval_local905_gt', '--data_root', args.data_root,
         '--split', args.assets / 'split_masked.json', '--predictions', directory / 'predictions.json',
         '--out', directory / 'evaluation.json'], args.out / f'{variant}_{subset}_gt.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if shutil.disk_usage(args.out.parent).free < 8 * 1024 ** 3:
        raise RuntimeError('Less than 8GiB free')
    args.out.mkdir(parents=True, exist_ok=False)
    plan = Path('experiments/surface_token/PLAN.md')
    protocol = {'protocol': 'surface_token_trial_v1', 'plan_sha256': digest(plan),
                'started_unix': time.time(), 'conditions': ['L0', 'box', 'surface'],
                'steps_per_trained_condition': 138, 'seed': 37, 'lr': 1e-4,
                'effective_batch': 8, 'microbatch': 2, 'selection': 'fixed_final_step',
                'source_sha256': {str(p): digest(p) for p in [
                    Path('tools/run_surface_trial.py'), Path('tools/train_surface_trial.py'),
                    Path('models/surface_token_fusion.py'), Path('models/magic_fusion.py'),
                    Path('models/model_mink.py'), Path('tools/eval_local905_online.py'),
                    Path('tools/eval_local905_gt.py'), Path('tools/train_local905.py'),
                    Path('data/local905_query.py')]}}
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for subset in ('val', 'test'):
        evaluate(args.assets / 'official_l0.pt', 'L0', subset, args)
    for variant in ('box', 'surface'):
        run(['tools.train_surface_trial', '--variant', variant, '--data_root', args.data_root,
             '--assets', args.assets, '--out', args.out / variant], args.out / f'{variant}_train.log')
    for variant in ('box', 'surface'):
        for subset in ('val', 'test'):
            evaluate(args.out / variant / 'final.pt', variant, subset, args)
    report = {'protocol': protocol['protocol'], 'development_only': True, 'subsets': {}}
    for subset in ('val', 'test'):
        results = {v: json.loads((args.out / f'{v}_{subset}/evaluation.json').read_text())
                   for v in ('L0', 'box', 'surface')}
        report['subsets'][subset] = {
            'metrics': {v: {k: x for k, x in r.items() if k != 'rows'} for v, r in results.items()},
            'surface_minus_L0': compare(results['surface'], results['L0']),
            'surface_minus_box': compare(results['surface'], results['box']),
            'box_minus_L0': compare(results['box'], results['L0'])}
    metrics = report['subsets']['test']['metrics']
    keys = ('all_frame_mpe_mean_m', 'all_frame_moe_mean_deg',
            'all_frame_mpe_p90_m', 'all_frame_moe_p90_deg')
    report['passed_screen'] = (
        all(metrics[v]['failed_frames'] == 0 for v in metrics) and
        all(metrics['surface'][k] is not None and metrics[other][k] is not None and
            (metrics['surface'][k] < metrics[other][k] if 'mean' in k else
             metrics['surface'][k] <= metrics[other][k])
            for other in ('L0', 'box') for k in keys))
    report['elapsed_total_seconds'] = time.time() - protocol['started_unix']
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed': True, 'passed_screen': report['passed_screen'],
                      'comparison': str(args.out / 'comparison.json')}), flush=True)


if __name__ == '__main__':
    main()
