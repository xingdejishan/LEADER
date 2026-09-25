import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from tools.compare_local905 import compare
from tools.run_surface_trial import digest, evaluate, run


def verified_result(directory, frames):
    prediction = directory / 'predictions.json'
    result = json.loads((directory / 'evaluation.json').read_text())
    if (result['predictions_sha256'] != digest(prediction) or result['frames'] != frames or
            len(result['rows']) != frames or
            (directory / 'predictions.sha256').read_text().strip() != digest(prediction)):
        raise ValueError(f'Invalid frozen result: {directory}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if shutil.disk_usage(args.out.parent).free < 5 * 1024 ** 3:
        raise RuntimeError('Less than 5GiB free')
    history = {}
    for subset, count in (('val', 40), ('test', 313)):
        history[subset] = {variant: verified_result(args.previous / f'{variant}_{subset}', count)
                           for variant in ('L0', 'box', 'surface')}
    old_settings = json.loads((args.previous / 'box/settings.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    sources = sorted({*Path('models').rglob('*.py'), *Path('tools').rglob('*.py'),
                      *Path('data').rglob('*.py'), *Path('utils').rglob('*.py'), Path('run_mink.py'),
                      Path('experiments/spatial_decoder/PLAN.md'),
                      Path('experiments/spatial_decoder/audit.json')})
    protocol = dict(protocol='spatial_decoder_trial_v1', started_unix=time.time(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    trained_conditions=['spatial'], reused_conditions=['L0', 'box', 'surface'],
                    source_sha256={str(p): digest(p) for p in sources}, previous=str(args.previous),
                    development_only=True, steps=138, selection='fixed_final_step')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    with zipfile.ZipFile(args.out / 'executed_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path))
    for subset in history:
        for variant in history[subset]:
            shutil.copytree(args.previous / f'{variant}_{subset}', args.out / f'{variant}_{subset}')
    run(['tools.train_spatial_trial', '--data_root', args.data_root, '--assets', args.assets,
         '--out', args.out / 'spatial'], args.out / 'spatial_train.log')
    settings = json.loads((args.out / 'spatial/settings.json').read_text())
    matched = ('sample_order_sha256', 'split_sha256', 'sam_manifest_sha256',
               'base_checkpoint_sha256', 'fusion_init_sha256', 'effective_batch', 'microbatch',
               'optimizer_steps', 'fusion_lr', 'seed', 'data_seed')
    if any(settings[key] != old_settings[key] for key in matched):
        raise ValueError('Training settings differ from historical box beyond architecture')
    checkpoint = args.out / 'spatial/final.pt'
    checkpoint_hash = digest(checkpoint)
    report = dict(protocol=protocol['protocol'], development_only=True, subsets={},
                  matched_training_settings={key: settings[key] for key in matched},
                  checkpoint_sha256=checkpoint_hash)
    for subset, count in (('val', 40), ('test', 313)):
        evaluate(checkpoint, 'spatial', subset, args)
        results = {**history[subset], 'spatial': verified_result(args.out / f'spatial_{subset}', count)}
        report['subsets'][subset] = {
            'metrics': {v: {key: val for key, val in result.items() if key != 'rows'}
                        for v, result in results.items()},
            **{f'spatial_minus_{v}': compare(results['spatial'], results[v])
               for v in ('L0', 'box', 'surface')}}
    if digest(checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation')
    metrics = report['subsets']['test']['metrics']
    keys = ('all_frame_mpe_mean_m', 'all_frame_moe_mean_deg',
            'all_frame_mpe_p90_m', 'all_frame_moe_p90_deg')
    report['passed_screen'] = (
        all(metrics[v]['failed_frames'] == 0 for v in ('spatial', 'L0', 'box')) and
        all(metrics['spatial'][key] is not None and metrics[other][key] is not None and
            (metrics['spatial'][key] < metrics[other][key] if 'mean' in key else
             metrics['spatial'][key] <= metrics[other][key])
            for other in ('L0', 'box') for key in keys))
    report['elapsed_total_seconds'] = time.time() - protocol['started_unix']
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
