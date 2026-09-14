import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import run


def prepare(root, output):
    audit = json.loads((root / 'projection_audit/report.json').read_text())
    retained = {}
    for split in ['train', 'val', 'all']:
        rows = [r for r in audit['records'] if split == 'all' or r['split'] == split]
        geometric = sum(r['representative']['black_mask'] for r in rows)
        final = sum(r['representative']['depth_consistent'] for r in rows)
        retained[split] = dict(frames=len(rows), geometric=geometric, final=final,
                               total=sum(r['voxels'] for r in rows), retain=final / geometric,
                               rejected=geometric - final,
                               definition='positive depth, inside raster, valid undistortion region; before occlusion filtering')
    output.mkdir(parents=True, exist_ok=True)
    run.save_json(output / 'retention.json', retained)
    for name, source in [('lidar', root / 'lidar'), ('visual', root / 'visual_raw')]:
        target = output / name
        if target.exists():
            assert target.resolve() == source.resolve()
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    cache_hashes = {}
    for row in manifest:
        for name in ['lidar', 'visual']:
            path = output / name / (row['frame_id'] + '.npz')
            cache_hashes[str(path.relative_to(output))] = run.digest(path)
        with np.load(output / 'visual' / (row['frame_id'] + '.npz')) as data:
            assert np.isfinite(data['image']).all()
    run.save_json(output / 'manifest.json', manifest)
    protocol = dict(seed=2089, steps_per_arm=600, learning_rate=.0001, points_per_step=1024,
                    arms=['baseline', 'aligned', 'shuffled', 'aligned_wrong_eval', 'aligned_missing_eval'],
                    train_frames=64, val_frames=32, frozen=['LEADER.encoder', 'LEADER.decoder', 'DeDoDe/PCA'],
                    changed='projection sample point only; same legacy gate architecture and training budget',
                    projection='one actual raw representative from each stride-16 output cell; unchanged localization center',
                    visibility='unchanged 4px scan zbuffer, 0.5m tolerance, undistortion mask',
                    pass_rule='mean translation improves >=5%, mean rotation and p95 translation worsen <=5%, no additional 1m/5deg failures, and aligned beats shuffled mean translation',
                    ceiling_adjustment='Do not require positive rescue count: baseline is already 32/32 on this known development set',
                    scope='local development, same pretraining date; neither blind nor full NCLT',
                    cache_hashes=cache_hashes)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol differs; use a new output directory')
    run.save_json(protocol_path, protocol)
    print(json.dumps(retained, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--output', type=Path, default=Path('/home/zhang/leader-image-gate-raw'))
    args = parser.parse_args()
    root = Path('/home/zhang/leader-image-gate')
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare(root, args.output)
        return
    rows = json.loads((args.output / 'manifest.json').read_text())
    args.checkpoint = run.WORKSPACE / 'research/image_gate_checkpoint'
    if args.stage == 'train':
        run.train(args, rows)
        return
    run.evaluate(args, rows)
    result = json.loads((args.output / 'result.json').read_text())
    base, aligned, shuffled = [result['metrics'][k] for k in ['baseline', 'aligned', 'shuffled']]
    passed = aligned['mean'][0] <= .95 * base['mean'][0] and aligned['mean'][1] <= 1.05 * base['mean'][1]
    passed &= aligned['p95'][0] <= 1.05 * base['p95'][0] and result['damage'] == 0
    passed &= aligned['mean'][0] < shuffled['mean'][0]
    result['legacy_rescue_rule_passed'] = result.pop('passed')
    result['passed'] = bool(passed)
    result['next_action'] = 'confirm on more development data/seeds' if passed else 'no improvement demonstrated; do not tune projection to validation scores'
    run.save_json(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
