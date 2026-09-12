import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    patches = ROOT / 'LOCAL_PATCHES.json'
    if patches.exists():
        manifest.update(json.loads(patches.read_text(encoding='utf-8')))
    for name, row in manifest.items():
        path = ROOT / name
        if path.stat().st_size != row['size']:
            raise ValueError('File size mismatch: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1048576), b''):
                h.update(block)
        if h.hexdigest() != row['sha256']:
            raise ValueError('Checksum mismatch: ' + name)
    counts = {label: len(list((ROOT / 'data' / (label + '_scene') / split / 'rgb').glob('*.jpg')))
        for label, split in [('train', 'train'), ('validation', 'train'), ('test', 'test')]}
    if counts != dict(train=907, validation=303, test=148):
        raise ValueError(counts)
    print(json.dumps(dict(verified_files=len(manifest), images=counts), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'summary', 'audit', 'joint', 'confidence', 'infer', 'train', 'rotation', 'replay', 'refine-ablation'])
    parser.add_argument('--variant', choices=['selected', 'previous', 'balanced', 'confidence'], default='selected')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
        return
    if args.command == 'summary':
        print((ROOT / 'PROVENANCE.json').read_text(encoding='utf-8'))
        for name in ['raw_audit.json', 'filtered_audit.json']:
            path = ROOT / 'reports/depth' / name
            if path.exists():
                report = json.loads(path.read_text(encoding='utf-8'))
                print(name, json.dumps(report['summary']['all']['metrics'], indent=2))
        return
    out = args.out or ROOT / 'outputs' / (args.command + '-' + args.variant)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError('Choose a new --out path: ' + str(out))
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'code'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
    if args.command == 'audit':
        import numpy as np
        from research.glace_fusion.correspondence_audit import metrics, summarize
        rows = json.loads((ROOT / 'data/test_rows.json').read_text(encoding='utf-8'))
        meta = json.loads((ROOT / 'data/train_scene/scene_meta.json').read_text(encoding='utf-8'))
        E = np.asarray(meta['T_BC_camera_to_body'])
        records = []
        for row in rows[:args.limit or None]:
            data = np.load(ROOT / 'cache' / args.variant / 'coordinates' / (row['image'] + '.npz'))
            scan = ROOT / 'data/scans' / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')
            records.append(dict(row, metrics=metrics(data['xyz'], data['uv'], data['K'], data['GT'], E, scan)))
        out.write_text(json.dumps(dict(records=records, summary=summarize(records)), indent=2), encoding='utf-8')
        print(str(out))
        return
    if args.command == 'refine-ablation':
        if args.variant not in ('selected', 'balanced'):
            raise ValueError('Fixed replay inputs are available for selected and balanced')
        arguments = ['fixed_origin_refine', '--bundle', str(ROOT), '--variant', args.variant,
            '--out', str(out), '--limit', str(args.limit)]
    elif args.command == 'replay':
        if args.variant == 'confidence':
            raise ValueError('Choose a neural coordinate cache for controlled replay')
        arguments = ['correspondence_replay', '--bundle', str(ROOT), '--variant', args.variant,
            '--out', str(out), '--limit', str(args.limit)]
        if args.reference:
            arguments.extend(['--reference', str(args.reference.resolve())])
    elif args.command == 'rotation':
        if args.variant == 'confidence':
            raise ValueError('Choose selected, balanced or previous for rotation diagnostics')
        arguments = ['rotation_complementarity', '--bundle', str(ROOT), '--variant', args.variant,
            '--out', str(out), '--limit', str(args.limit)]
    elif args.command == 'joint':
        arguments = ['cached_joint_eval', '--coordinates', str(ROOT / 'cache' / args.variant / 'coordinates'),
            '--out', str(out), '--bundle', str(ROOT), '--limit', str(args.limit)]
    elif args.command == 'confidence':
        arguments = ['correspondence_confidence', '--validation', str(ROOT / 'cache/validation_selected'),
            '--test', str(ROOT / 'cache/selected'), '--out', str(out), '--quality-target', 'joint3d', '--bundle', str(ROOT)]
    elif args.command == 'infer':
        if args.variant == 'confidence':
            raise ValueError('Use selected for neural inference; confidence is a filtered cache')
        model = args.model_dir.resolve() if args.model_dir else ROOT / 'models' / args.variant
        arguments = ['evaluate_scene', '--scene', str(ROOT / 'data/test_scene'), '--head', str(model / 'head.pt'),
            '--vendor', str(model / 'vendor'), '--deit-checkpoint', str(ROOT / 'models/global/CVPR23_DeitS_Rerank.pth'),
            '--valid-mask', str(ROOT / 'data/valid_mask.npy'), '--out', str(out), '--limit', str(args.limit)]
    else:
        if os.name == 'nt':
            from wsl import run
            run(sys.argv[1:])
            return
        if args.variant == 'confidence':
            raise ValueError('Choose a neural model variant')
        import shutil
        model = ROOT / 'models' / args.variant
        out.mkdir()
        shutil.copytree(model / 'vendor', out / 'vendor')
        from training_compat import patch_training_vendor
        compatibility = patch_training_vendor(out / 'vendor')
        config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
        train_args = list(config['train_args'])
        train_args[train_args.index('--max_iterations') + 1] = str(args.iterations)
        train_args[train_args.index('--training_buffer_size') + 1] = str(907 * 1024)
        config.update(train_args=train_args, train_images=907, local_run=True, training_from_scratch=True,
            auxiliary_images=len(list((ROOT / 'data/train_scene/train/lidar_world').glob('*.npy'))),
            training_runtime_compatibility=compatibility,
            vendor_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'vendor').glob('*.py')})
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
            '--nproc_per_node', '1', str(out / 'vendor/train_ace.py'), str(ROOT / 'data/train_scene'),
            str(out / 'head.pt')] + train_args, cwd=out / 'vendor', env=dict(env, GLACE_SEED='2089'), check=True)
        return
    subprocess.run([sys.executable, '-m', 'research.glace_fusion.' + arguments[0]] + arguments[1:],
        cwd=ROOT / 'code', env=env, check=True)


if __name__ == '__main__':
    main()
