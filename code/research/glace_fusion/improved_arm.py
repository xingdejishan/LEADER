"""Local from-scratch training arm with the stage-4 improved recipe.

Mirrors `local.py train` exactly (same scene, seed, train_args, buffer,
PyTorch-compat patch) and additionally applies
`upgrade_lidar_supervision` to the copied vendor, so the arm trains with the
decomposed auxiliary (pixel bearing + log-depth) and the per-cell reliability
head while the baseline arm keeps the legacy camera-frame L1.

Windows: re-invokes itself inside the WSL glace environment (same forwarding
as wsl.py). Inside WSL it prepares the run directory and launches training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', default='selected')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--bearing-weight', type=float, default=1.0)
    parser.add_argument('--depth-weight', type=float, default=5.0)
    parser.add_argument('--bearing-beta-px', type=float, default=1.0)
    parser.add_argument('--no-reliability-head', action='store_true')
    parser.add_argument('--reliability-lr', type=float, default=1e-3)
    parser.add_argument('--depth-ratio-tol', type=float, default=1.25)
    return parser.parse_args()


def forward_to_wsl(args):
    config = json.loads((ROOT / 'WSL_ENVIRONMENT.json').read_text(encoding='utf-8'))
    prefix = ['wsl.exe', '-d', config['distribution']]

    def convert(path):
        return subprocess.check_output(
            prefix + ['--exec', 'wslpath', '-u', str(path).replace('\\', '/')],
            text=True).strip()

    linux_root = convert(ROOT)
    forwarded = []
    for value in sys.argv[1:]:
        if len(value) > 2 and value[1] == ':':
            forwarded.append(convert(value))
        else:
            forwarded.append(value.replace('\\', '/'))
    command = prefix + ['--cd', linux_root, '--exec', 'env',
                        'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=2',
                        'PYTHONPATH=' + linux_root + '/code',
                        config['python'], '-m', 'research.glace_fusion.improved_arm'] + forwarded
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    if os.name == 'nt':
        forward_to_wsl(args)
        return
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError('Choose a new --out path: ' + str(args.out))
    sys.path.insert(0, str(ROOT / 'code'))
    from training_compat import patch_training_vendor
    from research.glace_fusion.upgrade_lidar_supervision import upgrade_lidar_supervision

    model = ROOT / 'models' / args.variant
    base_config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True)
    shutil.copytree(model / 'vendor', args.out / 'vendor')
    compatibility = patch_training_vendor(args.out / 'vendor')
    upgrade = upgrade_lidar_supervision(
        args.out / 'vendor', overall_weight=base_config.get('auxiliary_weight', 1.0),
        depth_weight=args.depth_weight, bearing_weight=args.bearing_weight,
        bearing_beta_px=args.bearing_beta_px,
        reliability_head=not args.no_reliability_head,
        reliability_lr=args.reliability_lr, depth_ratio_tol=args.depth_ratio_tol)
    train_args = list(base_config['train_args'])
    train_args[train_args.index('--max_iterations') + 1] = str(args.iterations)
    train_args[train_args.index('--training_buffer_size') + 1] = str(907 * 1024)
    config = dict(base_config)
    config.update(train_args=train_args, train_images=907, local_run=True,
                  training_from_scratch=True,
                  auxiliary_images=len(list((ROOT / 'data/train_scene/train/lidar_world').glob('*.npy'))),
                  training_runtime_compatibility=compatibility,
                  stage4_upgrade=upgrade,
                  vendor_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.out / 'vendor').glob('*.py')})
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = dict(os.environ, GLACE_SEED='2089')
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nnodes', '1', '--nproc_per_node', '1',
                    str(args.out / 'vendor/train_ace.py'), str(ROOT / 'data/train_scene'),
                    str(args.out / 'head.pt')] + train_args,
                   cwd=args.out / 'vendor', env=env, check=True)


if __name__ == '__main__':
    main()
