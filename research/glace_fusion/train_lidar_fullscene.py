"""FULL-scene LiDAR-supervised GLACE training (stage-4 scale-up).

Takes the stage-2 recipe that was validated on a 40 m local region
(valid-FOV mask + sparse training-only LiDAR ray-target auxiliary) and applies
it to the complete stage-1 training scene (43,012 images across the four
LEADER training dates, R2Former RGB features, 480 px, full budget).

Run root layout (never overwritten):
    <root>/scene/train/{rgb,poses,calibration}  -> symlinks into the stage-1 scene
    <root>/scene/train/features.npy             -> symlink (R2Former, official protocol)
    <root>/scene/train/valid_mask.npy           -> copy of the stage-2 valid-FOV mask
    <root>/scene/test                           -> symlink into the stage-1 eval scene
    <root>/scene/lidar_world/<stem>.npy         -> float16 nominal-camera cloud (voxel downsampled)
    <root>/scene/lidar_world/<stem>_twc.npy     -> float32 nominal camera->world pose
    <root>/vendor/                              -> stage-1 vendor + valid-region + LiDAR patches
    <root>/head.pt / train.log / eval_{train,test}/

LiDAR targets are derived from training images only (exact-timestamp scans);
test LiDAR is never used for optimization or inference. Evaluation remains
RGB-only with the valid-region mask.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .fullscene_lidar_targets import build_targets
from .patch_lidar_supervision import patch_lidar_supervision
from .patch_valid_region import patch_valid_region
from .retrain_rgb_baseline import digest, write_json
from .valid_region import validate_mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True,
                        help='stage-1 full-scene run (scene + patched vendor), '
                             'e.g. /root/rivermind-data/glace_nclt_rgb_large_20260912')
    parser.add_argument('--eval-scene', type=Path, required=True,
                        help='scene providing the test split (rgb/poses/calibration/scene_meta)')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--valid-mask', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    parser.add_argument('--aux-weight', type=float, default=1.0)
    parser.add_argument('--aux-log-depth', action='store_true',
                        help='log-depth auxiliary residual instead of camera-frame Smooth L1')
    parser.add_argument('--max-points', type=int, default=8192)
    parser.add_argument('--voxel', type=float, default=0.1)
    parser.add_argument('--max-iterations', type=int, default=80000)
    parser.add_argument('--eval-limit', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2089)
    return parser.parse_args()


def prepare_scene(args, meta):
    scene = args.run_root / 'scene'
    train = scene / 'train'
    train.mkdir(parents=True)
    source_train = args.source_run / 'scene' / 'train'
    for name in ('rgb', 'poses', 'calibration', 'features.npy', 'features_manifest.json'):
        source = source_train / name
        if not source.exists():
            raise FileNotFoundError(source)
        (train / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    mask = validate_mask(np.load(args.valid_mask))
    shutil.copyfile(args.valid_mask, train / 'valid_mask.npy')
    (scene / 'test').symlink_to((args.eval_scene / 'test').resolve(), target_is_directory=True)
    merged = json.loads(json.dumps(meta))
    eval_meta = json.loads((args.eval_scene / 'scene_meta.json').read_text())
    merged['splits']['test'] = eval_meta['splits']['test']
    if not merged['splits']['test'].get('pairs'):
        raise ValueError('Eval scene provides no test pairs')
    write_json(scene / 'scene_meta.json', merged)
    return mask


def prepare_vendor(args, config_stage1):
    vendor = args.run_root / 'vendor'
    vendor.mkdir()
    source_vendor = args.source_run / 'vendor'
    base = (source_vendor / 'ace_trainer.py').read_text()
    if 'write_progress' not in base:
        raise ValueError('Source vendor lacks the retrain patch; refusing to chain patches')
    if 'lidar_folder' in base:
        raise ValueError('Source vendor already carries a LiDAR patch')
    for path in source_vendor.glob('*.py'):
        shutil.copyfile(path, vendor / path.name)
    shutil.copytree(source_vendor / 'datasets', vendor / 'datasets',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (vendor / 'ace_encoder_pretrained.pt').symlink_to(
        (source_vendor / 'ace_encoder_pretrained.pt').resolve())
    patch_valid_region(vendor)
    patch_lidar_supervision(vendor, weight=args.aux_weight, relative_storage=True,
                            log_depth=args.aux_log_depth)
    return vendor


def train_args_for(args, config_stage1):
    train_args = list(config_stage1['train_args'])
    key = '--max_iterations'
    if key not in train_args:
        raise ValueError('Stage-1 train_args lack --max_iterations')
    index = train_args.index(key)
    train_args[index + 1] = str(args.max_iterations)
    return train_args


def main():
    args = parse_args()
    if not np.isfinite(args.aux_weight) or args.aux_weight <= 0:
        raise ValueError('Auxiliary weight must be finite and positive')
    if args.run_root.exists():
        raise FileExistsError('Refusing to overwrite an existing run root')
    created = False
    try:
        args.run_root.mkdir(parents=True)
        created = True
        meta = json.loads((args.source_run / 'scene' / 'scene_meta.json').read_text())
        if len(meta['splits']['train']['pairs']) < 40000:
            raise ValueError('Source scene is not the full training scene')
        config_stage1 = json.loads((args.source_run / 'stage1_80k' / 'config.json').read_text()) \
            if (args.source_run / 'stage1_80k' / 'config.json').exists() \
            else json.loads((args.source_run / 'config.json').read_text())

        write_json(args.run_root / 'state.json', dict(stage='prepare', time=time.time()))
        prepare_scene(args, meta)
        vendor = prepare_vendor(args, config_stage1)
        train_args = train_args_for(args, config_stage1)

        write_json(args.run_root / 'state.json', dict(stage='lidar_targets', time=time.time()))
        written, missing = build_targets(
            args.run_root / 'scene' / 'train', meta, args.dataset_root,
            args.run_root / 'scene' / 'train' / 'lidar_world',
            max_points=args.max_points, voxel=args.voxel,
            progress_path=args.run_root / 'lidar_targets.progress.json')
        if len(written) < 0.9 * (len(written) + len(missing)):
            raise RuntimeError(f'Too many missing scans: {len(missing)} of {len(written) + len(missing)}')

        config = dict(train_args=train_args, aux_weight=args.aux_weight,
                      aux_loss='log-depth Smooth L1' if args.aux_log_depth
                      else 'Smooth L1 camera-frame 3D residual, beta=1m',
                      aux_storage='nominal-camera float16 cloud + nominal T_WC sidecar '
                                  f'(voxel {args.voxel} m, cap {args.max_points})',
                      train_images=len(meta['splits']['train']['pairs']),
                      lidar_images=len(written), missing_scan_images=len(missing),
                      seed=args.seed, max_iterations=args.max_iterations,
                      source_run=str(args.source_run), eval_scene=str(args.eval_scene),
                      valid_mask_sha256=digest(args.valid_mask),
                      encoder_sha256=digest(vendor / 'ace_encoder_pretrained.pt'),
                      global_backbone_sha256=digest(args.deit_checkpoint),
                      backbone_training=False,
                      vendor_hashes={p.name: digest(p) for p in vendor.glob('*.py')},
                      started=time.time(), ready_for_fusion=False)
        write_json(args.run_root / 'config.json', config)

        env = dict(os.environ, GLACE_SEED=str(args.seed))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                   '--nproc_per_node', '1', str(vendor / 'train_ace.py'),
                   str(args.run_root / 'scene'), str(args.run_root / 'head.pt')] + train_args
        write_json(args.run_root / 'state.json', dict(stage='training', command=command, time=time.time()))
        with (args.run_root / 'train.log').open('w') as log:
            subprocess.run(command, cwd=vendor, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        for split in ('train', 'test'):
            write_json(args.run_root / 'state.json', dict(stage='evaluation', split=split, time=time.time()))
            command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene',
                       '--scene', str(args.run_root / 'scene'), '--head', str(args.run_root / 'head.pt'),
                       '--vendor', str(vendor), '--deit-checkpoint', str(args.deit_checkpoint),
                       '--out', str(args.run_root / ('eval_' + split)), '--split', split,
                       '--limit', str(args.eval_limit),
                       '--valid-mask', str(args.run_root / 'scene' / 'train' / 'valid_mask.npy')]
            with (args.run_root / ('eval_' + split + '.log')).open('w') as log:
                subprocess.run(command, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(args.run_root / 'state.json', dict(stage='complete', time=time.time(),
                   head_sha256=digest(args.run_root / 'head.pt'),
                   next_step='full_region_validation + real_candidate_eval + fusion wiring'))
        print('FULLSCENE_LIDAR_COMPLETE ' + str(args.run_root), flush=True)
    except Exception:
        if created:
            write_json(args.run_root / 'state.json', dict(stage='failed', time=time.time(),
                       error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
