import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .patch_lidar_supervision import patch_lidar_supervision
from .retrain_rgb_baseline import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    parser.add_argument('--weight', type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.weight) or args.weight <= 0:
        raise ValueError('Auxiliary weight must be finite and positive')
    root = args.experiment / 'lidar_auxiliary'
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    try:
        scene = root / 'scene'
        (scene / 'train').mkdir(parents=True)
        for path in (args.experiment / 'masked/scene/train').iterdir():
            (scene / 'train' / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        (scene / 'test').symlink_to(args.experiment / 'scene/test', target_is_directory=True)
        (scene / 'scene_meta.json').symlink_to(args.experiment / 'scene/scene_meta.json')
        meta = json.loads((scene / 'scene_meta.json').read_text())
        E = np.asarray(meta['T_BC_camera_to_body'])
        rows = {r['image']: r for r in meta['splits']['train']['pairs']}
        test_stems = {r['image'] for r in meta['splits']['test']['pairs']}
        world_dir = scene / 'train/lidar_world'
        world_dir.mkdir()
        dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
        supervision = []
        missing = []
        for image in sorted((scene / 'train/rgb').iterdir()):
            if image.stem in test_stems:
                raise ValueError('Test image reached auxiliary training')
            row = rows[image.stem]
            scan = args.dataset_root / row['sequence'] / 'velodyne_sync' / (str(row['image_timestamp_us']) + '.bin')
            if not scan.exists():
                missing.append(image.stem)
                continue
            data = np.fromfile(scan, dtype=dtype)
            body = np.column_stack([data[k] for k in ['x', 'y', 'z']]).astype(float) * .005 - 100
            camera = (body - E[:3, 3]) @ E[:3, :3]
            body = body[(camera[:, 2] > 2) & (camera[:, 2] < 80)]
            gt = np.loadtxt(scene / 'train/poses' / (image.stem + '.txt'))
            T_WB = gt @ np.linalg.inv(E)
            world = body @ T_WB[:3, :3].T + T_WB[:3, 3]
            destination = world_dir / (image.stem + '.npy')
            np.save(destination, world.astype(np.float32))
            supervision.append(dict(image=image.stem, scan=str(scan), exact_timestamp=True,
                points=len(world), sha256=digest(destination)))
        write_json(root / 'supervision.json', dict(source='training images only; no test depth in optimization',
            missing=missing, samples=supervision))
        vendor = root / 'vendor'
        vendor.mkdir()
        for path in (args.experiment / 'vendor').glob('*.py'):
            shutil.copyfile(path, vendor / path.name)
        shutil.copytree(args.experiment / 'vendor/datasets', vendor / 'datasets', ignore=shutil.ignore_patterns('__pycache__'))
        (vendor / 'ace_encoder_pretrained.pt').symlink_to((args.experiment / 'vendor/ace_encoder_pretrained.pt').resolve())
        patch_lidar_supervision(vendor, args.weight)
        config = json.loads((args.experiment / 'masked/config.json').read_text())
        config.update(experiment_arm='masked_plus_sparse_lidar', started=time.time(),
            auxiliary_supervision='Smooth L1 camera-frame 3D residual, beta=1m; sum / whole batch size',
            auxiliary_weight=args.weight, auxiliary_images=len(supervision),
            vendor_hashes={p.name: digest(p) for p in vendor.glob('*.py')})
        write_json(root / 'config.json', config)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
            '--nproc_per_node', '1', str(vendor / 'train_ace.py'), str(scene), str(root / 'head.pt')] + config['train_args']
        write_json(root / 'state.json', dict(stage='training', command=command, time=time.time()))
        with (root / 'train.log').open('w') as log:
            subprocess.run(command, cwd=vendor, env=dict(os.environ, GLACE_SEED='2089'),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        for split in ['train', 'test']:
            write_json(root / 'state.json', dict(stage='evaluation', split=split, time=time.time()))
            command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene', '--scene', str(scene),
                '--head', str(root / 'head.pt'), '--vendor', str(vendor), '--deit-checkpoint', str(args.deit_checkpoint),
                '--out', str(root / ('eval_' + split)), '--split', split, '--limit', '64',
                '--valid-mask', str(args.experiment / 'valid_mask.npy')]
            with (root / ('eval_' + split + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(root / 'state.json', dict(stage='complete', time=time.time(), ready_for_fusion=False))
        print('AUXILIARY_COMPLETE ' + str(root), flush=True)
    except Exception:
        import traceback
        write_json(root / 'state.json', dict(stage='failed', time=time.time(), error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
