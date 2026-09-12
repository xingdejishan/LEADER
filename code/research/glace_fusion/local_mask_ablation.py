import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from .patch_valid_region import patch_valid_region
from .retrain_rgb_baseline import digest, write_json
from .valid_region import make_valid_mask, resize_valid_mask, sample_valid_region


def read_split(scene, split):
    folder = scene / split
    paths = sorted((folder / 'rgb').iterdir())
    manifest = json.loads((folder / 'features_manifest.json').read_text())
    if manifest['images'] != [p.name for p in paths] or manifest['sha256'] != digest(folder / 'features.npy'):
        raise ValueError('Source feature cache mismatch')
    poses = np.stack([np.loadtxt(folder / 'poses' / (p.stem + '.txt')) for p in paths])
    return paths, poses, np.load(folder / 'features.npy'), manifest


def select_region(train_poses, test_poses, count, radius):
    tree = cKDTree(train_poses[:, :3, 3])
    test_tree = cKDTree(test_poses[:, :3, 3])
    best = None
    for anchor in range(0, len(train_poses), 100):
        def nearby(poses, index):
            indices = np.array(index.query_ball_point(train_poses[anchor, :3, 3], radius), dtype=int)
            traces = np.einsum('ij,nij->n', train_poses[anchor, :3, :3], poses[indices, :3, :3])
            return np.sort(indices[traces > 1 + 2 * np.cos(np.deg2rad(30))])
        train = nearby(train_poses, tree)
        test = nearby(test_poses, test_tree)
        if len(train) < count or len(test) < 16:
            continue
        score = (min(len(test), 64), min(len(train), 4 * count))
        if best is None or score > best[0]:
            best = score, anchor, train, test
    if best is None:
        raise ValueError('No region meets the declared training count and test coverage')
    _, anchor, train, test = best
    train = train[np.linspace(0, len(train) - 1, count, dtype=int)]
    test = test[np.linspace(0, len(test) - 1, min(64, len(test)), dtype=int)]
    return train, test, anchor


def build_split(destination, source_scene, split, data, indices):
    paths, poses, features, manifest = data
    for name in ['rgb', 'poses', 'calibration']:
        (destination / name).mkdir(parents=True, exist_ok=False)
    selected = [paths[i] for i in indices]
    for image in selected:
        (destination / 'rgb' / image.name).symlink_to(image.resolve())
        for name in ['poses', 'calibration']:
            (destination / name / (image.stem + '.txt')).symlink_to(
                (source_scene / split / name / (image.stem + '.txt')).resolve())
    np.save(destination / 'features.npy', features[indices])
    write_json(destination / 'features_manifest.json', dict(manifest,
        images=[p.name for p in selected], sha256=digest(destination / 'features.npy')))
    return {p.stem for p in selected}


def prepare(args):
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    args.run_root.mkdir(parents=True)
    source_scene = args.source_run / 'scene'
    train_data = read_split(source_scene, 'train')
    test_data = read_split(args.test_scene, 'test')
    train_indices, test_indices, anchor = select_region(train_data[1], test_data[1], args.train_images, args.radius)
    scene = args.run_root / 'scene'
    train_stems = build_split(scene / 'train', source_scene, 'train', train_data, train_indices)
    test_stems = build_split(scene / 'test', args.test_scene, 'test', test_data, test_indices)
    if train_stems & test_stems:
        raise ValueError('Training/test image overlap')
    meta = json.loads((source_scene / 'scene_meta.json').read_text())
    test_meta = json.loads((args.test_scene / 'scene_meta.json').read_text())
    for split, original, stems in [('train', meta, train_stems), ('test', test_meta, test_stems)]:
        rows = [r for r in original['splits'][split]['pairs'] if r['image'] in stems]
        meta['splits'][split] = dict(dates=sorted({r['sequence'] for r in rows}), pairs=rows)
    meta['diagnostic_scope'] = 'Local development region chosen by pose coverage, not model scores; not a final test set'
    write_json(scene / 'scene_meta.json', meta)
    calibration = args.calibration
    mask = make_valid_mask(np.load(calibration / 'Cam5_mapu.npy'), np.load(calibration / 'Cam5_mapv.npy'),
                           (1232, 1616), (616, 808))
    np.save(args.run_root / 'valid_mask.npy', mask)
    vendor = args.run_root / 'vendor'
    vendor.mkdir()
    for path in (args.source_run / 'vendor').glob('*.py'):
        shutil.copyfile(path, vendor / path.name)
    shutil.copytree(args.source_run / 'vendor/datasets', vendor / 'datasets', ignore=shutil.ignore_patterns('__pycache__'))
    (vendor / 'ace_encoder_pretrained.pt').symlink_to((args.source_run / 'vendor/ace_encoder_pretrained.pt').resolve())
    patch_valid_region(vendor)
    mask_grid = resize_valid_mask(mask, 480, 630)
    y, x = np.mgrid[:60, :79]
    valid = sample_valid_region(mask_grid, np.column_stack([8 * (x.ravel() + .5), 8 * (y.ravel() + .5)]))
    report = dict(created=time.time(), anchor_image=train_data[0][anchor].stem,
        anchor_camera_center=train_data[1][anchor, :3, 3].tolist(), radius_m=args.radius,
        maximum_orientation_difference_deg=30, train_images=len(train_stems), test_images=len(test_stems),
        valid_grid_fraction=float(valid.mean()), mask_sha256=digest(args.run_root / 'valid_mask.npy'),
        comparison='same local images, seed, head and budget; with vs without physical FOV mask',
        shared_changes_from_stage1=['local scene', '5000-step diagnostic schedule by default', 'batch 8192',
            'sample mask at exact GLACE output pixel centers; conservatively exclude augmentation padding'],
        stage1_checkpoint=str(args.source_run / 'stage1_80k/head.pt'))
    write_json(args.run_root / 'experiment.json', report)
    print(json.dumps(report), flush=True)
    return scene, vendor


def evaluate(args, head, vendor, scene, out, split):
    command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene', '--scene', str(scene),
        '--head', str(head), '--vendor', str(vendor), '--deit-checkpoint', str(args.deit_checkpoint),
        '--out', str(out), '--split', split, '--limit', '64',
        '--valid-mask', str(args.run_root / 'valid_mask.npy')]
    with out.with_suffix('.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--test-scene', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    parser.add_argument('--train-images', type=int, default=256)
    parser.add_argument('--radius', type=float, default=40)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    scene, vendor = prepare(args)
    if args.prepare_only:
        return
    subprocess.run([sys.executable, '-m', 'research.glace_fusion.mask_preflight',
        '--vendor', str(vendor), '--scene', str(scene), '--mask', str(args.run_root / 'valid_mask.npy'),
        '--out', str(args.run_root / 'preflight.json')], check=True)
    source_config = json.loads((args.source_run / 'config.json').read_text())
    try:
        for arm in ['unmasked', 'masked']:
            root = args.run_root / arm
            root.mkdir()
            arm_scene = root / 'scene'
            (arm_scene / 'train').mkdir(parents=True)
            for path in (scene / 'train').iterdir():
                (arm_scene / 'train' / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
            if arm == 'masked':
                (arm_scene / 'train/valid_mask.npy').symlink_to(args.run_root / 'valid_mask.npy')
            (arm_scene / 'test').symlink_to(scene / 'test', target_is_directory=True)
            (arm_scene / 'scene_meta.json').symlink_to(scene / 'scene_meta.json')
            train_args = list(source_config['train_args'])
            for name, value in [('training_buffer_size', args.train_images * 1024),
                                ('batch_size', 8192), ('max_iterations', args.iterations)]:
                train_args[train_args.index('--' + name) + 1] = str(value)
            config = dict(source_config, train_args=train_args, train_images=args.train_images,
                training_from_scratch=True, experiment_arm=arm,
                valid_mask_sha256=digest(args.run_root / 'valid_mask.npy') if arm == 'masked' else None,
                vendor_hashes={p.name: digest(p) for p in vendor.glob('*.py')},
                deviations_from_aachen=['local diagnostic experiment; not an Aachen reproduction'], started=time.time())
            write_json(root / 'config.json', config)
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                '--nproc_per_node', '1', str(vendor / 'train_ace.py'), str(arm_scene), str(root / 'head.pt')] + train_args
            write_json(args.run_root / 'state.json', dict(stage='training', arm=arm, command=command, time=time.time()))
            with (root / 'train.log').open('w') as log:
                subprocess.run(command, cwd=vendor, env=dict(os.environ, GLACE_SEED='2089'),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            for split in ['train', 'test']:
                evaluate(args, root / 'head.pt', vendor, scene, root / ('eval_' + split), split)
        for split in ['train', 'test']:
            evaluate(args, args.source_run / 'stage1_80k/head.pt', args.source_run / 'vendor', scene,
                     args.run_root / ('stage1_eval_' + split), split)
        result = {}
        for arm in ['unmasked', 'masked', 'stage1']:
            result[arm] = {}
            for split in ['train', 'test']:
                folder = args.run_root / arm / ('eval_' + split) if arm != 'stage1' else args.run_root / ('stage1_eval_' + split)
                result[arm][split] = json.loads((folder / 'summary.json').read_text())
        write_json(args.run_root / 'results.json', result)
        write_json(args.run_root / 'state.json', dict(stage='complete', time=time.time(), ready_for_fusion=False))
        print('EXPERIMENT_COMPLETE ' + str(args.run_root), flush=True)
    except Exception:
        import traceback
        write_json(args.run_root / 'state.json', dict(stage='failed', time=time.time(), error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
