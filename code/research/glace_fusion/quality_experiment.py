import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .local_mask_ablation import read_split, build_split
from .retrain_rgb_baseline import digest, write_json


def run(command, log, cwd=None):
    with log.open('w') as output:
        subprocess.run(command, cwd=cwd, env=dict(os.environ, GLACE_SEED='2089',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), stdout=output,
            stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    root.mkdir(exist_ok=False)
    base = Path('/root/rivermind-data')
    previous = base / 'glace_nclt_stage2_local_mask_20260912/lidar_auxiliary'
    source = base / 'glace_nclt_rgb_large_20260912/scene'
    checkpoint = base / 'LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth'
    config = json.loads((previous / 'config.json').read_text())
    meta = json.loads((source / 'scene_meta.json').read_text())
    data = read_split(source, 'train')
    rows = {r['image']: r for r in meta['splits']['train']['pairs']}
    anchor = np.array([-79.050591184, -320.427472972, 7.196297961])
    nearby = np.flatnonzero(np.linalg.norm(data[1][:, :3, 3] - anchor, axis=1) <= 40)
    indices = {name: np.array([i for i in nearby if
        (rows[data[0][i].stem]['sequence'] == '2012-02-18') == validation])
        for name, validation in [('train', False), ('validation', True)]}
    if set(indices['train']) & set(indices['validation']):
        raise ValueError('Split overlap')
    protocol = dict(created=time.time(), source_commit='540a7e0', radius_m=40,
        orientation_filter=None, validation_date='2012-02-18',
        train_count=len(indices['train']), validation_count=len(indices['validation']),
        arms=['uniform', 'depth_balanced'], iterations=10000, seed=2089,
        selection='highest mean validation q10; no test metrics before selection',
        changes_from_previous=['all orientations', 'three dates train, fourth date validation',
            '10000 iterations', 'depth-balanced buffer sampling in second arm only'],
        test='frozen existing 148 frames; previously inspected development test, not untouched holdout',
        comparison='uniform versus balanced uses identical images, labels, seed and budget')
    write_json(root / 'protocol.json', protocol)
    for name, selected in indices.items():
        scene = root / (name + '_scene')
        stems = build_split(scene / 'train', source, 'train', data, selected)
        scene_meta = dict(meta)
        selected_rows = [rows[s] for s in sorted(stems)]
        scene_meta['splits'] = dict(train=dict(dates=sorted({r['sequence'] for r in selected_rows}), pairs=selected_rows))
        write_json(scene / 'scene_meta.json', scene_meta)
        write_json(root / (name + '_stems.json'), sorted(stems))
        (scene / 'train/valid_mask.npy').symlink_to(previous / 'scene/train/valid_mask.npy')
    world_dir = root / 'train_scene/train/lidar_world'
    world_dir.mkdir()
    E = np.asarray(meta['T_BC_camera_to_body'])
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
    labels, missing = [], []
    for i in indices['train']:
        stem = data[0][i].stem
        row = rows[stem]
        scan = base / 'datasets/NCLT' / row['sequence'] / 'velodyne_sync' / (str(row['image_timestamp_us']) + '.bin')
        if not scan.exists():
            missing.append(stem)
            continue
        raw = np.fromfile(scan, dtype=dtype)
        body = np.column_stack([raw[k] for k in ['x', 'y', 'z']]).astype(float) * .005 - 100
        camera = (body - E[:3, 3]) @ E[:3, :3]
        body = body[(camera[:, 2] > 2) & (camera[:, 2] < 80)]
        T = data[1][i] @ np.linalg.inv(E)
        world = body @ T[:3, :3].T + T[:3, 3]
        np.save(world_dir / (stem + '.npy'), world.astype(np.float32))
        labels.append(dict(image=stem, sequence=row['sequence'], scan=str(scan), points=len(world)))
    write_json(root / 'supervision.json', dict(samples=labels, missing=missing,
        validation_depth_used_for_training=False, test_depth_used_for_training=False))
    results = {}
    for arm in protocol['arms']:
        folder = root / arm
        folder.mkdir()
        vendor = folder / 'vendor'
        shutil.copytree(previous / 'vendor', vendor, symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if arm == 'depth_balanced':
            path = vendor / 'ace_trainer.py'
            text = path.read_text()
            needle = '                    sample_idxs = torch.multinomial(image_mask_N1.view(-1),'
            replacement = (
                '                    from depth_sampling import depth_sampling_weights\n'
                '                    image_mask_N1 = depth_sampling_weights(image_mask_N1, batch_data["lidar_valid"])\n'
                + needle)
            if text.count(needle) != 1:
                raise ValueError('Unexpected trainer sampling implementation')
            path.write_text(text.replace(needle, replacement))
            shutil.copyfile(Path(__file__).with_name('depth_sampling.py'), vendor / 'depth_sampling.py')
        train_args = list(config['train_args'])
        for key, value in [('training_buffer_size', len(indices['train']) * 1024), ('max_iterations', 10000)]:
            train_args[train_args.index('--' + key) + 1] = str(value)
        arm_config = dict(config, train_args=train_args, train_images=len(indices['train']),
            experiment_arm=arm, started=time.time(), training_from_scratch=True,
            vendor_hashes={p.name: digest(p) for p in vendor.glob('*.py')},
            validation_date='2012-02-18', ready_for_fusion=False)
        write_json(folder / 'config.json', arm_config)
        write_json(root / 'state.json', dict(stage='training', arm=arm, time=time.time()))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
            '--nproc_per_node', '1', str(vendor / 'train_ace.py'), str(root / 'train_scene'), str(folder / 'head.pt')] + train_args
        run(command, folder / 'train.log', cwd=vendor)
        write_json(root / 'state.json', dict(stage='validation', arm=arm, time=time.time()))
        command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene',
            '--scene', str(root / 'validation_scene'), '--split', 'train', '--limit', '0',
            '--head', str(folder / 'head.pt'), '--vendor', str(vendor),
            '--deit-checkpoint', str(checkpoint), '--valid-mask', str(previous / 'scene/train/valid_mask.npy'),
            '--out', str(folder / 'validation')]
        run(command, folder / 'validation.log')
        results[arm] = json.loads((folder / 'validation/summary.json').read_text())
        write_json(root / 'validation_results.json', results)
    selected = max(results, key=lambda arm: results[arm]['gt_evidence_mean']['q_C'])
    write_json(root / 'selection.json', dict(arm=selected, time=time.time(),
        criterion=protocol['selection'], head_sha256=digest(root / selected / 'head.pt'),
        validation_q10={arm: r['gt_evidence_mean']['q_C'] for arm, r in results.items()}))
    write_json(root / 'state.json', dict(stage='selected', selected=selected, time=time.time()))
    folder = root / selected
    command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene',
        '--scene', str(base / 'glace_nclt_rgb_eval_20260912/scene'),
        '--image-stems', str(base / 'glace_stage3_region_full_20260912/stems.json'),
        '--head', str(folder / 'head.pt'), '--vendor', str(folder / 'vendor'),
        '--deit-checkpoint', str(checkpoint), '--valid-mask', str(previous / 'scene/train/valid_mask.npy'),
        '--out', str(root / 'test')]
    run(command, root / 'test.log')
    write_json(root / 'state.json', dict(stage='complete', selected=selected, time=time.time(), ready_for_fusion=False))


if __name__ == '__main__':
    main()
