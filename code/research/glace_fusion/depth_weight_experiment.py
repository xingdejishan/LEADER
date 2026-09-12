import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .correspondence_audit import metrics
from .quality_experiment import run
from .retrain_rgb_baseline import digest, write_json


def audit_validation(folder, scene):
    meta = json.loads((scene / 'scene_meta.json').read_text())
    E = np.asarray(meta['T_BC_camera_to_body'])
    records = []
    for row in meta['splits']['train']['pairs']:
        data = np.load(folder / 'coordinates' / (row['image'] + '.npz'))
        scan = Path('/root/rivermind-data/datasets/NCLT') / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')
        result = metrics(data['xyz'], data['uv'], data['K'], data['GT'], E, scan)
        records.append(dict(image=row['image'], metrics=result))
    q10 = np.mean([r['metrics']['q10'] for r in records])
    precision = np.mean([r['metrics']['precision_3d_1m'] for r in records if 'precision_3d_1m' in r['metrics']])
    return dict(records=records, q10=float(q10), precision_3d_1m=float(precision), selection_score=float(q10 * precision))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source_root, args.out
    root.mkdir(exist_ok=False)
    protocol = dict(created=time.time(), selection='mean validation q10 times mean sparse 3D precision at 1m',
        reason='reprojection alone does not establish 3D correspondence accuracy',
        changed_parameter='3D Smooth L1 weight 1 to 5; other training settings identical',
        validation_date='2012-02-18', test_metric_used_for_selection=False, seed=2089, iterations=10000)
    write_json(root / 'protocol.json', protocol)
    while json.loads((source / 'state.json').read_text())['stage'] != 'complete':
        time.sleep(5)
    baseline = source / 'depth_balanced'
    strong = root / 'strong'
    strong.mkdir()
    shutil.copytree(baseline / 'vendor', strong / 'vendor', symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    trainer = strong / 'vendor/ace_trainer.py'
    text = trainer.read_text()
    if text.count('loss = loss + 1.0 * lidar_loss') != 1:
        raise ValueError('Unexpected auxiliary loss implementation')
    trainer.write_text(text.replace('loss = loss + 1.0 * lidar_loss', 'loss = loss + 5.0 * lidar_loss'))
    config = json.loads((baseline / 'config.json').read_text())
    config.update(auxiliary_weight=5.0, experiment_arm='depth_balanced_weight5', started=time.time(),
        vendor_hashes={p.name: digest(p) for p in (strong / 'vendor').glob('*.py')})
    write_json(strong / 'config.json', config)
    for name in ['train_stems.json', 'validation_stems.json']:
        (root / name).symlink_to(source / name)
    write_json(root / 'state.json', dict(stage='training', time=time.time()))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
        '--nproc_per_node', '1', str(strong / 'vendor/train_ace.py'), str(source / 'train_scene'),
        str(strong / 'head.pt')] + config['train_args']
    run(command, strong / 'train.log', cwd=strong / 'vendor')
    checkpoint = '/root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth'
    mask = '/root/rivermind-data/glace_nclt_stage2_local_mask_20260912/valid_mask.npy'
    common = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene',
        '--head', str(strong / 'head.pt'), '--vendor', str(strong / 'vendor'),
        '--deit-checkpoint', checkpoint, '--valid-mask', mask]
    write_json(root / 'state.json', dict(stage='validation', time=time.time()))
    run(common + ['--scene', str(source / 'validation_scene'), '--split', 'train', '--limit', '0',
        '--out', str(strong / 'validation')], strong / 'validation.log')
    scores = {}
    for label, folder in [('weight1', baseline), ('weight5', strong)]:
        result = audit_validation(folder / 'validation', source / 'validation_scene')
        write_json(root / (label + '_validation_audit.json'), result)
        scores[label] = {k: v for k, v in result.items() if k != 'records'}
    chosen = max(scores, key=lambda label: scores[label]['selection_score'])
    folder = strong if chosen == 'weight5' else baseline
    (root / 'model').symlink_to(folder, target_is_directory=True)
    write_json(root / 'selection.json', dict(arm='model', chosen=chosen, time=time.time(),
        head_sha256=digest(folder / 'head.pt'), criterion=protocol['selection'], validation=scores))
    if chosen == 'weight5':
        run(common + ['--scene', '/root/rivermind-data/glace_nclt_rgb_eval_20260912/scene',
            '--image-stems', '/root/rivermind-data/glace_stage3_region_full_20260912/stems.json',
            '--out', str(root / 'test')], root / 'test.log')
    else:
        (root / 'test').symlink_to(source / 'test', target_is_directory=True)
    write_json(root / 'state.json', dict(stage='complete', chosen=chosen, time=time.time()))


if __name__ == '__main__':
    main()
