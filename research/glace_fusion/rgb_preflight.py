import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .nclt_camera import preprocess_image
from .retrain_rgb_baseline import write_json
from .rgb_features import CachedRGBFeatures, rgb_feature_extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    vendor = root / 'vendor'
    sys.path.insert(0, str(vendor))
    import torch
    from dataset import CamLocDataset
    from skimage.transform import rotate

    dataset = CamLocDataset(root / 'scene/train', mode=0, use_half=False,
                           image_height=480, augment=False)
    cached = CachedRGBFeatures(root / 'scene/train')
    selected = np.linspace(0, len(dataset) - 1, 64, dtype=int)
    local_error = 0.
    K_error = 0.
    for index in selected[::4]:
        image, _, pose, inv_pose, K, _, _, path, feature, idx = dataset[int(index)]
        stem = Path(path).stem
        raw_K = np.loadtxt(root / 'scene/train/calibration' / (stem + '.txt'))
        gray, actual_K = preprocess_image(path, raw_K, 480)
        expected = (gray - .4) / .25
        if tuple(image.shape) != (1,) + expected.shape:
            raise AssertionError('Local raster does not match the GLACE loader')
        local_error = max(local_error, float(np.max(np.abs(image.numpy()[0] - expected))))
        K_error = max(K_error, float(np.max(np.abs(K.numpy() - actual_K))))
        np.testing.assert_array_equal(feature.numpy(), cached[stem])
        np.testing.assert_allclose(pose.numpy() @ inv_pose.numpy(), np.eye(4), atol=1e-4)
    if local_error > 1e-5 or K_error > 5e-5:
        raise AssertionError(f'Preprocessing differs: image={local_error}, K={K_error}')

    K = np.array([[155.6, 0, 321.8], [0, 155.6, 242.2], [0, 0, 1.]])
    points = np.array([[20., 30.], [290., 70.], [510., 400.]])
    angle = np.deg2rad(15.)
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(K).T
    projected = rays @ R @ K.T
    projected = projected[:, :2] / projected[:, 2:]
    expected = (points - K[:2, 2]) @ R[:2, :2] + K[:2, 2]
    np.testing.assert_allclose(projected, expected, atol=1e-10)
    grid_y, grid_x = np.mgrid[:480, :630]
    ramp = np.stack([grid_x, grid_y]).astype(np.float32)
    rotated = dataset._rotate_image(torch.from_numpy(ramp), 15., 1, center=K[:2, 2]).numpy()
    target = np.array([310., 230.])
    source = (target - K[:2, 2]) @ R[:2, :2].T + K[:2, 2]
    np.testing.assert_allclose(rotated[:, 230, 310], source, atol=1e-3)

    extract = rgb_feature_extractor(vendor, args.deit_checkpoint)
    paths = [Path(dataset.rgb_files[int(i)]) for i in selected[:16]]
    online = extract(paths)
    cache = np.stack([cached[p.stem] for p in paths])
    rgb_error = float(np.max(np.abs(online - cache)))
    if rgb_error > 2e-5:
        raise AssertionError(f'RGB feature paths differ: {rgb_error}')
    del extract
    torch.cuda.empty_cache()

    smoke = root / 'preflight'
    split = smoke / 'scene/train'
    for name in ('rgb', 'calibration', 'poses'):
        (split / name).mkdir(parents=True, exist_ok=False)
    all_paths = [Path(dataset.rgb_files[int(i)]) for i in selected]
    for path in all_paths:
        for name, filename in [('rgb', path.name), ('calibration', path.stem + '.txt'), ('poses', path.stem + '.txt')]:
            (split / name / filename).symlink_to(root / 'scene/train' / name / filename)
    np.save(split / 'features.npy', np.stack([cached[p.stem] for p in all_paths]))
    config = json.loads((root / 'config.json').read_text())
    options = list(config['train_args'])
    for key, value in [('--training_buffer_size', str(len(selected) * 1024)), ('--max_iterations', '20')]:
        options[options.index(key) + 1] = value
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
               '--nproc_per_node', '1', str(vendor / 'train_ace.py'), str(smoke / 'scene'), str(smoke / 'head.pt')] + options
    with (smoke / 'train.log').open('w') as log:
        subprocess.run(command, cwd=vendor, env=dict(os.environ, GLACE_SEED=str(config['seed'])),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    progress = json.loads((smoke / 'head.pt.progress.json').read_text())
    if progress['optimizer_steps'] <= 1 or progress['completed'] != 20 or not np.isfinite(progress['loss']):
        raise AssertionError('Training smoke test did not perform finite optimizer updates')
    write_json(root / 'preflight.json', dict(passed=True, local_image_max_abs_error=local_error,
               intrinsics_max_abs_error=K_error, rgb_global_feature_max_abs_error=rgb_error,
               rotation_projection_verified=True, train_images=len(dataset),
               smoke_training=progress, scope='Geometry, feature contract, finite updates and full batch GPU memory; not model quality'))
    print('Preflight passed', flush=True)


if __name__ == '__main__':
    main()
