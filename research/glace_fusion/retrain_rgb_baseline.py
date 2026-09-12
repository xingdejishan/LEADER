import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .rgb_features import rgb_feature_extractor


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            hasher.update(block)
    return hasher.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected vendor source: ' + old[:100])
    return text.replace(old, new, 1)


def patch_vendor(vendor):
    path = vendor / 'ace_trainer.py'
    text = path.read_text()
    text = replace_once(text, 'import os\n', 'import os\nimport json\n')
    text = replace_once(text, 'self.base_seed = 2089 + device_id',
                        'self.base_seed = int(os.environ["GLACE_SEED"]) + device_id')
    start = text.index('        self.training_buffer = {')
    end = text.index('\n        # Features are computed', start)
    text = text[:start] + text[start:end].replace('device=self.device', "device='cpu'") + text[end:]
    text = replace_once(text, 'self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]',
                        'self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k].cpu()')
    text = replace_once(text, '                    buffer_idx = buffer_offset',
                        '                    buffer_idx = buffer_offset\n'
                        '                    if buffer_idx % (1024 * 100) == 0 or buffer_idx == self.options.training_buffer_size:\n'
                        '                        self.write_progress("buffer", buffer_idx, self.options.training_buffer_size)')
    text = replace_once(text, '        self.regressor.train()\n\n    def run_epoch',
                        '        counts = torch.bincount(self.training_buffer["img_idx"], minlength=len(self.dataset))\n'
                        '        if torch.any(counts != self.options.samples_per_image):\n'
                        '            raise RuntimeError("Buffer does not contain the requested samples for every image")\n'
                        '        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size)\n'
                        '        self.regressor.train()\n\n    def run_epoch')
    for key in ['target_px', 'gt_poses_inv', 'intrinsics', 'intrinsics_inv']:
        text = replace_once(text, f"self.training_buffer['{key}'][random_batch_indices].contiguous()",
                            f"self.training_buffer['{key}'][random_batch_indices].to(self.device).contiguous()")
    text = replace_once(text, '        loss /= batch_size',
                        '        loss /= batch_size\n'
                        '        if not torch.isfinite(loss):\n'
                        '            raise FloatingPointError("Non-finite training loss")')
    text = replace_once(text, '        if self.iteration % self.iterations_output == 0:',
                        '        if self.iteration % self.iterations_output == 0 or self.iteration + 1 == self.options.max_iterations:\n'
                        '            self.write_progress("training", self.iteration + 1, self.options.max_iterations,\n'
                        '                                loss=float(loss), optimizer_steps=int(self.optimizer._step_count),\n'
                        '                                valid_fraction=float(valid_mask_b1.float().mean()))')
    text += '\n'
    insert = '''    def write_progress(self, stage, completed, total, **extra):
        path = str(self.options.output_map_file) + '.progress.json'
        data = dict(stage=stage, completed=completed, total=total,
                    time=time.time(), started=self.training_start,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(), **extra)
        with open(path + '.tmp', 'w') as handle:
            json.dump(data, handle, indent=2)
        os.replace(path + '.tmp', path)

'''
    text = replace_once(text, '    def create_training_buffer(self):', insert + '    def create_training_buffer(self):')
    path.write_text(text)
    path = vendor / 'dataset.py'
    text = path.read_text()
    text = replace_once(text, "def _rotate_image(image, angle, order, mode='constant'):",
                        "def _rotate_image(image, angle, order, mode='constant', center=None):")
    text = replace_once(text, 'image = rotate(image, angle, order=order, mode=mode)',
                        'image = rotate(image, angle, order=order, mode=mode, center=center)')
    text = replace_once(text, "image = self._rotate_image(image, angle, 1, 'reflect')",
                        "image = self._rotate_image(image, angle, 1, 'reflect', center=centre_point)")
    text = replace_once(text, "image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')",
                        "image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant', center=centre_point)")
    path.write_text(text)


def prepare(args):
    if args.run_root.exists():
        raise FileExistsError('Refusing to overwrite an existing training run')
    args.run_root.mkdir(parents=True)
    source_scene = args.source_run / 'scene'
    meta = json.loads((source_scene / 'scene_meta.json').read_text())
    if set(meta['splits']['train']['dates']) != {'2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11'}:
        raise ValueError('Unexpected training dates')
    scene = args.run_root / 'scene'
    split = scene / 'train'
    split.mkdir(parents=True)
    for name in ('rgb', 'poses', 'calibration'):
        (split / name).symlink_to(source_scene / 'train' / name, target_is_directory=True)
    paths = sorted(p for p in (split / 'rgb').iterdir() if p.suffix.lower() in ('.jpg', '.png'))
    if len(paths) != 43012 or len(set(p.stem for p in paths)) != len(paths):
        raise ValueError('Unexpected image count or duplicate stems')
    for path in paths:
        pose = np.loadtxt(split / 'poses' / (path.stem + '.txt'))
        K = np.loadtxt(split / 'calibration' / (path.stem + '.txt'))
        if not np.isfinite(pose).all() or not np.isfinite(K).all():
            raise ValueError('Non-finite camera label')
        if not np.allclose(K[0, 0], K[1, 1], rtol=0, atol=1e-6):
            raise ValueError('Rotation augmentation requires square pixels')
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError('Invalid camera pose')
    meta.update(local_image_resolution=480, global_feature_protocol='official_rgb_r2former_480x640')
    write_json(scene / 'scene_meta.json', meta)
    vendor = args.run_root / 'vendor'
    source_vendor = args.source_run / 'vendor_corrected'
    vendor.mkdir()
    for path in source_vendor.glob('*.py'):
        shutil.copyfile(path, vendor / path.name)
    shutil.copytree(source_vendor / 'datasets', vendor / 'datasets', ignore=shutil.ignore_patterns('__pycache__'))
    (vendor / 'ace_encoder_pretrained.pt').symlink_to(source_vendor / 'ace_encoder_pretrained.pt')
    patch_vendor(vendor)
    train_args = ['--training_buffer_size', str(len(paths) * 1024), '--samples_per_image', '1024',
                  '--batch_size', '40960', '--max_iterations', '100000', '--image_resolution', '480',
                  '--use_aug', 'True', '--aug_rotation', '15', '--aug_scale', '1.5',
                  '--feat_noise_std', '0.1', '--num_decoder_clusters', '50', '--head_channels', '768',
                  '--num_head_blocks', '3', '--mlp_ratio', '2', '--repro_loss_soft_clamp', '50']
    config = dict(train_args=train_args, train_images=len(paths), seed=args.seed,
                  global_feature_protocol='official_rgb_r2former_480x640',
                  local_image_resolution=480, source_run=str(args.source_run),
                  source_head_sha256=digest(args.source_run / 'glace_head.pt'),
                  encoder_sha256=digest(vendor / 'ace_encoder_pretrained.pt'),
                  global_backbone_sha256=digest(args.deit_checkpoint),
                  backbone_training=False, training_from_scratch=True,
                  deviations_from_aachen=['one GPU, effective batch 40960 instead of 8 x 40960',
                      'CPU buffer with 43012 x 1024 samples, instead of 16M per GPU',
                      'rotate around calibrated principal point for NCLT',
                      'seed 2089 and FP32 head checkpoint serialization'],
                  vendor_hashes={p.name: digest(p) for p in vendor.glob('*.py')},
                  started=time.time(), ready_for_fusion=False)
    write_json(args.run_root / 'config.json', config)
    return paths, config


def extract_features(args, paths):
    root = args.run_root
    split = root / 'scene/train'
    extract = rgb_feature_extractor(root / 'vendor', args.deit_checkpoint)
    features = np.empty((len(paths), 256), dtype=np.float32)
    started = time.time()
    for start in range(0, len(paths), 16):
        batch = extract(paths[start:start + 16])
        if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid RGB global feature')
        features[start:start + len(batch)] = batch
        if start % 160 == 0 or start + len(batch) == len(paths):
            write_json(root / 'state.json', dict(stage='rgb_features', completed=start + len(batch),
                       total=len(paths), started=started, time=time.time()))
            print(f'RGB features {start + len(batch)}/{len(paths)}', flush=True)
    np.save(split / 'features.npy', features)
    write_json(split / 'features_manifest.json', dict(protocol='official_rgb_r2former_480x640',
               images=[p.name for p in paths], sha256=digest(split / 'features.npy'),
               checkpoint_sha256=digest(args.deit_checkpoint), batch_size=16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2089)
    args = parser.parse_args()
    created = False
    try:
        if args.run_root.exists():
            raise FileExistsError('Run directory already exists')
        created = True
        paths, config = prepare(args)
        extract_features(args, paths)
        import torch
        torch.cuda.empty_cache()
        env = dict(os.environ, GLACE_SEED=str(args.seed))
        preflight = [sys.executable, '-m', 'research.glace_fusion.rgb_preflight', '--run-root', str(args.run_root),
                     '--deit-checkpoint', str(args.deit_checkpoint)]
        subprocess.run(preflight, check=True, env=env)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                   '--nproc_per_node', '1', str(args.run_root / 'vendor/train_ace.py'),
                   str(args.run_root / 'scene'), str(args.run_root / 'glace_head.pt')] + config['train_args']
        write_json(args.run_root / 'state.json', dict(stage='training', time=time.time(), command=command))
        with (args.run_root / 'train.log').open('w') as log:
            subprocess.run(command, cwd=args.run_root / 'vendor', env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(args.run_root / 'state.json', dict(stage='training_complete', time=time.time(),
                   head_sha256=digest(args.run_root / 'glace_head.pt'), ready_for_fusion=False,
                   next_step='training geometry and held-out ranking evaluation'))
    except Exception:
        if created and args.run_root.exists():
            write_json(args.run_root / 'state.json', dict(stage='failed', time=time.time(), error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
