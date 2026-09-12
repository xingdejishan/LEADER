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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--vendor', required=True, type=Path)
    parser.add_argument('--deit_checkpoint', required=True, type=Path)
    parser.add_argument('--dataset_folder', required=True, type=Path)
    parser.add_argument('--camera_root', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'state.json').exists():
        raise SystemExit('Run directory already contains a run')
    state = {'started_at': time.time(), 'stage': 'prepare', 'pid': os.getpid()}

    def update(stage, **values):
        state.update(stage=stage, updated_at=time.time(), **values)
        temporary = out / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(out / 'state.json')
        print(json.dumps(state), flush=True)

    try:
        import torch
        from .glace_adapter import deit_global_feature_fn, GLACEAdapter
        from .nclt_camera import preprocess_image, TRAIN_DATES
        repo = Path(__file__).resolve().parents[2]
        snapshot = out / 'code'
        shutil.copytree(Path(__file__).parent, snapshot / 'research/glace_fusion',
                        ignore=shutil.ignore_patterns('__pycache__'))
        vendor = out / 'vendor'
        vendor.mkdir()
        for source in args.vendor.glob('*.py'):
            shutil.copyfile(source, vendor / source.name)
        shutil.copytree(args.vendor / 'datasets', vendor / 'datasets',
                        ignore=shutil.ignore_patterns('__pycache__'))
        (vendor / 'ace_encoder_pretrained.pt').symlink_to((args.vendor / 'ace_encoder_pretrained.pt').resolve())
        hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for directory in [snapshot, vendor] for p in directory.rglob('*.py')}
        hashes['deit_checkpoint'] = hashlib.sha256(args.deit_checkpoint.read_bytes()).hexdigest()
        hashes['ace_encoder'] = hashlib.sha256((vendor / 'ace_encoder_pretrained.pt').read_bytes()).hexdigest()
        (out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
        scene = out / 'scene'
        command = [sys.executable, '-m', 'research.glace_fusion.make_glace_scene',
                   '--dataset_folder', str(args.dataset_folder), '--camera_root', str(args.camera_root),
                   '--out', str(scene), '--test_dates']
        update('scene')
        with (out / 'scene.log').open('w') as log:
            subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, check=True)
        images = sorted((scene / 'train/rgb').iterdir())
        metadata = json.loads((scene / 'scene_meta.json').read_text())
        if not images or set(metadata['splits']['train']['dates']) != set(TRAIN_DATES):
            raise ValueError('Unexpected training split')
        torch.manual_seed(2089)
        torch.set_num_threads(4)
        update('global_features', training_images=len(images), completed_images=0)
        feature_fn = deit_global_feature_fn(vendor, args.deit_checkpoint)
        features = np.lib.format.open_memmap(scene / 'train/features.npy', mode='w+',
                                             dtype=np.float32, shape=(len(images), 256))
        for begin in range(0, len(images), 16):
            batch = [preprocess_image(path, np.eye(3), 616)[0] for path in images[begin:begin + 16]]
            result = feature_fn.batch(batch)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite global features')
            features[begin:begin + len(batch)] = result
            if begin % 256 == 0:
                features.flush()
                update('global_features', completed_images=begin + len(batch))
        features.flush()
        del features, feature_fn
        torch.cuda.empty_cache()
        buffer_size = len(images) * 128
        train_args = ['--training_buffer_size', str(buffer_size), '--samples_per_image', '128',
                      '--batch_size', '8192', '--max_iterations', '30000',
                      '--image_resolution', '616', '--aug_rotation', '0', '--aug_scale', '1',
                      '--feat_noise_std', '0.1', '--num_decoder_clusters', '1', '--head_channels', '768']
        config = {'seed': 2089, 'train_dates': list(TRAIN_DATES), 'test_images_used': 0,
                  'training_images': len(images), 'train_args': train_args,
                  'local_image_resolution': 616, 'global_image_size_hw': [480, 640],
                  'global_preprocessing': 'same grayscale-replicated adapter as inference',
                  'encoder_frozen': True, 'deit_frozen': True,
                  'deit_checkpoint': str(args.deit_checkpoint), 'vendor_dir': str(vendor),
                  'head_path': str(out / 'glace_nclt_head.pt')}
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        update('training', completed_images=len(images), training_buffer_size=buffer_size)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                   '--nproc_per_node', '1', str(vendor / 'train_ace.py'), str(scene),
                   str(out / 'glace_nclt_head.pt')] + train_args
        with (out / 'train.log').open('w') as log:
            subprocess.run(command, cwd=vendor, stdout=log, stderr=subprocess.STDOUT, check=True)
        update('verify_head')
        feature_fn = deit_global_feature_fn(vendor, args.deit_checkpoint)
        adapter = GLACEAdapter(vendor, out / 'glace_nclt_head.pt', global_feature_fn=feature_fn,
                               T_BC=np.asarray(metadata['T_BC_camera_to_body']))
        path = images[len(images) // 2]
        K = np.loadtxt(scene / 'train/calibration' / (path.stem + '.txt'))
        image, K = preprocess_image(path, K, 616)
        result = adapter.infer(image, K)
        if not np.isfinite(result.xyz_world).all():
            raise ValueError('Trained head produced nonfinite coordinates')
        update('complete', head=str(out / 'glace_nclt_head.pt'), finished_at=time.time(),
               head_sha256=hashlib.sha256((out / 'glace_nclt_head.pt').read_bytes()).hexdigest(),
               smoke_test={'finite_coordinates': True, 'camera_inliers': result.inlier_count,
                           'scope': 'training-image inference only, not test accuracy'})
    except Exception as exc:
        update('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
