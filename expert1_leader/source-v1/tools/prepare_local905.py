import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_root, output):
    scene = data_root / 'train_scene'
    metadata_path = scene / 'scene_meta.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    camera_to_body = np.asarray(metadata['T_BC_camera_to_body'], dtype=np.float64)
    body_to_camera = np.linalg.inv(camera_to_body)
    records = {}
    dates = {}
    for pair in metadata['splits']['train']['pairs']:
        stem = pair['image']
        date = pair['sequence']
        scan = data_root / 'scans' / date / 'velodyne_sync' / (stem + '.bin')
        if not scan.is_file():
            continue
        image = scene / 'train' / 'rgb' / (stem + '.jpg')
        calibration = scene / 'train' / 'calibration' / (stem + '.txt')
        pose = scene / 'train' / 'poses' / (stem + '.txt')
        for source in (image, calibration, pose):
            if not source.is_file():
                raise FileNotFoundError(source)
        intrinsic = np.loadtxt(calibration)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f'Invalid K: {stem}')
        key = scan.relative_to(data_root).as_posix()
        records[key] = {
            'image': os.path.relpath(image, output).replace('\\', '/'),
            'K': intrinsic.tolist(),
            'T_camera_lidar': body_to_camera.tolist(),
        }
        dates.setdefault(date, []).append(key)
    for keys in dates.values():
        keys.sort()
    if {date: len(keys) for date, keys in dates.items()} != {
            '2012-01-22': 319, '2012-02-02': 273, '2012-05-11': 313}:
        raise ValueError(f'Unexpected scan coverage: {[(date, len(keys)) for date, keys in dates.items()]}')
    splits = {
        'train': dates['2012-01-22'] + dates['2012-02-02'][:-40],
        'val': dates['2012-02-02'][-40:],
        'test': dates['2012-05-11'],
    }
    if len(set(sum(splits.values(), []))) != 905:
        raise ValueError('Split does not cover exactly 905 unique scans')
    output.mkdir(parents=True, exist_ok=True)
    raw_manifest = {'frames': dict(sorted(records.items()))}
    split_manifest = {
        'protocol': 'local905_date_holdout_v1',
        'source_scene_meta_sha256': sha256(metadata_path),
        'data_root': str(data_root),
        'splits': splits,
        'counts': {key: len(value) for key, value in splits.items()},
        'test_date': '2012-05-11',
        'validation_rule': 'last 40 synchronized scans of 2012-02-02',
    }
    for name, data in (('raw_manifest.json', raw_manifest), ('split.json', split_manifest)):
        path = output / name
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old != data:
                raise ValueError(f'Existing frozen file differs: {path}')
        else:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': split_manifest['counts'],
                      'raw_manifest_sha256': sha256(output / 'raw_manifest.json'),
                      'split_sha256': sha256(output / 'split.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.out.resolve())
