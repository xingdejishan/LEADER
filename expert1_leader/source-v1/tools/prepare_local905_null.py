import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def camera_key(record):
    value = np.asarray(record['T_camera_lidar'], dtype=np.float32).astype(np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    split = json.loads(args.split.read_text(encoding='utf-8'))
    manifest = json.loads(args.sam_manifest.read_text(encoding='utf-8'))
    train_keys = split['splits']['train']
    if len(train_keys) != 552:
        raise ValueError('Expected exactly 552 train-only frames')
    sums = {}
    counts = {}
    for key in train_keys:
        record = manifest['frames'][key]
        camera = camera_key(record)
        array = np.load(args.sam_manifest.parent / record['sam_features'], allow_pickle=False)
        if array.shape != (256, 64, 64) or not np.isfinite(array).all():
            raise ValueError(f'Invalid training SAM feature: {key}')
        if camera not in sums:
            sums[camera] = np.zeros(array.shape, dtype=np.float64)
            counts[camera] = 0
        sums[camera] += array
        counts[camera] += 1
    if sum(counts.values()) != len(train_keys):
        raise ValueError('Null template used the wrong train denominator')
    templates = {camera: (value / counts[camera]).astype(np.float32)
                 for camera, value in sums.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **templates)
    report = {
        'protocol': 'train_only_per_camera_sam_mean_v1',
        'split_sha256': digest(args.split),
        'sam_manifest_sha256': digest(args.sam_manifest),
        'train_frame_count': len(train_keys),
        'camera_counts': counts,
        'template_sha256': digest(args.out),
    }
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
