import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--extra', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    state = load_file(str(args.weights))
    center = json.loads(args.extra.read_text(encoding='utf-8'))['center_t']
    settings = {
        'variant': 'L0_official', 'magic': False, 'max_points': 0,
        'voxel_size': 0.2, 'split_sha256': digest(args.split),
        'source_checkpoint_sha256': digest(args.weights),
        'source_extra_sha256': digest(args.extra),
        'pretraining_overlap_with_local905_val_test': True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': state, 'center_t': center, 'settings': settings}, args.out)
    print(json.dumps({'checkpoint': str(args.out), 'settings': settings}), flush=True)


if __name__ == '__main__':
    main()
