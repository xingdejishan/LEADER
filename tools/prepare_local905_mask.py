import argparse
import json
from pathlib import Path

from data.local905_mask import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding='utf-8'))
    if split['protocol'] != 'local905_date_holdout_v1':
        raise ValueError('Expected frozen v1 split')
    split['protocol'] = 'local905_date_holdout_masked_v2'
    split['parent_split_sha256'] = file_sha256(args.split)
    split['valid_mask_sha256'] = file_sha256(
        args.data_root / 'train_scene' / 'train' / 'valid_mask.npy')
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding='utf-8'))
        if existing != split:
            raise ValueError('Existing masked split differs')
    else:
        args.out.write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps({'counts': split['counts'], 'split_sha256': file_sha256(args.out),
                      'valid_mask_sha256': split['valid_mask_sha256']}))


if __name__ == '__main__':
    main()
