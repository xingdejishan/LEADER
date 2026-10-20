import argparse
import json
from pathlib import Path

from data.local905_mask import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.split.read_text(encoding='utf-8'))
    if source['protocol'] != 'local905_date_holdout_masked_v2':
        raise ValueError('Expected frozen masked split')
    january = [key for key in source['splits']['train'] if '/2012-01-22/' in key]
    if len(january) != 319 or len(source['splits']['train']) != 552:
        raise ValueError('Unexpected parent training date counts')
    same_domain = january[133:185]
    held_out = set(same_domain)
    split = dict(source)
    split['protocol'] = 'local905_same_domain_holdout_v3'
    split['parent_split_sha256'] = file_sha256(args.split)
    split['same_domain_rule'] = '2012-01-22 ordered synchronized scans [133:185], contiguous 52-frame block'
    split['splits'] = dict(source['splits'])
    split['splits']['train'] = [key for key in source['splits']['train'] if key not in held_out]
    split['splits']['same_domain_val'] = same_domain
    split['counts'] = {key: len(value) for key, value in split['splits'].items()}
    if split['counts'] != {'train': 500, 'val': 40, 'test': 313,
                           'same_domain_val': 52}:
        raise ValueError('Unexpected diagnostic split counts')
    if len(set(sum(split['splits'].values(), []))) != 905:
        raise ValueError('Diagnostic split overlaps or omits scans')
    if args.out.exists():
        if json.loads(args.out.read_text(encoding='utf-8')) != split:
            raise ValueError('Existing diagnostic split differs')
    else:
        args.out.write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps({'counts': split['counts'], 'split_sha256': file_sha256(args.out)}))


if __name__ == '__main__':
    main()
