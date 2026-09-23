import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair_rows(first, second):
    left = first['rows']
    right = second['rows']
    if [row['scan'] for row in left] != [row['scan'] for row in right]:
        raise ValueError('Different frame order or denominator')
    return [(a, b) for a, b in zip(left, right)
            if a['status'] == 'ok' and b['status'] == 'ok']


def compare(first, second):
    paired = pair_rows(first, second)
    result = {'paired_successes': len(paired),
              'total_frames': len(first['rows']),
              'first_failures': first['failed_frames'],
              'second_failures': second['failed_frames']}
    if not paired:
        return result
    for key in ('mpe_m', 'moe_deg'):
        delta = np.asarray([a[key] - b[key] for a, b in paired])
        blocks = [delta[start:start + 25] for start in range(0, len(delta), 25)]
        generator = np.random.default_rng(20)
        samples = []
        for _ in range(2000):
            chosen = generator.integers(0, len(blocks), len(blocks))
            samples.append(np.concatenate([blocks[index] for index in chosen]).mean())
        result[key] = {
            'mean_delta': float(delta.mean()),
            'improved': int((delta < 0).sum()),
            'worsened': int((delta > 0).sum()),
            'block25_descriptive_interval': np.percentile(samples, [2.5, 97.5]).tolist(),
        }
    delta_t = np.asarray([a['mpe_m'] - b['mpe_m'] for a, b in paired])
    delta_q = np.asarray([a['moe_deg'] - b['moe_deg'] for a, b in paired])
    result['both_improved'] = int(((delta_t < 0) & (delta_q < 0)).sum())
    result['both_worsened'] = int(((delta_t > 0) & (delta_q > 0)).sum())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--magic', required=True)
    parser.add_argument('--lidar', required=True)
    parser.add_argument('--shuffled')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    magic = read(args.magic)
    lidar = read(args.lidar)
    if magic['subset'] != 'test' or lidar['subset'] != 'test':
        raise ValueError('Final comparison requires the fixed test subset')
    report = {'magic_vs_lidar': compare(magic, lidar)}
    if args.shuffled:
        shuffled = read(args.shuffled)
        if shuffled['subset'] != 'test':
            raise ValueError('Shuffle control requires the fixed test subset')
        report['magic_vs_shuffled'] = compare(magic, shuffled)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
