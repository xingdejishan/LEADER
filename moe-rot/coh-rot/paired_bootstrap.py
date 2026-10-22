"""Frame-level paired bootstrap for the coh-rot A/B, with an explicit noise floor.

Every comparison is paired frame by frame on `frame_key`.  The statistic is

    d_i = metric(reference arm, frame i) - metric(treatment arm, frame i)

so a positive mean means the treatment arm is better (lower error).

The noise floor is the same statistic computed between two evaluation passes of one and
the same checkpoint: Minkowski's CUDA forward pass is not bit-reproducible, so a rotation
difference of a few tenths of a degree can appear without any training difference.  An
effect smaller than that floor is not evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ('MOE_deg', 'MPE_m', 'field_rotation_deg')


def load(path):
    rows = {}
    with open(path, 'r', encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row['frame_key']] = row
    return rows


def bootstrap_mean(values, draws=20000, seed=20261010):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(reference, treatment, keys, draws=20000, seed=20261010):
    report = {'frames': len(keys)}
    for metric in METRICS:
        difference = np.asarray([reference[k][metric] - treatment[k][metric] for k in keys],
                                dtype=np.float64)
        mean, low, high = bootstrap_mean(difference, draws=draws, seed=seed)
        report[metric] = {
            'mean_difference': mean,
            'ci95': [low, high],
            'ci_excludes_zero': bool(low > 0 or high < 0),
            'reference_mean': float(np.mean([reference[k][metric] for k in keys])),
            'treatment_mean': float(np.mean([treatment[k][metric] for k in keys])),
            'fraction_improved': float(np.mean(difference > 0)),
            'abs_mean_difference': float(np.mean(np.abs(difference))),
            'median_difference': float(np.median(difference)),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True, help='arm A frames.jsonl')
    parser.add_argument('--treatment', type=Path, required=True, help='arm B frames.jsonl')
    parser.add_argument('--reference-repeat', type=Path, default=None,
                        help='second pass of the reference arm: the noise floor')
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    treatment = load(args.treatment)
    keys = sorted(set(reference) & set(treatment))
    missing = (set(reference) ^ set(treatment))
    if missing:
        raise ValueError('frame sets differ: %d keys' % len(missing))
    if not keys:
        raise ValueError('no paired frames')

    dates = {}
    for key in keys:
        dates.setdefault(reference[key]['date'], []).append(key)

    result = {
        'reference_file': str(args.reference),
        'treatment_file': str(args.treatment),
        'overall': compare(reference, treatment, keys, draws=args.draws),
        'per_date': {date: compare(reference, treatment, sorted(group), draws=args.draws)
                     for date, group in sorted(dates.items())},
        'reference_arms': {
            'MOE_deg': float(np.mean([reference[k]['MOE_deg'] for k in keys])),
            'MPE_m': float(np.mean([reference[k]['MPE_m'] for k in keys])),
            'field_rotation_deg': float(np.mean([reference[k]['field_rotation_deg'] for k in keys])),
        },
        'treatment_arms': {
            'MOE_deg': float(np.mean([treatment[k]['MOE_deg'] for k in keys])),
            'MPE_m': float(np.mean([treatment[k]['MPE_m'] for k in keys])),
            'field_rotation_deg': float(np.mean([treatment[k]['field_rotation_deg'] for k in keys])),
        },
    }

    if args.reference_repeat:
        repeat = load(args.reference_repeat)
        shared = sorted(set(reference) & set(repeat))
        floor = compare(reference, repeat, shared, draws=args.draws)
        result['noise_floor'] = {
            'note': 'one identical checkpoint evaluated twice; no training difference exists',
            'reference_file': str(args.reference),
            'repeat_file': str(args.reference_repeat),
            **floor,
        }
        for metric in ('MOE_deg', 'MPE_m'):
            effect = result['overall'][metric]['mean_difference']
            floor_abs = result['noise_floor'][metric]['abs_mean_difference']
            result['overall'][metric]['noise_floor_abs_mean'] = floor_abs
            result['overall'][metric]['effect_vs_noise_floor'] = (
                None if floor_abs == 0 else effect / floor_abs)

    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    print('=== paired frame-level comparison ===')
    print('  frames: %d   dates: %d' % (len(keys), len(dates)))
    for metric in METRICS:
        block = result['overall'][metric]
        print('  %-20s A=%.6f  B=%.6f  d=%+.6f  CI95=[%+.6f, %+.6f]  improved=%.3f'
              % (metric, block['reference_mean'], block['treatment_mean'],
                 block['mean_difference'], block['ci95'][0], block['ci95'][1],
                 block['fraction_improved']))
    if 'noise_floor' in result:
        print('  --- noise floor (same checkpoint, two passes) ---')
        for metric in ('MOE_deg', 'MPE_m'):
            block = result['noise_floor'][metric]
            print('  %-20s |d|=%.6f  d=%+.6f  CI95=[%+.6f, %+.6f]'
                  % (metric, block['abs_mean_difference'], block['mean_difference'],
                     block['ci95'][0], block['ci95'][1]))
    print('  --- per date ---')
    for date, block in result['per_date'].items():
        print('  %s  n=%d  dMoe=%+.6f CI=[%+.6f, %+.6f]  dMpe=%+.6f CI=[%+.6f, %+.6f]'
              % (date, block['frames'],
                 block['MOE_deg']['mean_difference'], block['MOE_deg']['ci95'][0],
                 block['MOE_deg']['ci95'][1],
                 block['MPE_m']['mean_difference'], block['MPE_m']['ci95'][0],
                 block['MPE_m']['ci95'][1]))


if __name__ == '__main__':
    main()
