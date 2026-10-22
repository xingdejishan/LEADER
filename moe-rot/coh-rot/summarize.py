"""Consolidate the coh-rot A/B evaluation outputs into one machine-checkable verdict.

Applies the PREREG verdict rules mechanically:

    improves            MOE d(A-B) > 0, CI excludes 0, and |d| > noise floor |d|
    hurts               the same with the sign reversed
    not distinguishable otherwise

Also reports the same for MPE and for the mechanism readout (`field_rotation_deg`, the
quantity L_coh-rot actually penalises).
"""

import argparse
import json
from pathlib import Path


def load(path):
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verdict(effect, ci_low, ci_high, noise_abs):
    if ci_low > 0 and effect > noise_abs:
        return 'IMPROVES'
    if ci_high < 0 and -effect > noise_abs:
        return 'HURTS'
    return 'NOT DISTINGUISHABLE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--comparisons', nargs='+', required=True,
                        help='tag triples: name:referenceTag:treatmentTag[:repeatTag]')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    reports = {}
    for path in sorted(args.eval_dir.glob('pass*-report.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        reports['pass%d-%s' % (payload['pass_index'], payload['label'])] = payload

    result = {'arms': reports, 'comparisons': {}}
    for spec in args.comparisons:
        parts = spec.split(':')
        name, reference, treatment = parts[0], parts[1], parts[2]
        repeat = parts[3] if len(parts) > 3 else None
        compare_path = args.eval_dir / ('compare-%s-vs-%s.json' % (reference, treatment))
        payload = load(compare_path)
        if payload is None:
            print('missing comparison %s' % compare_path, flush=True)
            continue
        noise = None
        if repeat:
            from paired_bootstrap import load as load_frames, compare
            ref_frames = load_frames(args.eval_dir / ('%s-frames.jsonl' % reference))
            rep_frames = load_frames(args.eval_dir / ('%s-frames.jsonl' % repeat))
            shared = sorted(set(ref_frames) & set(rep_frames))
            noise = compare(ref_frames, rep_frames, shared)
        entry = {
            'reference_tag': reference, 'treatment_tag': treatment,
            'reference_report': {k: reports[reference][k] for k in
                                 ('label', 'checkpoint_sha256', 'MPE_m', 'MOE_deg',
                                  'MPE_m_median', 'MOE_deg_median', 'field_rotation_deg')
                                 if k in reports.get(reference, {})},
            'treatment_report': {k: reports[treatment][k] for k in
                                 ('label', 'checkpoint_sha256', 'MPE_m', 'MOE_deg',
                                  'MPE_m_median', 'MOE_deg_median', 'field_rotation_deg')
                                 if k in reports.get(treatment, {})},
            'statistics': payload,
            'noise_floor': noise,
            'verdicts': {},
        }
        for metric in ('MOE_deg', 'MPE_m', 'field_rotation_deg'):
            block = payload['overall'][metric]
            noise_abs = noise[metric]['abs_mean_difference'] if noise else 0.0
            entry['verdicts'][metric] = {
                'mean_difference': block['mean_difference'],
                'ci95': block['ci95'],
                'noise_floor_abs_mean': noise_abs,
                'verdict': verdict(block['mean_difference'], block['ci95'][0],
                                   block['ci95'][1], noise_abs),
            }
        result['comparisons'][name] = entry

    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    print('=== consolidated verdicts ===')
    for name, entry in result['comparisons'].items():
        print('--- %s (%s vs %s) ---' % (name, entry['reference_tag'], entry['treatment_tag']))
        for metric, block in entry['verdicts'].items():
            print('  %-20s d=%+.6f CI=[%+.6f, %+.6f] noise=%.6f  %s'
                  % (metric, block['mean_difference'], block['ci95'][0], block['ci95'][1],
                     block['noise_floor_abs_mean'], block['verdict']))
        for date, block in entry['statistics'].get('per_date', {}).items():
            print('    %s n=%d dMOE=%+.6f CI=[%+.6f, %+.6f] dMPE=%+.6f'
                  % (date, block['frames'], block['MOE_deg']['mean_difference'],
                     block['MOE_deg']['ci95'][0], block['MOE_deg']['ci95'][1],
                     block['MPE_m']['mean_difference']))


if __name__ == '__main__':
    main()
