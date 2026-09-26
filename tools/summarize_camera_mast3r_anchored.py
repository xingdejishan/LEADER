import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_comparison(baseline, candidate):
    left = baseline['rows']
    right = candidate['rows']
    if len(left) != len(right) or [row['scan'] for row in left] != [row['scan'] for row in right]:
        raise ValueError('Baseline and candidate evaluation frame order differs')
    if any(row['status'] != 'ok' for row in left + right):
        raise ValueError('Paired comparison requires both complete successful denominators')
    result = {'frames': len(left)}
    deltas = {}
    for key, mean_key, median_key, p90_key, unit in (
            ('mpe_m', 'all_frame_mpe_mean_m', 'all_frame_mpe_median_m', 'all_frame_mpe_p90_m', 'm'),
            ('moe_deg', 'all_frame_moe_mean_deg', 'all_frame_moe_median_deg', 'all_frame_moe_p90_deg', 'deg')):
        delta = np.asarray([b[key] - a[key] for a, b in zip(left, right)], dtype=np.float64)
        delta[np.abs(delta) <= 1e-9] = 0.0
        blocks = [delta[start:start + 25] for start in range(0, len(delta), 25)]
        rng = np.random.default_rng(20)
        samples = np.asarray([
            np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]).mean()
            for _ in range(2000)
        ])
        baseline_mean = float(baseline[mean_key])
        candidate_mean = float(candidate[mean_key])
        result[key] = {
            'unit': unit,
            'baseline_mean': baseline_mean,
            'candidate_mean': candidate_mean,
            'mean_delta_candidate_minus_baseline': float(delta.mean()),
            'relative_mean_change_percent': float(100.0 * (candidate_mean - baseline_mean) / baseline_mean),
            'baseline_median': float(baseline[median_key]),
            'candidate_median': float(candidate[median_key]),
            'baseline_p90': float(baseline[p90_key]),
            'candidate_p90': float(candidate[p90_key]),
            'improved_frames': int((delta < -1e-9).sum()),
            'worsened_frames': int((delta > 1e-9).sum()),
            'unchanged_frames': int((np.abs(delta) <= 1e-9).sum()),
            'block25_descriptive_95_interval': np.percentile(samples, [2.5, 97.5]).tolist(),
        }
        deltas[key] = delta
    translation = deltas['mpe_m']
    rotation = deltas['moe_deg']
    result['both_improved'] = int(((translation < 0) & (rotation < 0)).sum())
    result['both_worsened'] = int(((translation > 0) & (rotation > 0)).sum())
    return result


def correspondence_coverage(details):
    rows = details['rows']
    selected = np.asarray([row.get('correspondences', 0) for row in rows], dtype=np.int64)
    grid_selected = np.asarray([row.get('after_grid_and_total_cap', 0) for row in rows], dtype=np.int64)
    stage_keys = ('raw_candidates', 'after_geometry', 'after_unique_query_pixel_and_landmark',
                  'after_grid_and_total_cap')
    stage_totals = {key: int(sum(row.get(key, 0) for row in rows)) for key in stage_keys}
    reference_summaries = [ref for row in rows for ref in row.get('references', [])]
    frame_seconds = np.asarray([row.get('seconds', 0.0) for row in rows], dtype=np.float64)
    reference_seconds = np.asarray([ref.get('pair_elapsed_seconds', 0.0) for ref in reference_summaries], dtype=np.float64)
    return {
        'frames': len(rows),
        'frames_with_at_least_6_correspondences': int((selected >= 6).sum()),
        'frames_refined': int(sum(bool(row.get('refined')) for row in rows)),
        'frames_fallback': int(sum(row.get('status') == 'fallback' for row in rows)),
        'fallback_reasons': dict(Counter(row.get('reason') or 'none' for row in rows
                                         if row.get('status') == 'fallback')),
        'correspondence_stages_total': stage_totals,
        'selected_correspondences_per_frame': {
            'mean': float(selected.mean()) if len(selected) else 0.0,
            'median': float(np.median(selected)) if len(selected) else 0.0,
            'p90': float(np.percentile(selected, 90)) if len(selected) else 0.0,
            'max': int(selected.max()) if len(selected) else 0,
            'total': int(selected.sum()),
        },
        'grid_selected_per_frame': {
            'mean': float(grid_selected.mean()) if len(grid_selected) else 0.0,
            'median': float(np.median(grid_selected)) if len(grid_selected) else 0.0,
            'p90': float(np.percentile(grid_selected, 90)) if len(grid_selected) else 0.0,
            'total': int(grid_selected.sum()),
        },
        'match_counts_total': {
            'coarse_matches': int(sum(ref.get('coarse_matches', 0) for ref in reference_summaries)),
            'raw_fine_matches': int(sum(ref.get('raw_matches', 0) for ref in reference_summaries)),
            'confidence_kept_matches': int(sum(ref.get('confidence_kept', 0) for ref in reference_summaries)),
            'exact_anchor_matches': int(sum(ref.get('anchor_identity_kept', 0) for ref in reference_summaries)),
            'identity_mismatch': int(sum(ref.get('identity_mismatch', 0) for ref in reference_summaries)),
        },
        'references_per_query': {
            'mean': float(np.mean([len(row.get('references', [])) for row in rows])) if rows else 0.0,
            'median': float(np.median([len(row.get('references', [])) for row in rows])) if rows else 0.0,
            'max': int(max((len(row.get('references', [])) for row in rows), default=0)),
        },
        'runtime_seconds': {
            'total': float(details['elapsed_seconds']),
            'frame_mean': float(frame_seconds.mean()) if len(frame_seconds) else 0.0,
            'frame_p50': float(np.median(frame_seconds)) if len(frame_seconds) else 0.0,
            'frame_p90': float(np.percentile(frame_seconds, 90)) if len(frame_seconds) else 0.0,
            'reference_pairs': int(len(reference_seconds)),
            'reference_pair_mean': float(reference_seconds.mean()) if len(reference_seconds) else 0.0,
            'reference_pair_p50': float(np.median(reference_seconds)) if len(reference_seconds) else 0.0,
            'reference_pair_p90': float(np.percentile(reference_seconds, 90)) if len(reference_seconds) else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-evaluation', type=Path, required=True)
    parser.add_argument('--candidate-evaluation', type=Path, required=True)
    parser.add_argument('--details', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = read(args.baseline_evaluation)
    candidate = read(args.candidate_evaluation)
    details = read(args.details)
    if baseline['subset'] != candidate['subset'] or baseline['subset'] != details['subset']:
        raise ValueError('Evaluation and detail subsets differ')
    result = {
        'protocol': 'camera_mast3r_anchored_paired_comparison_v1',
        'subset': baseline['subset'],
        'baseline_predictions_sha256': baseline['predictions_sha256'],
        'candidate_predictions_sha256': candidate['predictions_sha256'],
        'training_map_sha256': details['training_map_sha256'],
        'comparison': paired_comparison(baseline, candidate),
        'coverage': correspondence_coverage(details),
        'interval_method': 'fixed-order nonoverlapping 25-frame blocks; resample blocks with replacement; 2000 replicates; seed 20; descriptive only',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
