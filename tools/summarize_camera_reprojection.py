import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_comparison(baseline, refined):
    left = baseline['rows']
    right = refined['rows']
    if [row['scan'] for row in left] != [row['scan'] for row in right]:
        raise ValueError('Evaluation frame order or denominator differs')
    if len(left) != len(right):
        raise ValueError('Evaluation denominators differ')
    if any(a['status'] != 'ok' or b['status'] != 'ok' for a, b in zip(left, right)):
        raise ValueError('Paired metrics require full successful baseline and refinement denominators')
    result = {'frames': len(left), 'baseline_failures': baseline['failed_frames'],
              'refined_failures': refined['failed_frames']}
    paired_values = {}
    for key, mean_key, median_key, p90_key, unit in (
            ('mpe_m', 'all_frame_mpe_mean_m', 'all_frame_mpe_median_m', 'all_frame_mpe_p90_m', 'm'),
            ('moe_deg', 'all_frame_moe_mean_deg', 'all_frame_moe_median_deg', 'all_frame_moe_p90_deg', 'deg')):
        delta = np.asarray([b[key] - a[key] for a, b in zip(left, right)], dtype=np.float64)
        numerical_tolerance = 1e-9
        delta[np.abs(delta) <= numerical_tolerance] = 0.0
        blocks = [delta[start:start + 25] for start in range(0, len(delta), 25)]
        generator = np.random.default_rng(20)
        samples = []
        for _ in range(2000):
            chosen = generator.integers(0, len(blocks), len(blocks))
            samples.append(np.concatenate([blocks[index] for index in chosen]).mean())
        baseline_mean = float(baseline[mean_key])
        refined_mean = float(refined[mean_key])
        result[key] = {
            'unit': unit,
            'baseline_mean': baseline_mean,
            'refined_mean': refined_mean,
            'mean_delta_refined_minus_baseline': float(delta.mean()),
            'relative_mean_change_percent': float(100.0 * (refined_mean - baseline_mean) / baseline_mean),
            'baseline_median': float(baseline[median_key]),
            'refined_median': float(refined[median_key]),
            'baseline_p90': float(baseline[p90_key]),
            'refined_p90': float(refined[p90_key]),
            'numerical_tolerance': numerical_tolerance,
            'improved_frames': int((delta < -numerical_tolerance).sum()),
            'worsened_frames': int((delta > numerical_tolerance).sum()),
            'unchanged_frames': int((np.abs(delta) <= numerical_tolerance).sum()),
            'block25_descriptive_95_interval': np.percentile(samples, [2.5, 97.5]).tolist(),
        }
        paired_values[key] = delta
    translation = paired_values['mpe_m']
    rotation = paired_values['moe_deg']
    result['both_improved'] = int(((translation < 0) & (rotation < 0)).sum())
    result['both_worsened'] = int(((translation > 0) & (rotation > 0)).sum())
    return result


def correspondence_coverage(details):
    rows = details['rows']
    counts = np.asarray([row.get('correspondences', 0) for row in rows], dtype=np.int64)
    selected = np.asarray([row.get('after_grid_cap', 0) for row in rows], dtype=np.int64)
    projected = np.asarray([row.get('after_projection_filter', 0) for row in rows], dtype=np.int64)
    references = np.asarray([len(row.get('references', [])) for row in rows], dtype=np.int64)
    all_nonzero = counts[counts > 0]
    return {
        'frames': len(rows),
        'frames_with_any_selected_2d3d': int((selected > 0).sum()),
        'frames_with_at_least_6_correspondences': int((counts >= 6).sum()),
        'frames_refined': int(sum(bool(row.get('refined')) for row in rows)),
        'frames_fallback': int(sum(row.get('status') == 'fallback' for row in rows)),
        'zero_correspondence_frames': int((counts == 0).sum()),
        'correspondences_total': int(counts.sum()),
        'correspondences_per_frame': {
            'mean': float(counts.mean()) if len(counts) else 0.0,
            'median': float(np.median(counts)) if len(counts) else 0.0,
            'p90': float(np.percentile(counts, 90)) if len(counts) else 0.0,
            'max': int(counts.max()) if len(counts) else 0,
            'nonzero_median': float(np.median(all_nonzero)) if len(all_nonzero) else 0.0,
            'nonzero_p90': float(np.percentile(all_nonzero, 90)) if len(all_nonzero) else 0.0,
        },
        'post_projection_matches_total': int(projected.sum()),
        'post_grid_cap_matches_total': int(selected.sum()),
        'reference_frames_used': {
            'mean_per_query': float(references.mean()) if len(references) else 0.0,
            'median_per_query': float(np.median(references)) if len(references) else 0.0,
            'max_per_query': int(references.max()) if len(references) else 0,
        },
        'fallback_reasons': dict(Counter(row.get('reason') or 'none' for row in rows
                                         if row.get('status') == 'fallback')),
        'runtime_seconds': {
            'total': float(details['elapsed_seconds']),
            'per_frame_mean': float(details['elapsed_seconds'] / len(rows)) if rows else 0.0,
            'frame_p50': float(np.median([row.get('seconds', 0.0) for row in rows])) if rows else 0.0,
            'frame_p90': float(np.percentile([row.get('seconds', 0.0) for row in rows], 90)) if rows else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-evaluation', type=Path, required=True)
    parser.add_argument('--refined-evaluation', type=Path, required=True)
    parser.add_argument('--details', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = read(args.baseline_evaluation)
    refined = read(args.refined_evaluation)
    details = read(args.details)
    if baseline['subset'] != refined['subset'] or baseline['subset'] != details['subset']:
        raise ValueError('Baseline, refined prediction, and diagnostics subsets differ')
    result = {
        'protocol': 'camera_reprojection_paired_comparison_v1',
        'subset': baseline['subset'],
        'baseline_predictions_sha256': baseline['predictions_sha256'],
        'refined_predictions_sha256': refined['predictions_sha256'],
        'training_map_sha256': details['training_map_sha256'],
        'comparison': paired_comparison(baseline, refined),
        'visual_correspondence_coverage': correspondence_coverage(details),
        'interval_method': 'fixed-order non-overlapping 25-frame blocks, resample blocks with replacement, 2000 replicates, seed 20; descriptive only',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
