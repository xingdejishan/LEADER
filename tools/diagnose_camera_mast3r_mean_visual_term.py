import argparse
import json
import time
from pathlib import Path

import numpy as np

import camera_reprojection as camera
from camera_mast3r_anchored import ONLINE_PROTOCOL, load_json, sha256_file, write_json


MEAN_PROTOCOL = 'camera_mast3r_anchored_mean_visual_diagnostic_v1'


def frozen_predictions(path):
    path = Path(path).resolve()
    expected = path.with_suffix('.sha256').read_text(encoding='ascii').strip()
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f'Prediction SHA256 sidecar mismatch: {path}')
    return load_json(path), observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--baseline-predictions', type=Path, required=True)
    parser.add_argument('--primary-predictions', type=Path, required=True)
    parser.add_argument('--details', type=Path, required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_sha = frozen_predictions(args.baseline_predictions)
    primary, primary_sha = frozen_predictions(args.primary_predictions)
    details = load_json(args.details)
    details_sha = sha256_file(args.details)
    raw_manifest_sha = sha256_file(args.raw_manifest)
    if baseline.get('subset') != 'test' or primary.get('protocol') != ONLINE_PROTOCOL:
        raise ValueError('Expected frozen test baseline and primary MASt3R predictions')
    if details.get('subset') != 'test' or details.get('refined_predictions_sha256') != primary_sha:
        raise ValueError('Details do not belong to the frozen primary predictions')
    if details.get('frozen_relation_predictions_sha256') != baseline_sha:
        raise ValueError('Details do not belong to the supplied frozen baseline')
    if details.get('raw_manifest_sha256') != raw_manifest_sha:
        raise ValueError('Raw manifest SHA256 differs from the frozen run')
    split_sha = sha256_file(args.split)
    split = load_json(args.split)
    keys = split['splits']['test']
    baseline_rows = baseline['predictions']
    primary_rows = primary['predictions']
    detail_rows = details['rows']
    if any([row['scan'] for row in rows] != keys for rows in (baseline_rows, primary_rows, detail_rows)):
        raise ValueError('Prediction and details scan order differs from the frozen split')
    if baseline.get('split_sha256') != split_sha or primary.get('split_sha256') != split_sha:
        raise ValueError('Frozen split SHA256 mismatch')
    raw_manifest = load_json(args.raw_manifest)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)

    started = time.perf_counter()
    diagnostic_predictions = []
    diagnostic_rows = []
    for baseline_row, detail in zip(baseline_rows, detail_rows):
        if baseline_row.get('status') != 'ok':
            raise ValueError(f'Frozen baseline prediction failed: {baseline_row["scan"]}')
        pose = np.asarray(baseline_row['T_world_body'], dtype=np.float64)
        correspondences = detail.get('selected_correspondences', [])
        world_points = np.asarray([row['world_xyz'] for row in correspondences], dtype=np.float64).reshape(-1, 3)
        query_uv = np.asarray([row['query_uv'] for row in correspondences], dtype=np.float64).reshape(-1, 2)
        calibration = raw_manifest['frames'][detail['scan']]
        refined_pose, refinement = camera.refine_pose(
            pose,
            np.asarray(calibration['T_camera_lidar'], dtype=np.float64),
            np.asarray(calibration['K'], dtype=np.float64),
            world_points,
            query_uv,
        )
        diagnostic_predictions.append({
            'scan': detail['scan'],
            'status': 'ok',
            'T_world_body': refined_pose.tolist(),
            'correspondences': len(correspondences),
        })
        diagnostic_rows.append({
            'scan': detail['scan'],
            'status': refinement['status'],
            'reason': refinement.get('reason'),
            'correspondences': int(refinement.get('correspondences', len(correspondences))),
            'objective_before': refinement.get('objective_before'),
            'objective_after': refinement.get('objective_after'),
            'iterations': refinement.get('iterations'),
            'normalized_delta': refinement.get('normalized_delta'),
            'median_reprojection_before_px': refinement.get('median_reprojection_before_px'),
            'median_reprojection_after_px': refinement.get('median_reprojection_after_px'),
        })
    elapsed = time.perf_counter() - started
    prediction = {
        'protocol': MEAN_PROTOCOL,
        'subset': 'test',
        'expected_frames': len(keys),
        'split_sha256': split_sha,
        'parent_relation_predictions_sha256': baseline_sha,
        'primary_sum_predictions_sha256': primary_sha,
        'primary_details_sha256': details_sha,
        'raw_manifest_sha256': raw_manifest_sha,
        'training_map_sha256': details['training_map_sha256'],
        'elapsed_seconds': None,
        'diagnostic_optimizer_seconds': elapsed,
        'correspondence_source': 'exact selected_correspondences from frozen primary details; no rematching or MASt3R inference',
        'objective': 'mean(Huber(norm(pixel_residual)/4px)) + ||delta_t/0.1m||^2 + ||delta_w/1deg||^2',
        'query_ground_truth_access': 'none in this diagnostic runner',
        'predictions': diagnostic_predictions,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / 'predictions.json'
    write_json(prediction_path, prediction)
    prediction_sha = sha256_file(prediction_path)
    prediction_path.with_suffix('.sha256').write_text(prediction_sha + '\n', encoding='ascii')
    diagnostic_details = {
        'protocol': MEAN_PROTOCOL,
        'subset': 'test',
        'prediction_sha256': prediction_sha,
        'baseline_predictions_sha256': baseline_sha,
        'primary_sum_predictions_sha256': primary_sha,
        'primary_details_sha256': details_sha,
        'raw_manifest_sha256': raw_manifest_sha,
        'correspondence_source': prediction['correspondence_source'],
        'optimizer': 'camera_reprojection.refine_pose; fixed L-BFGS-B bounds/iterations/fallback; mean visual term',
        'diagnostic_optimizer_seconds': elapsed,
        'query_ground_truth_access': 'none in this diagnostic runner',
        'rows': diagnostic_rows,
    }
    write_json(args.out_dir / 'details.json', diagnostic_details)
    print(json.dumps({
        'subset': 'test',
        'prediction_sha256': prediction_sha,
        'frames': len(diagnostic_predictions),
        'refined_frames': sum(row['status'] == 'refined' for row in diagnostic_rows),
        'fallback_frames': sum(row['status'] == 'fallback' for row in diagnostic_rows),
        'fallback_reasons': {
            reason: sum(row['reason'] == reason for row in diagnostic_rows)
            for reason in sorted({row['reason'] for row in diagnostic_rows if row['reason']})
        },
        'selected_correspondences': sum(row['correspondences'] for row in diagnostic_rows),
        'diagnostic_optimizer_seconds': elapsed,
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
