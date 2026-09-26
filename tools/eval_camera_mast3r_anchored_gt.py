import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_frozen_predictions(path, protocols):
    path = Path(path).resolve()
    sidecar = path.with_suffix('.sha256')
    expected = sidecar.read_text(encoding='ascii').strip()
    observed = digest(path)
    if observed != expected:
        raise ValueError(f'Prediction SHA256 sidecar mismatch: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('protocol') not in protocols:
        raise ValueError(f'Unexpected prediction protocol in {path}: {data.get("protocol")}')
    return data, observed


def error_rows(data, keys, poses, camera_from_body):
    rows = []
    for prediction, key, camera_pose in zip(data['predictions'], keys, poses):
        result = {'scan': key, 'status': prediction.get('status')}
        if prediction.get('status') == 'ok':
            predicted = np.asarray(prediction.get('T_world_body'), dtype=np.float64)
            if predicted.shape != (4, 4) or not np.isfinite(predicted).all():
                result.update({'status': 'invalid_transform', 'reason': 'shape_or_nonfinite'})
            else:
                rotation = predicted[:3, :3]
                orthogonality = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord='fro')
                determinant = np.linalg.det(rotation)
                if orthogonality > 0.01 or abs(determinant - 1.0) > 0.01:
                    result.update({'status': 'invalid_rotation',
                                   'orthogonality_error': float(orthogonality),
                                   'rotation_determinant': float(determinant)})
                else:
                    truth = camera_pose @ camera_from_body
                    translation = np.linalg.norm(predicted[:3, 3] - truth[:3, 3])
                    cosine = (np.trace(rotation.T @ truth[:3, :3]) - 1.0) / 2.0
                    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                    result.update({'mpe_m': float(translation), 'moe_deg': float(angle)})
        rows.append(result)
    return rows


def summarize(data, rows, predictions_sha256):
    successes = [row for row in rows if row['status'] == 'ok']
    translation = np.asarray([row['mpe_m'] for row in successes], dtype=np.float64)
    rotation = np.asarray([row['moe_deg'] for row in successes], dtype=np.float64)
    all_success = len(successes) == len(rows)
    joint = (translation < 1.0) & (rotation < 2.0)
    return {
        'protocol': 'camera_mast3r_anchored_independent_gt_evaluator_v1',
        'predictions_sha256': predictions_sha256,
        'subset': data['subset'],
        'frames': len(rows),
        'successful_frames': len(successes),
        'failed_frames': len(rows) - len(successes),
        'all_frame_mpe_mean_m': float(translation.mean()) if all_success else None,
        'all_frame_moe_mean_deg': float(rotation.mean()) if all_success else None,
        'all_frame_mpe_median_m': float(np.median(translation)) if all_success else None,
        'all_frame_moe_median_deg': float(np.median(rotation)) if all_success else None,
        'all_frame_mpe_p90_m': float(np.percentile(translation, 90)) if all_success else None,
        'all_frame_moe_p90_deg': float(np.percentile(rotation, 90)) if all_success else None,
        'all_frame_mpe_p95_m': float(np.percentile(translation, 95)) if all_success else None,
        'all_frame_moe_p95_deg': float(np.percentile(rotation, 95)) if all_success else None,
        'all_frame_rotation_gt_10_deg': int((rotation > 10.0).sum()) if all_success else None,
        'all_frame_rotation_gt_90_deg': int((rotation > 90.0).sum()) if all_success else None,
        'all_frame_joint_1m_2deg_count': int(joint.sum()),
        'all_frame_joint_1m_2deg_recall': float(joint.sum() / len(rows)),
        'success_only_mpe_mean_m': float(translation.mean()) if len(successes) else None,
        'success_only_moe_mean_deg': float(rotation.mean()) if len(successes) else None,
        'elapsed_online_seconds': data['elapsed_seconds'],
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--baseline-predictions', type=Path, required=True)
    parser.add_argument('--candidate-predictions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_sha = load_frozen_predictions(
        args.baseline_predictions, {'local905_gt_isolated_online_v1'})
    candidate, candidate_sha = load_frozen_predictions(
        args.candidate_predictions, {
            'relation_mast3r_lidar_anchor_refinement_v1',
            'camera_mast3r_anchored_mean_visual_diagnostic_v1',
        })
    split = json.loads(args.split.read_text(encoding='utf-8'))
    split_sha = digest(args.split)
    if baseline['subset'] != candidate['subset'] or baseline['subset'] not in ('val', 'test'):
        raise ValueError('Baseline and candidate subsets differ or are invalid')
    if baseline.get('split_sha256') != split_sha or candidate.get('split_sha256') != split_sha:
        raise ValueError('Prediction split SHA256 mismatch')
    keys = split['splits'][baseline['subset']]
    for data in (baseline, candidate):
        if [row['scan'] for row in data['predictions']] != keys:
            raise ValueError('Predictions do not cover the fixed subset in order')
        if data.get('expected_frames') != len(keys):
            raise ValueError('Prediction count does not match the fixed subset denominator')

    scene = Path(args.data_root) / 'train_scene'
    metadata = json.loads((scene / 'scene_meta.json').read_text(encoding='utf-8'))
    camera_from_body = np.linalg.inv(np.asarray(metadata['T_BC_camera_to_body'], dtype=np.float64))
    poses = [np.loadtxt(scene / 'train' / 'poses' / (Path(key).stem + '.txt'), dtype=np.float64)
             for key in keys]
    if any(pose.shape != (4, 4) or not np.isfinite(pose).all() for pose in poses):
        raise ValueError('Ground-truth pose file has invalid shape or values')

    baseline_rows = error_rows(baseline, keys, poses, camera_from_body)
    candidate_rows = error_rows(candidate, keys, poses, camera_from_body)
    baseline_summary = summarize(baseline, baseline_rows, baseline_sha)
    candidate_summary = summarize(candidate, candidate_rows, candidate_sha)
    baseline_summary['ground_truth_access'] = 'after baseline and candidate prediction sidecars, split hashes, and frame order were verified'
    candidate_summary['ground_truth_access'] = 'after baseline and candidate prediction sidecars, split hashes, and frame order were verified'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, summary in (('baseline_evaluation.json', baseline_summary),
                          ('candidate_evaluation.json', candidate_summary)):
        output = args.out_dir / name
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'subset': baseline['subset'],
        'baseline_predictions_sha256': baseline_sha,
        'candidate_predictions_sha256': candidate_sha,
        'baseline': {key: value for key, value in baseline_summary.items() if key not in ('rows', 'ground_truth_access')},
        'candidate': {key: value for key, value in candidate_summary.items() if key not in ('rows', 'ground_truth_access')},
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
