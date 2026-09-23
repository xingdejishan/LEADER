import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected_hash = args.predictions.with_suffix('.sha256').read_text(encoding='ascii').strip()
    observed_hash = digest(args.predictions)
    if observed_hash != expected_hash:
        raise ValueError('Prediction file changed after online runner closed')
    data = json.loads(args.predictions.read_text(encoding='utf-8'))
    split = json.loads(args.split.read_text(encoding='utf-8'))
    keys = split['splits']['test']
    if data['protocol'] != 'local905_gt_isolated_online_v1':
        raise ValueError('Unknown online protocol')
    if data['split_sha256'] != digest(args.split):
        raise ValueError('Online split hash mismatch')
    if [row['scan'] for row in data['predictions']] != keys:
        raise ValueError('Predictions do not cover the fixed test denominator in order')
    scene = args.data_root / 'train_scene'
    meta = json.loads((scene / 'scene_meta.json').read_text(encoding='utf-8'))
    body_from_camera = np.asarray(meta['T_BC_camera_to_body'], dtype=np.float64)
    camera_from_body = np.linalg.inv(body_from_camera)
    rows = []
    for row in data['predictions']:
        stem = Path(row['scan']).stem
        camera_pose = np.loadtxt(scene / 'train' / 'poses' / (stem + '.txt'))
        truth = camera_pose @ camera_from_body
        result = {'scan': row['scan'], 'status': row['status']}
        if row['status'] == 'ok':
            predicted = np.asarray(row['T_world_body'], dtype=np.float64)
            if predicted.shape != (4, 4) or not np.isfinite(predicted).all():
                raise ValueError(f'Invalid predicted transform: {stem}')
            translation = np.linalg.norm(predicted[:3, 3] - truth[:3, 3])
            cosine = (np.trace(predicted[:3, :3].T @ truth[:3, :3]) - 1) / 2
            rotation = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            result.update({'mpe_m': float(translation), 'moe_deg': float(rotation)})
        rows.append(result)
    successes = [row for row in rows if row['status'] == 'ok']
    errors_t = np.asarray([row['mpe_m'] for row in successes])
    errors_q = np.asarray([row['moe_deg'] for row in successes])
    all_success = len(successes) == len(rows)
    summary = {
        'protocol': 'local905_gt_isolated_evaluator_v1',
        'predictions_sha256': observed_hash,
        'test_frames': len(rows), 'successful_frames': len(successes),
        'failed_frames': len(rows) - len(successes),
        'all_frame_mpe_mean_m': float(errors_t.mean()) if all_success else None,
        'all_frame_moe_mean_deg': float(errors_q.mean()) if all_success else None,
        'all_frame_mpe_p90_m': float(np.percentile(errors_t, 90)) if all_success else None,
        'all_frame_moe_p90_deg': float(np.percentile(errors_q, 90)) if all_success else None,
        'success_only_mpe_mean_m': float(errors_t.mean()) if len(successes) else None,
        'success_only_moe_mean_deg': float(errors_q.mean()) if len(successes) else None,
        'elapsed_online_seconds': data['elapsed_seconds'],
        'rows': rows,
    }
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
