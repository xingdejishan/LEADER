import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from .correspondence_replay import diagnostic_metrics, dump, supervision_from_scan
from .replay_geometry import point_errors
from .pose_boundary import solver_pose


def compact(errors, mask):
    angle = errors['angle_deg'][mask]
    finite = angle[np.isfinite(angle)]
    return dict(points=int(mask.sum()), q10=float(np.mean(errors['squared_px'][mask] < 100)) if mask.any() else None,
        median_angle_deg=float(np.median(finite)) if len(finite) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['selected', 'balanced'], default='selected')
    args = parser.parse_args()
    root, out = args.bundle.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache = root / 'cache' / ('validation_' + args.variant)
    manifest = json.loads((cache / 'manifest.json').read_text())
    rows = manifest['samples']
    if {r['sequence'] for r in rows} != {'2012-02-18'}:
        raise ValueError('Expected validation date only')
    E = solver_pose(np.asarray(json.loads((root / 'data/train_scene/scene_meta.json').read_text())['T_BC_camera_to_body']))
    dump(out / 'protocol.json', dict(date='2012-02-18', role='diagnostic only; already used for historical head selection',
        parameter_updates=False, thresholds_px=[10], input_features='per-image median Sobel magnitude and predicted range bins',
        semantic_labels_available=False, independent_reference_available=False,
        signed_residuals='projection minus observed pixel, saved for all pixels; invalid projections retain flags',
        depth='same-time sparse LiDAR target; not independent reference'))
    records = []
    for row in rows:
        stem = row['image']
        with np.load(cache / 'coordinates' / (stem + '.npz')) as data:
            xyz, uv, K, gt = [data[k] for k in ('xyz', 'uv', 'K', 'GT')]
        shape = (480, round(row['stored_size_hw'][1] * 480 / row['stored_size_hw'][0]))
        scan = root / 'data/scans' / row['sequence'] / 'velodyne_sync' / (stem + '.bin')
        target, support = supervision_from_scan(scan, uv, K, gt, E, shape) if scan.exists() else (None, np.zeros(len(uv), bool))
        errors = point_errors(xyz, uv, K, gt, shape, target)
        image_path = root / 'data/validation_scene/train/rgb' / (stem + '.jpg')
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert('L').resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=float) / 255
        gradient = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
        pixels = np.clip(np.rint(uv).astype(int), [0, 0], [shape[1] - 1, shape[0] - 1])
        texture = gradient[pixels[:, 1], pixels[:, 0]]
        high_texture = texture >= np.median(texture)
        ranges = np.linalg.norm(errors['camera'], axis=1)
        high_range = ranges >= np.median(ranges[np.isfinite(ranges)])
        masks = dict(all=np.ones(len(uv), bool), sparse_supported=support, high_texture=high_texture,
            low_texture=~high_texture, far_prediction=high_range, near_prediction=~high_range)
        metrics = diagnostic_metrics(errors, uv, shape)
        if target is not None:
            for name in ('along_error_m', 'range_error_m'):
                values = errors[name][support]
                metrics[name] = dict(median=float(np.median(values)), median_abs=float(np.median(np.abs(values)))) if len(values) else None
        record = dict(image=stem, sequence=row['sequence'], metrics=metrics,
            groups={name: compact(errors, mask) for name, mask in masks.items()})
        np.savez_compressed(out / (stem + '.npz'), uv=uv, signed_px=errors['signed_px'],
            angle_deg=errors['angle_deg'], positive=errors['positive'], support=support,
            predicted_range_m=ranges, texture=texture,
            **{name: errors[name] for name in ('perpendicular_m', 'along_error_m', 'range_error_m') if name in errors})
        records.append(record)
    groups = {}
    for name in records[0]['groups']:
        values = [r['groups'][name] for r in records if r['groups'][name]['points']]
        groups[name] = dict(frames=len(values), total_points=sum(r['points'] for r in values),
            mean_frame_q10=float(np.mean([r['q10'] for r in values])),
            median_frame_angle_deg=float(np.median([r['median_angle_deg'] for r in values if r['median_angle_deg'] is not None])))
    dump(out / 'records.json', records)
    dump(out / 'summary.json', dict(frames=len(records), groups=groups,
        median_frame_residual_direction_coherence=float(np.median([r['metrics']['residual_direction_coherence'] for r in records])),
        warning='Texture/depth bins are descriptive; no semantic/static labels, no causal attribution or threshold fitting'))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
