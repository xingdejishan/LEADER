import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .inference_contract import InferenceSession, resolve_contract, RGB_PROTOCOL
from .joint_solver import pose_distance
from .nclt_camera import preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--head', required=True, type=Path)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--deit-checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pose-backend', choices=['opencv', 'dsacstar', 'none'], default='opencv')
    parser.add_argument('--coordinate-precision', choices=['amp', 'fp32_head'], default=None)
    args = parser.parse_args()
    scene = args.run_root / 'scene'
    meta = json.loads((scene / 'scene_meta.json').read_text())
    grouped = {}
    for row in meta['splits']['train']['pairs']:
        grouped.setdefault(row['sequence'], []).append(row['image'])
    selected = [(date, images[index]) for date, images in sorted(grouped.items())
                for index in np.linspace(0, len(images) - 1, 16, dtype=int)]
    paths = {p.stem: p for p in (scene / 'train/rgb').iterdir()}
    rgb = resolve_contract(args.head)['global_feature_protocol'] == RGB_PROTOCOL
    session = InferenceSession(args.vendor, args.head, args.deit_checkpoint,
        split=scene / 'train' if rgb else None,
        T_BC=np.asarray(meta['T_BC_camera_to_body']), pose_backend=args.pose_backend, coordinate_precision=args.coordinate_precision)
    criteria = {'minimum_pose_success_fraction': .8, 'success_translation_m': 2.,
                'success_rotation_deg': 5., 'maximum_median_reprojection_px': 20.,
                'minimum_mean_fraction_4px': .05}
    records = []
    for date, stem in selected:
        gt = np.loadtxt(scene / 'train/poses' / (stem + '.txt'))
        K = np.loadtxt(scene / 'train/calibration' / (stem + '.txt'))
        result = session.infer(paths[stem], K)
        K = result.K
        camera = (result.xyz_world - gt[:3, 3]) @ gt[:3, :3]
        projected = camera @ K.T
        with np.errstate(divide='ignore', invalid='ignore'):
            reprojection = np.linalg.norm(projected[:, :2] / projected[:, 2:] - result.uv, axis=1)
        reprojection[(camera[:, 2] <= 0) | ~np.isfinite(reprojection)] = np.inf
        row = {'sequence': date, 'image': stem, 'inliers': result.inlier_count,
               'median_reprojection_px': float(np.median(reprojection)),
               'fraction_4px': float(np.mean(reprojection < 4)),
               'fraction_10px': float(np.mean(reprojection < 10)), 't_m': None, 'r_deg': None}
        if result.T_WC is not None:
            translation, rotation = pose_distance(result.T_WC, gt)
            row.update(t_m=float(translation), r_deg=float(np.rad2deg(rotation)))
        records.append(row)
        print(json.dumps(row), flush=True)
    success = [r['t_m'] is not None and r['t_m'] < 2 and r['r_deg'] < 5 for r in records]
    reprojection = float(np.median([r['median_reprojection_px'] for r in records]))
    fraction = float(np.mean([r['fraction_4px'] for r in records]))
    ready = float(np.mean(success)) >= .8 and reprojection < 20 and fraction >= .05
    report = {'inference_contract': session.contract, 'ready_for_test': ready, 'head_sha256': session.contract['head_sha256'],
              'scope': '64 deterministic training images, 16 per sequence; no test frames',
              'criteria': criteria, 'n_images': len(records), 'pose_successes': int(sum(success)),
              'pose_success_fraction': float(np.mean(success)), 'median_reprojection_px': reprojection,
              'mean_fraction_4px': fraction, 'records': records}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
