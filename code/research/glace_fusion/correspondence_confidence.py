import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .lidar_supervision import camera_targets


def features(xyz, uv):
    xyz = np.asarray(xyz, dtype=float)
    uv = np.asarray(uv, dtype=float)
    if xyz.shape != (len(uv), 3) or uv.shape != (len(xyz), 2):
        raise ValueError('Correspondence dimensions differ')
    cells = np.rint((uv - 4) / 8).astype(int)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    distances = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        index = np.array([lookup.get((x + dx, y + dy), -1) for x, y in cells])
        delta = np.linalg.norm(xyz - xyz[np.maximum(index, 0)], axis=1)
        delta[index < 0] = np.nan
        distances.append(np.log1p(delta))
    distances = np.column_stack(distances)
    return np.nan_to_num(np.column_stack([uv / [630, 480],
        (xyz - [-79.050591184, -320.427472972, 7.196297961]) / 40,
        distances]), nan=10, posinf=10, neginf=-10).astype(np.float32)


def labels(data):
    xyz, uv, K, gt = [data[k] for k in ['xyz', 'uv', 'K', 'GT']]
    camera = (xyz - gt[:3, 3]) @ gt[:3, :3]
    projection = camera @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.linalg.norm(projection[:, :2] / projection[:, 2:] - uv, axis=1)
    return (camera[:, 2] > 0) & (error < 10) & np.isfinite(error)


def select(probability, fraction=.25):
    if not 0 < fraction <= 1:
        raise ValueError('Invalid retained fraction')
    return np.sort(np.argsort(-np.asarray(probability), kind='stable')[:max(1, int(len(probability) * fraction))])


def classifier():
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
        learning_rate=.08, l2_regularization=10, min_samples_leaf=100,
        random_state=2089, early_stopping=False)


def point_probabilities(model, xyz, uv, protocol):
    probability = model.predict_proba(features(xyz, uv))[:, 1]
    if 'supported_grid_cells' in protocol:
        cells = np.clip((uv / [630, 480] * 8).astype(int), 0, 7)
        supported = np.asarray(protocol['supported_grid_cells'])[cells[:, 1] * 8 + cells[:, 0]]
        probability[~supported] = 0
    return probability


def quality_labels(data, row, E, dataset_root='/root/rivermind-data/datasets/NCLT'):
    path = Path(dataset_root) / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')
    if not path.exists():
        return np.zeros(len(data['uv']), bool), np.zeros(len(data['uv']), bool)
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
    raw = np.fromfile(path, dtype=dtype)
    body = np.column_stack([raw[k] for k in ['x', 'y', 'z']]).astype(float) * .005 - 100
    gt, uv, K = data['GT'], data['uv'], data['K']
    T = gt @ np.linalg.inv(E)
    world = body @ T[:3, :3].T + T[:3, 3]
    target, support = camera_targets(world, uv, K, np.linalg.inv(gt), 480, 630, 3)
    camera = (data['xyz'] - gt[:3, 3]) @ gt[:3, :3]
    return (np.linalg.norm(camera - target, axis=1) < 1) & labels(data), support > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--quality-target', choices=['reprojection', 'joint3d'], default='reprojection')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    manifest = json.loads((args.validation / 'manifest.json').read_text())
    test_manifest = json.loads((args.test / 'manifest.json').read_text())
    if manifest['inference_contract'] != test_manifest['inference_contract']:
        raise ValueError('Confidence model and test inference contracts differ')
    protocol = dict(head_sha256=manifest['inference_contract']['head_sha256'],
        valid_mask_sha256=manifest['valid_mask']['sha256'],
        inference_contract=manifest['inference_contract'],
        inputs='predicted world coordinates, pixel location, eight-neighbor coordinate differences',
        inference_uses_gt=False, inference_uses_lidar=False,
        training_labels=args.quality_target + ' on held-out training date 2012-02-18',
        quality_target=args.quality_target,
        joint3d_definition='sparse 3D distance below 1m and reprojection below 10px; unlabeled pixels excluded',
        validation_split='one-minute time blocks modulo three; block zero reserved for filter audit',
        final_fit='all validation-date samples after fixed filter audit',
        samples_per_frame=512, retained_fraction=.25, hyperparameters=classifier().get_params(),
        caveat='local confidence model; correlated frames, no whole-NCLT calibration claim')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rng = np.random.default_rng(2089)
    fitting, testing, arrays = [], [], []
    rows = {r['image']: r for r in manifest['samples']}
    meta_path = args.bundle / 'data/train_scene/scene_meta.json' if args.bundle else Path('/root/rivermind-data/glace_nclt_rgb_large_20260912/scene/scene_meta.json')
    dataset_root = args.bundle / 'data/scans' if args.bundle else Path('/root/rivermind-data/datasets/NCLT')
    meta = json.loads(meta_path.read_text())
    E = np.asarray(meta['T_BC_camera_to_body'])
    grid_counts = np.zeros(64, dtype=int)
    fitting_grid_counts = np.zeros(64, dtype=int)
    for path in sorted((args.validation / 'coordinates').glob('*.npz')):
        data = np.load(path)
        X, reprojection = features(data['xyz'], data['uv']), labels(data)
        y, support = (quality_labels(data, rows[path.stem], E, dataset_root) if args.quality_target == 'joint3d'
            else (reprojection, np.ones(len(reprojection), bool)))
        eligible = np.flatnonzero(support)
        if not len(eligible):
            continue
        selected = rng.choice(eligible, min(512, len(eligible)), replace=False)
        cells = np.clip((data['uv'][eligible] / [630, 480] * 8).astype(int), 0, 7)
        grid_counts += np.bincount(cells[:, 1] * 8 + cells[:, 0], minlength=64)
        arrays.append((X[selected], y[selected]))
        if (int(path.stem) // 60000000) % 3 == 0:
            testing.append((path.stem, X, reprojection, y, support))
        else:
            fitting.append((X[selected], y[selected]))
            fitting_grid_counts += np.bincount(cells[:, 1] * 8 + cells[:, 0], minlength=64)
    if not fitting or not testing:
        raise ValueError('Confidence fitting or validation split empty')
    model = classifier().fit(np.concatenate([x for x, _ in fitting]), np.concatenate([y for _, y in fitting]))
    heldout = []
    for stem, X, y, quality, support in testing:
        probability = model.predict_proba(X)[:, 1]
        if args.quality_target == 'joint3d':
            cells = np.clip((X[:, :2] * 8).astype(int), 0, 7)
            probability[fitting_grid_counts[cells[:, 1] * 8 + cells[:, 0]] < 50] = 0
        keep = select(probability)
        supported_keep = keep[support[keep]]
        heldout.append(dict(image=stem, raw_q10=float(y.mean()), selected_q10=float(y[keep].mean()),
            raw_joint_quality=float(quality[support].mean()),
            selected_joint_quality=None if not len(supported_keep) else float(quality[supported_keep].mean()),
            count=len(y), retained=len(keep)))
    (args.out / 'heldout_validation.json').write_text(json.dumps(dict(records=heldout,
        mean_raw_q10=float(np.mean([r['raw_q10'] for r in heldout])),
        mean_selected_q10=float(np.mean([r['selected_q10'] for r in heldout]))), indent=2))
    model.fit(np.concatenate([x for x, _ in arrays]), np.concatenate([y for _, y in arrays]))
    if args.quality_target == 'joint3d':
        protocol['supported_grid_cells'] = (grid_counts >= 50).tolist()
        protocol['coverage_rule'] = 'zero confidence in image cells with fewer than 50 sparse validation labels'
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    joblib.dump(model, args.out / 'confidence.joblib')
    (args.out / 'coordinates').mkdir()
    records = []
    for path in sorted((args.test / 'coordinates').glob('*.npz')):
        data = np.load(path)
        probability = point_probabilities(model, data['xyz'], data['uv'], protocol)
        keep = select(probability)
        np.savez_compressed(args.out / 'coordinates' / path.name, xyz=data['xyz'][keep],
            uv=data['uv'][keep], K=data['K'], GT=data['GT'], confidence=probability[keep], original_indices=keep)
        y = labels(data)
        records.append(dict(image=path.stem, raw_q10=float(y.mean()), selected_q10=float(y[keep].mean()),
            count=len(y), retained=len(keep)))
    (args.out / 'test_results.json').write_text(json.dumps(dict(records=records,
        mean_raw_q10=float(np.mean([r['raw_q10'] for r in records])),
        mean_selected_q10=float(np.mean([r['selected_q10'] for r in records]))), indent=2))


if __name__ == '__main__':
    main()
