"""Quality evaluation of the per-cell reliability head (stage-4 arm B).

For each test frame: run the arm's regressor + ReliabilityHead, build
LiDAR depth-agreement labels with the same `camera_targets` logic the audit
uses, and report the rank AUC of reliability against those labels (0.5 =
random, 1.0 = perfect separation of within-25% depth cells).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / 'code'))
    from research.glace_fusion.glace_adapter import GLACEAdapter
    from research.glace_fusion.lidar_supervision import camera_targets
    from research.glace_fusion.inference_contract import preprocess_image

    meta = json.loads((ROOT / 'data/train_scene/scene_meta.json').read_text(encoding='utf-8'))
    E = np.asarray(meta['T_BC_camera_to_body'])
    rows = json.loads((ROOT / 'data/test_rows.json').read_text(encoding='utf-8'))[:args.limit or None]
    manifest = json.loads((ROOT / 'data/test_scene/test/features_manifest.json').read_text(encoding='utf-8'))
    features = np.load(ROOT / 'data/test_scene/test/features.npy')
    lookup = {Path(name).stem: i for i, name in enumerate(manifest['images'])}

    adapter = GLACEAdapter(ROOT / 'models/selected/vendor',  # encoder path anchor; head overrides below
                           args.model_dir / 'head.pt',
                           encoder_path=ROOT / 'models/selected/vendor/ace_encoder_pretrained.pt',
                           T_BC=E, reliability_head_path=args.model_dir / 'head.pt.rel.pt')
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
                      ('intensity', 'u1'), ('ring', 'u1')])
    records = []
    for row in rows:
        stem = row['image']
        K = np.loadtxt(ROOT / 'data/test_scene/test/calibration' / (stem + '.txt'))
        gt = np.loadtxt(ROOT / 'data/test_scene/test/poses' / (stem + '.txt'))
        gray, scaled_K = preprocess_image(
            ROOT / 'data/test_scene/test/rgb' / (stem + '.jpg'), K, 480)
        out = adapter.infer(gray, scaled_K, global_feature=features[lookup[stem]])
        camera = (out.xyz_world - gt[:3, 3]) @ gt[:3, :3]
        pred_depth = camera[:, 2]
        scan = ROOT / 'data/scans' / row['sequence'] / 'velodyne_sync' / (stem + '.bin')
        if not scan.exists():
            continue
        raw = np.fromfile(scan, dtype=dtype)
        body = np.column_stack([raw[k] for k in ('x', 'y', 'z')]).astype(float) * .005 - 100
        T = gt @ np.linalg.inv(E)
        world = body @ T[:3, :3].T + T[:3, 3]
        target, support = camera_targets(world, out.uv, scaled_K, np.linalg.inv(gt),
                                         gray.shape[0], gray.shape[1], 3)
        supported = support > 0
        if not supported.any():
            continue
        log_ratio = np.log(np.maximum(pred_depth, 1e-3)) - np.log(np.maximum(target[:, 2], 1e-3))
        labels = supported & (np.abs(log_ratio) <= np.log(1.25))
        auc = rank_auc(out.reliability, labels)
        records.append(dict(image=stem, auc=auc, supported=int(supported.sum()),
                            positive=float(labels.sum() / max(supported.sum(), 1))))
    if not records:
        raise SystemExit('No frames with LiDAR support')
    aucs = [r['auc'] for r in records if r['auc'] is not None]
    summary = dict(n_frames=len(records),
                   mean_auc=float(np.mean(aucs)) if aucs else None,
                   median_auc=float(np.median(aucs)) if aucs else None,
                   mean_positive_rate=float(np.mean([r['positive'] for r in records])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, records=records), indent=2),
                        encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
