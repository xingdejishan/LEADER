import argparse
import json
from pathlib import Path

import numpy as np

from .camera_separability import camera_evidence
from .inference_contract import InferenceSession, resolve_contract, RGB_PROTOCOL
from .joint_solver import pose_distance
from .nclt_camera import validate_dates
from .pairwise_camera_ranking import LEVELS, candidates, pair_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--deit-checkpoint', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--image-stems', type=Path, help='JSON array of exact stems for paired comparisons')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--pose-backend', choices=['none', 'opencv', 'dsacstar'], default='none')
    parser.add_argument('--coordinate-precision', choices=['amp', 'fp32_head'], default=None)
    args = parser.parse_args()
    meta = json.loads((args.scene / 'scene_meta.json').read_text())
    validate_dates(meta['splits'].get('train', {}).get('dates', []),
                   meta['splits'].get('test', {}).get('dates', []))
    rows = sorted(meta['splits'][args.split]['pairs'], key=lambda r: (r['sequence'], r['image_timestamp_us']))
    if args.image_stems:
        stems = json.loads(args.image_stems.read_text())
        by_stem = {r['image']: r for r in rows}
        if len(stems) != len(set(stems)) or not set(stems) <= set(by_stem):
            raise ValueError('Invalid requested image stems')
        rows = [by_stem[s] for s in stems]
    elif args.limit > 0 and len(rows) > args.limit:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, args.limit, dtype=int)]
    if not rows:
        raise ValueError('No evaluation frames')
    split = args.scene / args.split
    rgb = resolve_contract(args.head)['global_feature_protocol'] == RGB_PROTOCOL
    session = InferenceSession(args.vendor, args.head, args.deit_checkpoint,
        split=split if rgb and not args.online else None,
        T_BC=np.asarray(meta['T_BC_camera_to_body']), pose_backend=args.pose_backend, coordinate_precision=args.coordinate_precision)
    paths = {p.stem: p for p in (split / 'rgb').iterdir()}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'coordinates').mkdir()
    manifest = dict(inference_contract=session.contract, split=args.split,
                    samples=rows, levels=LEVELS, missing_sequences=meta.get('missing_sequences', []),
                    scope='selected frames only; not full NCLT unless all test frames supplied',
                    score='mean(min((L2 reprojection / 10)^2,1)); invalid=1',
                    tie_epsilon=1e-8, gt_usage='diagnostic candidate generation only')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    records = []
    for row in rows:
        stem = row['image']
        K = np.loadtxt(split / 'calibration' / (stem + '.txt'))
        gt = np.loadtxt(split / 'poses' / (stem + '.txt'))
        result = session.infer(paths[stem], K)
        record = dict(sequence=row['sequence'], timestamp_us=row['image_timestamp_us'],
                      gt_evidence=camera_evidence(result.xyz_world, result.uv, result.K, gt),
                      modalities={}, pose_backend=result.diagnostics)
        if result.T_WC is not None:
            dt, dr = pose_distance(result.T_WC, gt)
            record['pose_error'] = dict(translation_m=float(dt), rotation_deg=float(np.rad2deg(dr)))
        else:
            record['pose_error'] = None
        for mode in LEVELS:
            scored = [dict(level=c['level'], direction=c['direction'], pose=c['pose'].tolist(),
                           **camera_evidence(result.xyz_world, result.uv, result.K, c['pose']))
                      for c in candidates(gt, mode)]
            record['modalities'][mode] = dict(candidates=scored, all_pairs=pair_counts(scored),
                same_direction=pair_counts(scored, same_direction=True),
                adjacent={f'{a}:{b}': pair_counts(scored, levels=(a, b))
                          for a, b in zip(LEVELS[mode][:-1], LEVELS[mode][1:])})
        np.savez_compressed(args.out / 'coordinates' / (stem + '.npz'),
                            xyz=result.xyz_world, uv=result.uv, K=result.K, GT=gt)
        records.append(record)
        print(json.dumps(dict(frames=len(records), total=len(rows))), flush=True)
    (args.out / 'records.json').write_text(json.dumps(records, indent=2))
    summary = dict(n_frames=len(records), inference_contract=session.contract, modalities={})
    summary['gt_evidence_mean'] = {key: float(np.mean([r['gt_evidence'][key] for r in records]))
                                  for key in ['S_C', 'q_C', 'positive_finite_depth_fraction']}
    for mode in LEVELS:
        counts = {k: sum(r['modalities'][mode]['all_pairs'][k] for r in records)
                  for k in ['correct', 'wrong', 'ties', 'pairs']}
        counts.update(accuracy=counts['correct'] / counts['pairs'],
                      half_credit_accuracy=(counts['correct'] + .5 * counts['ties']) / counts['pairs'])
        summary['modalities'][mode] = counts
        counts['adjacent'] = {}
        for interval in records[0]['modalities'][mode]['adjacent']:
            values = {k: sum(r['modalities'][mode]['adjacent'][interval][k] for r in records)
                      for k in ['correct', 'wrong', 'ties', 'pairs']}
            values.update(accuracy=values['correct'] / values['pairs'],
                half_credit_accuracy=(values['correct'] + .5 * values['ties']) / values['pairs'])
            counts['adjacent'][interval] = values
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.out / 'complete.json').write_text(json.dumps(dict(complete=True, frames=len(records))))


if __name__ == '__main__':
    main()
