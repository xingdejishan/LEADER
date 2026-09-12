import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    result = dict(experiment=json.loads((args.run_root / 'experiment.json').read_text()), arms={})
    records = {}
    paths_by_split = {}
    arms = ['unmasked', 'masked', 'stage1']
    if (args.run_root / 'lidar_auxiliary/eval_test/complete.json').exists():
        arms.append('lidar_auxiliary')
    for arm in arms:
        result['arms'][arm] = {}
        for split in ['train', 'test']:
            folder = args.run_root / arm / ('eval_' + split) if arm != 'stage1' else args.run_root / ('stage1_eval_' + split)
            summary = json.loads((folder / 'summary.json').read_text())
            entries = json.loads((folder / 'records.json').read_text())
            records[arm, split] = entries
            paths = sorted((folder / 'coordinates').glob('*.npz'))
            if split in paths_by_split:
                if [p.name for p in paths] != [p.name for p in paths_by_split[split]]:
                    raise ValueError('Evaluation image sets differ')
            else:
                paths_by_split[split] = paths
            median_errors = []
            for path, reference in zip(paths, paths_by_split[split]):
                with np.load(path) as data, np.load(reference) as ref:
                    for key in ['uv', 'K', 'GT']:
                        np.testing.assert_array_equal(data[key], ref[key])
                    camera = (data['xyz'] - data['GT'][:3, 3]) @ data['GT'][:3, :3]
                    projection = camera @ data['K'].T
                    with np.errstate(divide='ignore', invalid='ignore'):
                        error = np.linalg.norm(projection[:, :2] / projection[:, 2:] - data['uv'], axis=1)
                    error[(camera[:, 2] <= 0) | ~np.isfinite(error)] = np.inf
                    median_errors.append(float(np.median(error)))
            result['arms'][arm][split] = dict(n_frames=summary['n_frames'],
                head_sha256=summary['inference_contract']['head_sha256'],
                median_of_frame_median_reprojection_px=float(np.median(median_errors)),
                gt_evidence_mean=summary['gt_evidence_mean'],
                adjacent={mode: summary['modalities'][mode]['adjacent'] for mode in ['translation', 'rotation']})
    count = len(records['masked', 'test'])
    blocks = np.array_split(np.arange(count), min(8, count))
    draws = np.random.default_rng(2089).integers(0, len(blocks), (10000, len(blocks)))
    intervals = [('translation', '0.2:0.5'), ('translation', '0.5:1.0'), ('rotation', '0.5:1.0')]
    comparisons = [('paired_mask_effect', 'unmasked', 'masked')]
    if 'lidar_auxiliary' in arms:
        comparisons.append(('paired_lidar_effect', 'masked', 'lidar_auxiliary'))
    for label, baseline, candidate in comparisons:
        first, second = records[baseline, 'test'], records[candidate, 'test']
        if [r['timestamp_us'] for r in first] != [r['timestamp_us'] for r in second]:
            raise ValueError('Unpaired test results')
        result[label] = {}
        for mode, key in intervals:
            values = []
            for rows in [first, second]:
                counts = [r['modalities'][mode]['adjacent'][key] for r in rows]
                numerator = np.array([c['correct'] + .5 * c['ties'] for c in counts])
                denominator = np.array([c['pairs'] for c in counts])
                n = np.array([numerator[b].sum() for b in blocks])
                d = np.array([denominator[b].sum() for b in blocks])
                values.append((float(n.sum() / d.sum()), n[draws].sum(1) / d[draws].sum(1)))
            result[label][mode + '_' + key] = dict(
                half_credit_accuracy_delta=values[1][0] - values[0][0],
                paired_block_ci95=np.quantile(values[1][1] - values[0][1], [.025, .975]).tolist())
    result['scope'] = 'One local development region, one seed, 5000 steps; intervals are descriptive, not whole-NCLT evidence'
    (args.run_root / 'diagnostic_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
