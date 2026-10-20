import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--groundtruth', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding='utf-8'))
    train_xy = np.stack([
        np.loadtxt(args.data_root / 'train_scene' / 'train' / 'poses' /
                   (Path(key).stem + '.txt'))[:2, 3]
        for key in split['splits']['train']
    ])
    gt = np.loadtxt(args.groundtruth, delimiter=',')
    sampled = gt[::100]
    distance, _ = cKDTree(train_xy).query(sampled[:, 1:3])
    windows = []
    window = 100
    for start in range(0, len(sampled) - window + 1):
        chunk = distance[start:start + window]
        windows.append((int((chunk < 20).sum()), start,
                        int(sampled[start, 0]), int(sampled[start + window - 1, 0]),
                        float(np.median(chunk))))
    windows.sort(reverse=True)
    report = {
        'train_frame_count': len(train_xy), 'groundtruth_row_count': len(gt),
        'sampled_rows': len(sampled),
        'train_xy_min': train_xy.min(axis=0).tolist(),
        'train_xy_max': train_xy.max(axis=0).tolist(),
        'candidate_xy_min': sampled[:, 1:3].min(axis=0).tolist(),
        'candidate_xy_max': sampled[:, 1:3].max(axis=0).tolist(),
        'within_10m': int((distance < 10).sum()),
        'within_20m': int((distance < 20).sum()),
        'within_50m': int((distance < 50).sum()),
        'median_nearest_train_m': float(np.median(distance)),
        'best_100_sample_windows': [
            {'within_20m': count, 'sample_start': start,
             'timestamp_start_us': first, 'timestamp_end_us': last,
             'median_nearest_train_m': median}
            for count, start, first, last, median in windows[:5]
        ],
    }
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
