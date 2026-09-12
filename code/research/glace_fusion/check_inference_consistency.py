import argparse
import json
from pathlib import Path

import numpy as np

from .inference_contract import InferenceSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError('count must be positive')
    cached = InferenceSession(args.vendor, args.head, args.checkpoint, split=args.split, pose_backend='none')
    online = InferenceSession(args.vendor, args.head, args.checkpoint, pose_backend='none')
    if cached.contract['head_sha256'] != online.contract['head_sha256']:
        raise ValueError('Checkpoint changed while loading; use an immutable snapshot')
    paths = [args.split / 'rgb' / name for name in cached.cache.names]
    selected = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.count, len(paths)), dtype=int)]
    records = []
    for path in selected:
        K = np.loadtxt(args.split / 'calibration' / (path.stem + '.txt'))
        for precision in ['amp', 'fp32_head']:
            cached.adapter.coordinate_precision = online.adapter.coordinate_precision = precision
            a, repeat, b = cached.infer(path, K), cached.infer(path, K), online.infer(path, K)
            np.testing.assert_array_equal(a.uv, b.uv)
            np.testing.assert_array_equal(a.K, b.K)
            np.testing.assert_array_equal(a.xyz_world, repeat.xyz_world)
            if not np.isfinite(a.xyz_world).all() or not np.isfinite(b.xyz_world).all():
                raise ValueError('Non-finite scene coordinates')
            distance = np.linalg.norm(a.xyz_world - b.xyz_world, axis=1)
            records.append(dict(image=path.stem, precision=precision,
                global_feature_max_abs=float(np.max(np.abs(cached.cache[path.stem] - online.extract([path])[0]))),
                coordinate_difference_m=dict(median=float(np.median(distance)),
                    p99=float(np.quantile(distance, .99)), maximum=float(distance.max()))))
    report = dict(head_sha256=cached.contract['head_sha256'], records=records,
                  repeated_cache_exact=True, pixel_grid_and_K_exact=True,
                  scope='numerical path comparison only; not a localization quality test')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
