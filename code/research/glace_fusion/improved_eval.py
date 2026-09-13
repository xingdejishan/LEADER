"""Audit freshly inferred coordinates against the test scans (stage-4 arms).

Reads an evaluate_scene output directory (coordinates/<stem>.npz + GT/K from
the scene meta) and computes the same per-frame correspondence audit metrics
the bundle reports use, so baseline and improved arms are directly comparable
with the cached selected model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates', type=Path, required=True,
                        help='evaluate_scene output containing coordinates/<stem>.npz')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    sys_path = str(ROOT / 'code')
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from research.glace_fusion.correspondence_audit import metrics, summarize

    rows = json.loads((ROOT / 'data/test_rows.json').read_text(encoding='utf-8'))
    meta = json.loads((ROOT / 'data/train_scene/scene_meta.json').read_text(encoding='utf-8'))
    E = np.asarray(meta['T_BC_camera_to_body'])
    records = []
    for row in rows:
        path = args.coordinates / 'coordinates' / (row['image'] + '.npz')
        if not path.exists():
            continue
        data = np.load(path)
        scan = ROOT / 'data/scans' / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')
        records.append(dict(row, metrics=metrics(data['xyz'], data['uv'], data['K'],
                                                 data['GT'], E, scan)))
    if not records:
        raise SystemExit('No coordinate files matched the test rows under '
                         + str(args.coordinates))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(records=records, summary=summarize(records)),
                                   indent=2), encoding='utf-8')
    print(json.dumps(summarize(records), indent=2))


if __name__ == '__main__':
    main()
