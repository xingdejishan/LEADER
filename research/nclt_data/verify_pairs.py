import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(workspace):
    workspace = Path(workspace)
    data = workspace / 'glace-local/data'
    audit = data / 'train907_raw_scan_audit'
    reference = Path(r'\\wsl.localhost\Ubuntu\home\zhang\rscore-l-local\data\manifest.json')
    original = json.loads(reference.read_text())
    download = json.loads((audit / 'manifest.json').read_text())
    expected = {r['frame_id']: r for r in download['records']}
    pairs, summary = {}, {}
    for split, folder in [('train', 'train_scene/train'), ('val', 'validation_scene/train'), ('test', 'test_scene/test')]:
        rows = []
        for row in original[split]:
            stem = row['frame_id']
            image = data / folder / 'rgb' / (stem+'.jpg')
            pose = data / folder / 'poses' / (stem+'.txt')
            calibration = data / folder / 'calibration' / (stem+'.txt')
            scan = data / 'scans' / row['session_id'] / 'velodyne_sync' / (stem+'.bin')
            if hashlib.sha256(image.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError('Image differs from frozen split: '+stem)
            p, k = np.loadtxt(pose), np.loadtxt(calibration)
            if p.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(k).all():
                raise ValueError('Invalid pose or intrinsics: '+stem)
            record = dict(row, image=str(image), pose=str(pose), calibration=str(calibration),
                          scan=str(scan) if scan.exists() else None, paired=scan.exists())
            if scan.exists():
                payload = scan.read_bytes()
                if not payload or len(payload) % 8:
                    raise ValueError('Invalid raw scan format: '+stem)
                record['scan_sha256'] = hashlib.sha256(payload).hexdigest()
                record['raw_points'] = len(payload)//8
                if split == 'train' and record['scan_sha256'] != expected[stem]['sha256']:
                    raise ValueError('Scan differs from server: '+stem)
            rows.append(record)
        pairs[split] = rows
        summary[split] = dict(original_frames=len(rows), fully_paired=sum(r['paired'] for r in rows),
                              missing_scans=[r['frame_id'] for r in rows if not r['paired']])
    summary['original_split_unchanged'] = True
    summary['timestamp_policy'] = 'Exact image/frame ID; no neighboring scan substituted'
    (audit / 'paired_manifest.json').write_text(json.dumps(pairs, indent=2))
    (audit / 'pairing_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    verify(args.workspace)
