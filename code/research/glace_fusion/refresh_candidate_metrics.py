import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .real_candidate_eval import dominance_counts, pose_errors, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root / 'real_candidates'
    for name in ['records.jsonl','report.json']:
        backup = root / ('before_so3_' + name)
        if not backup.exists():
            shutil.copyfile(root / name, backup)
    records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    for row in records:
        with np.load(root / 'pools' / (row['image'] + '.npz')) as data:
            errors = pose_errors(data['candidate_T_WB'], data['GT'])
            row['errors']['leader'] = pose_errors(data['leader'][None], data['GT'])[0].tolist()
            row['errors']['v1_two_stage'] = pose_errors(data['v1_two_stage'][None], data['GT'])[0].tolist()
            for label in ['stage1','improved']:
                row['errors']['camera_' + label] = errors[row['selected']['camera_' + label]].tolist()
                row['dominance'][label] = dominance_counts(errors, data[label + '_scores'])
            oracle = int(np.argmin(np.maximum(errors[:, 0], errors[:, 1] / 2)))
            row['errors']['oracle'] = errors[oracle].tolist()
            payload = {key: data[key] for key in data.files}
            if 'candidate_errors_before_so3' not in payload:
                payload['candidate_errors_before_so3'] = payload['candidate_errors']
            payload['candidate_errors'] = errors
            payload['metric_version'] = np.asarray('so3_atan2_v1')
        path = root / 'pools' / (row['image'] + '.npz')
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    report = json.loads((root / 'report.json').read_text())
    report['rotation_metric'] = 'nearest SO(3) for near-rigid float32 matrices, then atan2 angle; identical for all methods'
    report['groups'] = summarize(records, ['leader','v1_two_stage','camera_stage1','camera_improved','oracle'])
    (root / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (root / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
