import json
from pathlib import Path
import numpy as np

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912/paired_diagnostic_64')
report = json.loads((root / 'report.json').read_text())
records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
baseline = np.array([r['errors']['v1_two_stage'] for r in records])
fusion = np.array([r['errors']['fusion_with_v1_fallback'] for r in records])
delta = fusion - baseline
rng = np.random.default_rng(2089)
bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
comparison = {
    'scope': '64 preselected synchronized frames from 2012-02-12; diagnostic, not full benchmark',
    'frames': len(records),
    'fusion_minus_v1_mean_translation_m': float(delta[:, 0].mean()),
    'fusion_minus_v1_mean_rotation_deg': float(delta[:, 1].mean()),
    'translation_improved_over_1cm': int(np.sum(delta[:, 0] < -.01)),
    'translation_worsened_over_1cm': int(np.sum(delta[:, 0] > .01)),
    'translation_within_1cm': int(np.sum(np.abs(delta[:, 0]) <= .01)),
    'rotation_improved_over_0_1deg': int(np.sum(delta[:, 1] < -.1)),
    'rotation_worsened_over_0_1deg': int(np.sum(delta[:, 1] > .1)),
    'joint_frames': sum(r['status'] == 'JOINT' for r in records),
    'fallback_frames': sum(r['errors']['v2_native'] is None for r in records),
    'sync_abs_ms_median_max': np.percentile([abs(r['sync_delta_s'])*1000 for r in records], [50,100]).tolist(),
    'camera_support_at_v1_median': float(np.median([r['camera_support_at_v1'] for r in records])),
    'camera_support_at_scan_gt_median': float(np.median([r['camera_support_at_scan_gt'] for r in records])),
    'bootstrap_mean_delta_95pct': np.quantile(bootstrap,[.025,.975],axis=0).tolist(),
    'bootstrap_caution': 'Descriptive frame bootstrap; temporal dependence and one-sequence scope limit inference',
    'status_counts': report['status_counts'],
    'metrics': report['all_frames'],
}
(root / 'paired_summary.json').write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2))
