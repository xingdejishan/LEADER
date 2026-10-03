import json
import os
from pathlib import Path
import sys
import time

from trial_common import PLAN, STORE, atomic, guard, sha, verify_source
assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
verify_source()
guard()
import numpy as np

epoch = int(sys.argv[1])
assert epoch in PLAN['evaluate_epochs']
inputs = json.loads((STORE / 'inputs.json').read_text())['dev']
folder = STORE / 'eval' / ('e%02d' % epoch)
baseline_folder = Path(PLAN['leader_finetune_reuse_folder'])
paths = {'vera': folder, 'leader_finetune': baseline_folder}
frozen = {}
rows = {}
for arm, current in paths.items():
    frozen[arm] = json.loads((current / 'prediction-freeze.json').read_text())
    assert frozen[arm]['frames'] == 1200 and sha(current / 'predictions.jsonl') == frozen[arm]['prediction_sha256']
    rows[arm] = [json.loads(row) for row in (current / 'predictions.jsonl').read_text().splitlines()]
    assert [row['key'] for row in rows[arm]] == [record['key'] for record in inputs]
truth, truth_hashes = {}, {}
data_root = Path(PLAN['data_root'])
for date in PLAN['dev_dates']:
    public = json.loads((data_root / date / 'manifest.json').read_text())
    transform = np.array(public['roi']['T_BC'])
    gt_root = data_root / 'evaluator_only/test_gt' / date
    manifest = gt_root / 'manifest.json'
    digest = sha(manifest)
    assert (gt_root / 'manifest.sha256').read_text().split()[0] == digest
    spec = json.loads(manifest.read_text())
    assert spec['split'] == 'test_evaluator_only'
    truth_hashes[date] = digest
    frames = {frame['timestamp_us']: frame for frame in spec['frames']}
    for record in inputs:
        if record['date'] != date:
            continue
        entry = frames[record['timestamp']]['ground_truth']
        path = (gt_root / entry['path']).resolve()
        assert gt_root.resolve() in path.parents and sha(path) == entry['sha256']
        truth[record['key']] = np.loadtxt(path).reshape(4, 4) @ np.linalg.inv(transform)
errors, results = {}, {}
for arm, predictions in rows.items():
    values = []
    for row in predictions:
        pose = np.array(row['pose'])
        gt = truth[row['key']]
        assert np.isfinite(pose).all() and np.max(np.abs(pose[3] - [0, 0, 0, 1])) < 1e-6
        assert np.max(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3))) < 1e-2
        values.append([np.linalg.norm(pose[:3, 3] - gt[:3, 3]), np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ gt[:3, :3]) - 1) / 2, -1, 1)))])
    error = np.array(values)
    errors[arm] = error
    results[arm] = {'frames': 1200, 'failures': sum(row['status'] != 'success' for row in predictions), 'mpe_m': float(error[:, 0].mean()), 'moe_deg': float(error[:, 1].mean()), 'mpe_p90_m': float(np.percentile(error[:, 0], 90)), 'moe_p90_deg': float(np.percentile(error[:, 1], 90)), 'joint_1m_2deg': int(((error[:, 0] < 1) & (error[:, 1] < 2)).sum()), 'historical_reuse': arm == 'leader_finetune', 'prediction_sha256': sha(paths[arm] / 'predictions.jsonl'), 'freeze_sha256': sha(paths[arm] / 'prediction-freeze.json'), 'dates': {date: {'frames': 300, 'mpe_m': float(error[[record['date'] == date for record in inputs], 0].mean()), 'moe_deg': float(error[[record['date'] == date for record in inputs], 1].mean())} for date in PLAN['dev_dates']}}
assert np.isclose(results['leader_finetune']['mpe_m'], .17949788079580942, atol=1e-10)
assert np.isclose(results['leader_finetune']['moe_deg'], 1.6242310898570118, atol=1e-10)
delta = errors['vera'] - errors['leader_finetune']
np.savez_compressed(folder / 'paired-errors.npz', **errors, keys=np.array([record['key'] for record in inputs]), truth=np.stack([truth[record['key']] for record in inputs]))
report = {'version': 'cam5-repair-v2', 'epoch': epoch, 'protocol': 'Original main unchanged, Cam5, 1200 Train / 1200 Dev, historical pure LEADER finetune reused', 'results': results, 'paired': {'mean_delta': delta.mean(0).tolist(), 'position_improved': int((delta[:, 0] < -1e-6).sum()), 'position_worsened': int((delta[:, 0] > 1e-6).sum()), 'rotation_improved': int((delta[:, 1] < -1e-6).sum()), 'rotation_worsened': int((delta[:, 1] > 1e-6).sum())}, 'admission': {'mean_admitted': float(np.mean([row.get('admitted', 0) for row in rows['vera']])), 'mean_admitted_in_pool': float(np.mean([row.get('admitted_in_pool', 0) for row in rows['vera']])), 'mean_pool_swaps': float(np.mean([row.get('pool_swaps', 0) for row in rows['vera']]))}, 'plan_sha256': sha(STORE / 'plan.json'), 'inputs_sha256': PLAN['inputs_sha256'], 'truth_manifest_sha256': truth_hashes, 'paired_errors_sha256': sha(folder / 'paired-errors.npz'), 'independent_test': False, 'limitations': ['Repeated previously-used Dev dates; no independent test or training-seed evidence', 'Historical finetune trained full RPGE/head and fixed concat; repair uses original-main frozen official-L0 backbone/head, new module only; whole-system comparison, not an isolated VERA ablation', 'New module is discrete candidate selection with internal predicted rotation transport and protected pool; does not claim unchanged VERA architecture', 'Frozen original-main concatenation may remain nondeterministic; no coordinate repair or CPU/CUDA parity claim', 'Short bounded trial does not establish full convergence or global mechanism invalidity']}
atomic(folder / 'report.json', report)
atomic(folder / 'completion.json', {'exit': 0, 'report_sha256': sha(folder / 'report.json'), 'frames': 1200, 'time': time.time()})
print(json.dumps({'event': 'score_complete', 'epoch': epoch, 'results': results, 'paired': report['paired'], 'admission': report['admission']}), flush=True)
