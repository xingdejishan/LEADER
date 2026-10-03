import json
import os
from pathlib import Path
import time

from common import PLAN, STORE, atomic, guard, prohibit_dev_gt, sha, verify_source

guard()
verify_source()
prohibit_dev_gt()
parent = Path(PLAN['parent_root']) / 'storage'
assert json.loads((parent / 'pipeline-exit.json').read_text())['exit'] == 0
assert sha(parent / 'plan.json') == PLAN['parent_plan_sha256']
assert sha(parent / 'inputs.json') == PLAN['inputs_sha256'] == sha(STORE / 'inputs.json')
comparison = json.loads((parent / 'comparison.json').read_text())
parent_plan = json.loads((parent / 'plan.json').read_text())
reports = []
for epoch in parent_plan['evaluate_epochs']:
    path = parent / 'eval' / ('e%02d' % epoch) / 'report.json'
    report = json.loads(path.read_text())
    assert report in comparison['reports'] and report['epoch'] == epoch
    freeze_path = path.parent / 'vera/prediction-freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['frames'] == 1200 and freeze['development_gt_reads'] == 0
    assert sha(freeze_path) == report['results']['vera']['freeze_sha256']
    assert sha(path.parent / 'vera/predictions.jsonl') == report['results']['vera']['prediction_sha256']
    reports.append(report)
reference = PLAN['selection_reference']
selected = min(reports, key=lambda report: (report['results']['vera']['failures'], report['results']['vera']['mpe_m'] / reference['mpe_mean_m'] + report['results']['vera']['moe_deg'] / reference['moe_mean_deg'], report['epoch']))
epoch = selected['epoch']
assert epoch == comparison['selected_epoch']
checkpoint = parent / 'vera' / ('epoch-%02d.pt' % epoch)
receipt_path = checkpoint.with_name('epoch-%02d-receipt.json' % epoch)
receipt = json.loads(receipt_path.read_text())
freeze_path = parent / 'eval' / ('e%02d' % epoch) / 'vera/prediction-freeze.json'
freeze = json.loads(freeze_path.read_text())
assert sha(checkpoint) == receipt['checkpoint_sha256'] == freeze['checkpoint_sha256']
assert receipt['plan_sha256'] == PLAN['parent_plan_sha256']
assert checkpoint.stat().st_uid == 1003
target = STORE / 'parent-checkpoint.pt'
if not target.exists():
    os.link(checkpoint, target)
assert sha(target) == sha(checkpoint)
binding = {'parent_root': PLAN['parent_root'], 'parent_plan_sha256': sha(parent / 'plan.json'), 'parent_comparison_sha256': sha(parent / 'comparison.json'), 'parent_completed_epochs': 30, 'selected_epoch': epoch, 'checkpoint': str(target), 'checkpoint_sha256': sha(target), 'parent_receipt_sha256': sha(receipt_path), 'prediction_freeze_sha256': sha(freeze_path), 'model_state_sha256': freeze['model_state_sha256'], 'inputs_sha256': sha(STORE / 'inputs.json'), 'selection_scope': 'previously used development set; selected support-v1 weights and Adam/RNG, not fresh independent test', 'phase_epoch_semantics': PLAN['epoch_semantics']}
path = STORE / 'parent-binding.json'
if path.exists():
    assert json.loads(path.read_text()) == binding
else:
    atomic(path, binding)
baseline = STORE / 'leader_finetune'
if not baseline.exists():
    baseline.symlink_to(parent / 'leader_finetune', target_is_directory=True)
assert sha(baseline / 'predictions.jsonl') == selected['results']['leader_finetune']['prediction_sha256']
assert sha(baseline / 'prediction-freeze.json') == selected['results']['leader_finetune']['freeze_sha256']
atomic(STORE / 'parent-binding-complete.json', {'exit': 0, 'binding_sha256': sha(path), 'selected_epoch': epoch, 'time': time.time()})
print(json.dumps({'bound': True, **binding}))
