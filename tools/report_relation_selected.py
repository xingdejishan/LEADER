import json
from pathlib import Path

import torch

from tools.compare_local905 import compare
from tools.train_local905 import digest
from tools.train_gravity_fusion import objective


def main():
    root = Path('../relation-training-20260925')
    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))
    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2)+'\n')
    protocol = read(root/'protocol.json')
    for name, fingerprint in protocol['source_sha256'].items():
        assert digest(Path(name)) == fingerprint, name
    selection = read(root/'selection_frozen.json')
    records = read(root/'validation_selection.json')['records']
    best = min(records, key=lambda row: row['objective'])
    assert best['epoch'] == selection['epoch'] == selection['epoch']
    assert digest(root/'best.pt') == selection['checkpoint_sha256'] == best['checkpoint_sha256']
    for row in records:
        folder = root/f"val_{row['epoch']:04d}"
        pred, evaluation = read(folder/'predictions.json'), read(folder/'evaluation.json')
        assert digest(folder/'predictions.json') == evaluation['predictions_sha256']
        assert pred['checkpoint_sha256'] == row['checkpoint_sha256']
    test, baseline = read(root/'selected_test/evaluation.json'), read(root/'L0_test/evaluation.json')
    pred = read(root/'selected_test/predictions.json')
    assert digest(root/'selected_test/predictions.json') == test['predictions_sha256']
    assert pred['checkpoint_sha256'] == selection['checkpoint_sha256']
    assert digest(root/'L0_test/predictions.json') == baseline['predictions_sha256']
    assert test['frames'] == baseline['frames'] == len(pred['predictions']) == 313
    score, eligible = objective(test, baseline)
    report = dict(goal_met=eligible and score<=.9, best_epoch=selection['epoch'],
        metrics={k:v for k,v in test.items() if k!='rows'},
        baseline_metrics={k:v for k,v in baseline.items() if k!='rows'},
        selected_minus_L0=compare(test,baseline),
        mpe_reduction=1-test['all_frame_mpe_mean_m']/baseline['all_frame_mpe_mean_m'],
        moe_reduction=1-test['all_frame_moe_mean_deg']/baseline['all_frame_moe_mean_deg'], development_only=True)
    previous = read(Path('../interaction-training-20260925/selected_test/evaluation.json'))
    report['selected_minus_previous_interaction'] = compare(test, previous)
    report['previous_interaction_metrics'] = {k:v for k,v in previous.items() if k!='rows'}
    write('comparison.json',report)
    history = [json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
    checkpoint = torch.load(root/'best.pt',map_location='cpu')
    write('training_report.json',dict(completed_epochs=history[-1]['epoch'],best_epoch=selection['epoch'],
        plateau_criterion_met=False,stop_reason='User stop after epoch10 comparison; best validation epoch5; partial epoch11 discarded',
        training_seconds=sum(row['seconds'] for row in history),peak_memory_mib=max(row['peak_mib'] for row in history),
        gradient_evidence=checkpoint['gradients'],pretrained_LEADER_initialization=True,joint_backbone_finetuning=True))
    write('final_integrity_review.json',dict(source_fingerprints_verified=True,
        checkpoint_prediction_evaluator_binding_verified=True,validation_best_epoch_verified=True,
        original_plateau_rule_superseded_by_user=True,checkpoint_sha256=selection['checkpoint_sha256'],
        predictions_sha256=test['predictions_sha256']))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
