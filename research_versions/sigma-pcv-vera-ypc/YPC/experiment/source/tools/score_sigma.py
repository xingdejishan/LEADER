import argparse
import json
from pathlib import Path

import numpy as np
from tools.far_scoring import ground_truth,score_rows,describe,sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    store=Path(config['store'])
    freeze=json.loads((store/'prediction_freeze.json').read_text())
    assert freeze['config_sha256']==sha256_file(args.config)
    assert freeze['prediction_sha256']==sha256_file(store/'predictions.jsonl')
    assert freeze['checkpoint_sha256']==sha256_file(store/('stage2-%02d.pt'%config['stage2_epochs']))
    assert sha256_file(config['baseline_predictions'])==config['baseline_prediction_sha256']
    expected,truths,manifests,gt_hashes=ground_truth(config['parent_config']['data_root'],config['parent_config']['gt_root'])
    assert manifests==freeze['manifest_hashes']
    baseline=[json.loads(line) for line in Path(config['baseline_predictions']).read_text().splitlines()]
    assert [row['frame_key'] for row in baseline]==expected
    indices=config['dev_indices']
    baseline=[baseline[i] for i in indices]
    truths=[truths[i] for i in indices]
    candidate=[json.loads(line) for line in (store/'predictions.jsonl').read_text().splitlines()]
    assert [row['frame_key'] for row in candidate]==[expected[i] for i in indices]==freeze['frame_keys']
    a,b=score_rows(baseline,truths),score_rows(candidate,truths)
    report=dict(baseline=describe(a,baseline),candidate=describe(b,candidate),
        candidate_minus_baseline=dict(mpe_m=float((b-a)[:,0].mean()),moe_deg=float((b-a)[:,1].mean())),
        improved_frames=dict(position=int((b[:,0]<a[:,0]).sum()),rotation=int((b[:,1]<a[:,1]).sum())),per_date={},
        config_sha256=freeze['config_sha256'],prediction_sha256=freeze['prediction_sha256'],freeze_sha256=sha256_file(store/'prediction_freeze.json'),
        checkpoint_sha256=freeze['checkpoint_sha256'],baseline_checkpoint_sha256=config['far_checkpoint_sha256'],
        baseline_prediction_sha256=config['baseline_prediction_sha256'],GT_manifest_sha256=gt_hashes,scorer_sha256=sha256_file(__file__),independent_test=False,
        limitations=['single seed; fixed last epoch','bounded three plus three epochs, not a convergence claim','joint backbone, covariance, descriptor and field updates; cannot attribute improvement solely to vision','shared local supports create correlated evidence; no statistical independence claim','reused development set'])
    for date in sorted(set(row['date'] for row in candidate)):
        ids=[i for i,row in enumerate(candidate) if row['date']==date]
        report['per_date'][date]=dict(baseline=describe(a[ids],[baseline[i] for i in ids]),candidate=describe(b[ids],[candidate[i] for i in ids]))
    np.savez(store/'paired_errors.npz',baseline=a,candidate=b)
    (store/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
