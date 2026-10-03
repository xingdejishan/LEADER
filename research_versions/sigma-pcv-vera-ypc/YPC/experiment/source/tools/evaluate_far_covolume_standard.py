import argparse
import json
from pathlib import Path

import numpy as np

from tools.far_scoring import ground_truth,read_predictions,score_rows,describe,sha256_file,DEV_DATES


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','predictions','freeze','gt-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    freeze=json.loads(Path(args.freeze).read_text())
    assert freeze['ground_truth_read_online'] is False and freeze['frames']==1200
    assert freeze['prediction_sha256']==sha256_file(args.predictions)
    assert freeze['settings_sha256']==sha256_file(args.config)
    assert freeze['source_sha256']==config['source_sha256']
    assert freeze['checkpoint_sha256']==sha256_file(config['base_checkpoint'] if args.baseline else args.checkpoint)
    root=Path(__file__).resolve().parents[1]
    assert all(sha256_file(root/name)==digest for name,digest in config['source_sha256'].items())
    expected,truths,manifests,gt_hashes=ground_truth(config['data_root'],args.gt_root)
    assert manifests==freeze['validation_manifest_sha256']
    rows=read_predictions(args.predictions,expected)
    errors=score_rows(rows,truths)
    result=dict(overall=describe(errors,rows),per_date={date:describe(errors[i*300:(i+1)*300],rows[i*300:(i+1)*300]) for i,date in enumerate(DEV_DATES)},
        role='A_development_after_standard_training',independent_test=False,selection_set_used=True,training_split='official_train_dates_1200',test_split='A',
        prediction_sha256=freeze['prediction_sha256'],freeze_sha256=sha256_file(args.freeze),
        checkpoint_sha256=freeze['checkpoint_sha256'],settings_sha256=freeze['settings_sha256'],
        source_sha256=config['source_sha256'],manifest_sha256=manifests,gt_manifest_sha256=gt_hashes,
        evaluator_sha256=sha256_file(__file__),scoring_sha256=sha256_file(root/'tools/far_scoring.py'))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result['overall']))


if __name__=='__main__':main()
