import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation_base import ground_truth,read_predictions,score_rows,describe,sha256_file,DEV_DATES


def main():
    parser=argparse.ArgumentParser()
    for name in ('data-root','gt-root','predictions','freeze','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    freeze=json.loads(Path(args.freeze).read_text())
    if freeze['prediction_sha256']!=sha256_file(args.predictions) or not freeze['ground_truth_read_online'] is False:
        raise ValueError('invalid prediction freeze')
    keys,truths,manifests,gt_hashes=ground_truth(args.data_root,args.gt_root)
    if manifests!=freeze['validation_manifest_sha256']:
        raise ValueError('validation manifest changed')
    rows=read_predictions(args.predictions,keys)
    errors=score_rows(rows,truths)
    record=describe(errors,rows)
    record.update(per_date={date:describe(errors[[r['date']==date for r in rows]],[r for r in rows if r['date']==date]) for date in DEV_DATES},
                  prediction_sha256=sha256_file(args.predictions),freeze_sha256=sha256_file(args.freeze),
                  evaluator_sha256=sha256_file(__file__),checkpoint_sha256=freeze['checkpoint_sha256'],
                  gt_manifest_sha256=gt_hashes,validation_manifest_sha256=manifests)
    with Path(args.output).open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
