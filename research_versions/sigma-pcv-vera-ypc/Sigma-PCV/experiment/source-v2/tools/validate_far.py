import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.magic_nclt_runtime import safe_json_dump,sha256_file


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','run','gt-root'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    run=Path(args.run)
    root=Path(__file__).resolve().parents[1]
    run.mkdir(exist_ok=True)
    (run/'results').mkdir(exist_ok=True)
    def evaluate(directory,checkpoint=None,baseline=False):
        with (run/'validation.log').open('a') as stream:
            command=[sys.executable,'-B','-u','-m','tools.predict_far','--config',args.config,'--output',str(directory)]
            command+=['--baseline'] if baseline else ['--checkpoint',str(checkpoint)]
            subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
            command=[sys.executable,'-B','-u','-m','tools.evaluate_far','--config',args.config,'--predictions',str(directory/'predictions.jsonl'),
                '--freeze',str(directory/'prediction_freeze.json'),'--gt-root',args.gt_root,'--output',str(directory/'evaluation.json')]
            command+=['--baseline'] if baseline else ['--checkpoint',str(checkpoint)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
            subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return json.loads((directory/'evaluation.json').read_text())
    try:
        baseline=evaluate(run/'baseline',baseline=True)
        b=baseline['overall']
        assert b['frames']==1200 and b['failures']==0
        assert abs(b['mpe_mean_m']-.20252501894885627)<1e-4
        assert abs(b['moe_mean_deg']-1.666125511210262)<1e-3
        safe_json_dump(run/'baseline_verified.json',dict(evaluation_sha256=sha256_file(run/'baseline/evaluation.json'),metrics=b))
        selection={name:None for name in ('mpe','moe','combined')}
        safe_json_dump(run/'selection.json',selection)
        while True:
            pending=[p for p in sorted((run/'queue').glob('*.json')) if not (run/'results'/p.name).exists()]
            if not pending:
                if (run/'training_done.json').exists():
                    safe_json_dump(run/'completion.json',dict(epochs=config['epochs'],best=selection))
                    return
                descriptor=run/'training_worker.json'
                if descriptor.exists():
                    end=Path(json.loads(descriptor.read_text())['directory'])/'exit.json'
                    if end.exists():
                        safe_json_dump(run/'validation_idle_after_training_exit.json',json.loads(end.read_text()))
                        return
                time.sleep(5)
                continue
            job=json.loads(pending[0].read_text())
            checkpoint=Path(job['checkpoint'])
            assert sha256_file(checkpoint)==job['checkpoint_sha256']
            assert sha256_file(run/'settings.json')==job['settings_sha256']==sha256_file(args.config)
            directory=run/'validation'/('{:02d}'.format(job['epoch']))
            safe_json_dump(run/'validation_progress.json',dict(epoch=job['epoch'],phase='validation-started'))
            result=evaluate(directory,checkpoint=checkpoint)
            metrics=result['overall']
            record=dict(epoch=job['epoch'],checkpoint_sha256=job['checkpoint_sha256'],evaluation_sha256=sha256_file(directory/'evaluation.json'),metrics=metrics)
            for name in selection:
                score=metrics['mpe_mean_m'] if name=='mpe' else metrics['moe_mean_deg'] if name=='moe' else .5*(metrics['mpe_mean_m']/b['mpe_mean_m']+metrics['moe_mean_deg']/b['moe_mean_deg'])
                if selection[name] is None or score<selection[name]['score']:
                    selection[name]=dict(record,score=score)
            safe_json_dump(run/'selection.json',selection)
            safe_json_dump(run/'results'/pending[0].name,record)
            safe_json_dump(run/'validation_progress.json',dict(epoch=job['epoch'],phase='completed',metrics=metrics))
    except Exception as error:
        safe_json_dump(run/'validation_failed.json',dict(error=repr(error),time=time.time()))
        raise


if __name__=='__main__':main()
