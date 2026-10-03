import argparse,json,os,subprocess,sys,time
from pathlib import Path

import torch

from models.far_covolume_standard import CoVolumeStandardFAR as CoVolumeFAR
from tools.far_covolume_budget import setup,budget
from tools.far_runtime import verify_sources
from tools.magic_nclt_runtime import safe_json_dump,sha256_file
from tools.predict_far_covolume_standard import predict


def evaluate_epoch(config_path,run,job,model):
    config_path=Path(config_path)
    run=Path(run)
    config=json.loads(config_path.read_text())
    epoch=job['epoch']
    checkpoint=Path(job['checkpoint'])
    assert sha256_file(checkpoint)==job['checkpoint_sha256']
    assert sha256_file(run/'settings.json')==job['settings_sha256']
    assert json.loads((run/'settings.json').read_text())==config
    directory=run/'validation'/('%02d'%epoch)
    if directory.exists():
        assert not (run/'results'/('%02d.json'%epoch)).exists()
        archive=run/'interrupted-validation'
        archive.mkdir(exist_ok=True)
        directory.rename(archive/('%02d-%d'%(epoch,time.time_ns())))
    safe_json_dump(run/'validation_progress.json',dict(epoch=epoch,phase='validation-started'))
    freeze=predict(config,directory,checkpoint,far=model)
    freeze['settings_sha256']=sha256_file(config_path)
    assert freeze['frames']==1200 and freeze['failures']==0
    safe_json_dump(directory/'prediction_freeze.json',freeze)
    command=[sys.executable,'-B','-u','-m','tools.evaluate_far_covolume_standard','--config',str(config_path),'--predictions',str(directory/'predictions.jsonl'),
        '--freeze',str(directory/'prediction_freeze.json'),'--gt-root',config['gt_root'],'--output',str(directory/'evaluation.json'),'--checkpoint',str(checkpoint)]
    with (run/'validation.log').open('a') as stream:
        subprocess.run(command,cwd=str(Path(__file__).resolve().parents[1]),env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    evaluation=json.loads((directory/'evaluation.json').read_text())
    metrics=evaluation['overall']
    assert metrics['frames']==1200 and metrics['failures']==0
    record=dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),evaluation_sha256=sha256_file(directory/'evaluation.json'),metrics=metrics)
    safe_json_dump(run/'results'/('%02d.json'%epoch),record)
    safe_json_dump(run/'validation_progress.json',dict(epoch=epoch,phase='completed',metrics=metrics))
    return record


def select(config,run):
    assert sha256_file(config['baseline_evaluation'])==config['baseline_evaluation_sha256']
    baseline=json.loads(Path(config['baseline_evaluation']).read_text())['overall']
    selection={name:None for name in ('mpe','moe','combined')}
    for path in sorted((run/'results').glob('*.json')):
        record=json.loads(path.read_text())
        metrics=record['metrics']
        for name in selection:
            score=metrics['mpe_mean_m'] if name=='mpe' else metrics['moe_mean_deg'] if name=='moe' else .5*(metrics['mpe_mean_m']/baseline['mpe_mean_m']+metrics['moe_mean_deg']/baseline['moe_mean_deg'])
            if selection[name] is None or score<selection[name]['score']:selection[name]=dict(record,score=score)
    safe_json_dump(run/'selection.json',selection)
    return selection


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','run'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--one-epoch',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    run=Path(args.run)
    assert run.exists()
    setup(run)
    verify_sources(config)
    for name in ('queue','results','validation'):(run/name).mkdir(exist_ok=True)
    model=CoVolumeFAR(**config['far_covolume']).to(torch.device('cuda:0')).eval()
    try:
        while True:
            budget(run)
            pending=[p for p in sorted((run/'queue').glob('*.json')) if not (run/'results'/p.name).exists()]
            if not pending:
                selection=select(config,run)
                if (run/'training_done.json').exists():
                    assert {p.stem for p in (run/'results').glob('*.json')}=={'%02d'%epoch for epoch in range(1,config['epochs']+1)}
                    safe_json_dump(run/'completion.json',dict(epochs=config['epochs'],best=selection))
                    return
                if args.one_epoch:return
                time.sleep(5)
                continue
            job=json.loads(pending[0].read_text())
            saved=torch.load(job['checkpoint'],map_location='cpu')
            assert saved['config']==config and saved['epoch']==job['epoch']
            model.load_state_dict(saved['model'],strict=True)
            del saved
            evaluate_epoch(args.config,run,job,model)
            selection=select(config,run)
            if args.one_epoch:
                if (run/'training_done.json').exists() and len(list((run/'results').glob('*.json')))==config['epochs']:
                    safe_json_dump(run/'completion.json',dict(epochs=config['epochs'],best=selection))
                return
    except Exception as error:
        safe_json_dump(run/'validation_failed.json',dict(error=repr(error),time=time.time()))
        raise


if __name__=='__main__':main()
