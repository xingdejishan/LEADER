import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT,STORE,PLAN,atomic,sha,guard,verify_source
import psutil

verify_source();guard()
launch=json.loads((STORE/'launch.json').read_text());gpu=launch['gpu']
def idle_gpu():
    def snapshot():
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        busy={line.split(',')[0].strip() for line in apps.splitlines()}
        cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        return {int(line.split(',')[0]) for line in cards.splitlines() if line.split(',')[1].strip() not in busy and float(line.split(',')[2])>12*1024}
    first=snapshot();time.sleep(1);second=snapshot();available=sorted(first&second)
    assert available,'No idle GPU; worker not launched'
    return available[0]
stages=[('wait-prepared',['wait_prepared.py']),('checks',['checks.py']),('train-lidar',['train.py','lidar_finetune']),('train-vera',['train.py','vera']),('predict-l0',['predict.py','l0']),('predict-lidar',['predict.py','lidar_finetune']),('predict-vera',['predict.py','vera']),('score',['score.py'])]
phase=sys.argv[1] if len(sys.argv)>1 else 'all'
tag=sys.argv[2] if len(sys.argv)>2 else ''
if phase=='wait-checks':stages=stages[:2]
elif phase=='checks-only':stages=stages[1:2]
elif phase=='diagnostic':stages=[('cat-diagnostic',['cat_diagnostic.py'])]
elif phase=='train-eval':
    assert (STORE/'checks.json').exists()
    stages=stages[2:]
history=json.loads((STORE/'stages.json').read_text()) if (STORE/'stages.json').exists() else []
for name,args in stages:
    if name=='predict-l0' and (STORE/'l0/prediction-freeze.json').exists():continue
    assert PLAN['baseline_training_enabled'] and PLAN['baseline_coordinate_alignment']=='original_main_unchanged'
    if name not in ('score','wait-prepared'):
        gpu=idle_gpu()
        atomic(STORE/'gpu-stage-selection.json',{'gpu':gpu,'stage':name,'time':time.time()})
    guard();environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1' if name in ('score','wait-prepared') else str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    log=STORE/(name+tag+'.log')
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid);start=process.create_time()
        receipt={'stage':name,'pid':child.pid,'process_create_time':start,'uid':os.getuid(),'time':time.time()}
        atomic(STORE/'current-stage.json',receipt)
        reason=None;last_disk=0
        while child.poll() is None:
            try:
                guard()
                rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())
                assert rss<PLAN['rss_limit_gib']*1024**3,'worker memory ceiling'
                if time.time()-last_disk>20:
                    total=sum(p.stat().st_size for p in STORE.rglob('*') if p.is_file())
                    assert total<PLAN['output_limit_gib']*1024**3,'private output budget'
                    last_disk=time.time()
            except (psutil.NoSuchProcess,FileNotFoundError):
                if child.poll() is not None:break
            except (AssertionError,RuntimeError) as error:
                reason=str(error)
                if process.is_running() and process.create_time()==start and process.uids().real==os.getuid():
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill()
                break
            time.sleep(1)
        code=child.wait()
    receipt.update({'exit':code,'end':time.time(),'guard_stop':reason});history.append(receipt);atomic(STORE/'stages.json',history)
    if code!=0 or reason:
        atomic(STORE/'pipeline-exit.json',{'exit':code,'failed_stage':name,'reason':reason,'time':time.time()})
        sys.exit(code or 1)
atomic(STORE/'pipeline-exit.json',{'exit':0,'phase':phase,'time':time.time(),'report_sha256':sha(STORE/'report.json') if (STORE/'report.json').exists() else None})
