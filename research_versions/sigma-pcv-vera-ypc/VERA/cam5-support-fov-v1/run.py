import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from common import PLAN,ROOT,STORE,atomic,guard,sha,verify_source

verify_source();os.nice(15)
lock=(STORE/'execution.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
launch=json.loads((STORE/'launch.json').read_text())
parent=Path(PLAN['parent_root'])/'storage';parent_launch=json.loads((parent/'launch.json').read_text())
def alive(pid,ticks):
    try:
        p=psutil.Process(pid)
        return p.uids().real==os.getuid() and int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19])==ticks and p.status()!=psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess,FileNotFoundError):return False
def parent_worker_alive():
    path=parent/'current-stage.json'
    if not path.exists():return False
    record=json.loads(path.read_text())
    try:
        process=psutil.Process(record['pid'])
        return process.uids().real==1003 and process.create_time()==record['process_create_time'] and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
while alive(parent_launch['pid'],parent_launch['start_ticks']) or parent_worker_alive():
    atomic(STORE/'progress.json',{'state':'waiting_parent_pipeline','time':time.time(),'parent_pid':parent_launch['pid']});time.sleep(30)
parent_exit=json.loads((parent/'pipeline-exit.json').read_text()) if (parent/'pipeline-exit.json').exists() else {}
if parent_exit.get('exit')!=0 or not (parent/'vera/epoch-30.pt').exists():
    atomic(STORE/'needs-attention.json',{'reason':'parent did not finish safely; original process not interrupted','parent_exit':parent_exit,'time':time.time()});sys.exit(1)
assert sha(parent/'plan.json')==PLAN['parent_plan_sha256'] and sha(parent/'inputs.json')==PLAN['inputs_sha256']
def choose_gpu():
    candidates=[]
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        g,free,util=[float(x.strip()) for x in line.split(',')]
        if int(g) in PLAN['allowed_gpus'] and free>=PLAN['gpu_initial_free_floor_gib']*1024 and util<=PLAN['gpu_initial_utilization_ceiling']:candidates.append((util,-free,int(g)))
    return min(candidates)[2] if candidates else None
def gpu_guard(gpu,pid):
    rows={int(line.split(',')[0]):float(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    assert gpu in PLAN['allowed_gpus'] and rows[gpu]>=PLAN['gpu_free_floor_gib']*1024,'shared GPU free-memory floor'
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        parts=line.split(',')
        if int(parts[0])==pid:assert float(parts[1])<=PLAN['gpu_process_ceiling_gib']*1024,'private GPU memory ceiling'
stages=[('bind-parent',['bind_parent.py'],False),('checks',['checks.py'],True),('trained-checks',['trained_checks.py'],True)]
for e in PLAN['evaluate_epochs']:
    stages.extend([(f'train-to-{e}',['train_to.py',str(e)],True),(f'diagnose-{e}',['diagnose_epoch.py',str(e)],True),(f'predict-{e}',['predict_epoch_compat.py',str(e)],True),(f'score-{e}',['score_epoch.py',str(e)],False)])
history=json.loads((STORE/'stages.json').read_text()) if (STORE/'stages.json').exists() else []
for name,args,is_gpu in stages:
    if any(r['stage']==name and r['exit']==0 and not r.get('guard_stop') for r in history):continue
    if name.startswith(('predict-','score-')):
        e=int(name.split('-')[-1]);folder=STORE/'eval'/f'e{e:02d}';folder.mkdir(exist_ok=True,parents=True)
        if not (folder/'leader_finetune').exists():(folder/'leader_finetune').symlink_to(STORE/'leader_finetune',target_is_directory=True)
    attempt=0
    while True:
        try:guard();safe=True
        except AssertionError:safe=False
        gpu=choose_gpu() if is_gpu and safe else None
        if not safe or (is_gpu and gpu is None):
            atomic(STORE/'progress.json',{'state':'waiting_safe_resources','stage':name,'time':time.time()});time.sleep(30);continue
        attempt+=1;log=STORE/(name+f'-attempt-{time.time_ns()}.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if is_gpu else '-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            process=psutil.Process(child.pid);start=process.create_time();reason=None;last_gpu=0;last_disk=0
            receipt={'stage':name,'pid':child.pid,'process_create_time':start,'uid':os.getuid(),'gpu':gpu,'log':str(log),'time':time.time(),'attempt':attempt}
            atomic(STORE/'current-stage.json',receipt)
            while child.poll() is None:
                try:
                    guard();rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running());assert rss<PLAN['rss_limit_gib']*1024**3,'worker RSS ceiling'
                    if is_gpu and time.time()-last_gpu>5:gpu_guard(gpu,child.pid);last_gpu=time.time()
                    if time.time()-last_disk>20:
                        total=sum(p.stat().st_size for p in STORE.rglob('*') if p.is_file());assert total<PLAN['output_limit_gib']*1024**3,'private output ceiling';last_disk=time.time()
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
        receipt.update(exit=code,end=time.time(),guard_stop=reason);history.append(receipt);atomic(STORE/'stages.json',history)
        if code==0 and reason is None:break
        tail=log.read_text()[-4000:];recoverable=bool(reason) or 'out of memory' in tail.lower()
        if not recoverable or attempt>=3:
            atomic(STORE/'needs-attention.json',{'stage':name,'exit':code,'guard_stop':reason,'log':str(log),'time':time.time(),'checkpoint_preserved':True});sys.exit(1)
        atomic(STORE/'progress.json',{'state':'resource_retry_from_checkpoint','stage':name,'time':time.time(),'attempt':attempt});time.sleep(60)
reports=[json.loads((STORE/'eval'/f'e{e:02d}'/'report.json').read_text()) for e in PLAN['evaluate_epochs']]
reference=PLAN['selection_reference']
best=min(reports,key=lambda r:(r['results']['vera']['failures'],r['results']['vera']['mpe_m']/reference['mpe_mean_m']+r['results']['vera']['moe_deg']/reference['moe_mean_deg'],r['epoch']))
atomic(STORE/'comparison.json',{'selected_epoch':best['epoch'],'selection':'development selected, not independent test','reports':reports,'not_a_convergence_or_global_optimum_claim':True,'epoch_semantics':PLAN['epoch_semantics'],'parent_binding':json.loads((STORE/'parent-binding.json').read_text()),'remaining_task':'Review convergence, FOV fallback, attention/visual scale and reliability adaptation; freeze same-encoding pose interventions; continue if incomplete, not automatically reject VERA'})
atomic(STORE/'pipeline-exit.json',{'exit':0,'time':time.time(),'comparison_sha256':sha(STORE/'comparison.json'),'requires_research_review':True})
