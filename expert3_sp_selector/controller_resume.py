import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;STORE=Path('/data2/zhangcy/sp-selector-20261014-v1');PY='/home/zhangcy/.conda/envs/leader/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');started=time.monotonic();stages=[]
def run(stage,workers=1,gpu=False):
 processes=[];(STORE/'controller-progress.json').write_text(json.dumps(dict(stage=stage,started=started,pid=os.getpid())))
 for i in range(workers):
  args=[PY,'-B','-u',str(ROOT/(stage+'.py'))]
  if stage=='cls':args.extend([str(i),str(workers)])
  elif workers>1:args.extend(['--shard',str(i),'--of',str(workers)])
  log=(STORE/(stage+'-%02d.log'%i)).open('wb');e=dict(env)
  if gpu:e['CUDA_VISIBLE_DEVICES']='2'
  processes.append((subprocess.Popen(['/usr/bin/nice','-n','19']+args,env=e,stdout=log,stderr=subprocess.STDOUT),log))
 start=time.monotonic();codes=[]
 for proc,log in processes:codes.append(proc.wait());log.close()
 stages.append(dict(stage=stage,workers=workers,codes=codes,seconds=time.monotonic()-start));(STORE/'stages.json').write_text(json.dumps(stages))
 if any(codes):
  for i in range(workers):print((STORE/(stage+'-%02d.log'%i)).read_text()[-3500:],flush=True)
  raise RuntimeError(stage+' failed')
try:
 for stage,workers,gpu in [('match',16,False),('features',8,False),('calibrate',1,False),('decide',1,False),('score',1,False)]:run(stage,workers,gpu)
 (STORE/'controller-completed.json').write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-started,stages=stages)))
except Exception as e:
 (STORE/'failure.json').write_text(json.dumps(dict(error=repr(e),seconds=time.monotonic()-started,stages=stages)));raise
