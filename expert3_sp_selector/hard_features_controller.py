import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;store=Path('/data2/zhangcy/sp-selector-20261014-v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.monotonic();procs=[]
for i in range(8):
 log=(store/('hard-feature-%02d.log'%i)).open('wb');p=subprocess.Popen(['/usr/bin/nice','-n','19','/home/zhangcy/.conda/envs/leader/bin/python','-B','-u',str(root/'features.py'),'--group','hard','--shard',str(i),'--of','8'],env=env,stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
(store/'hard-feature-launch.json').write_text(json.dumps(dict(pid=os.getpid(),children=[p.pid for p,l in procs])))
codes=[]
for p,log in procs:codes.append(p.wait());log.close()
(store/'hard-feature-completed.json').write_text(json.dumps(dict(complete=not any(codes),codes=codes,seconds=time.monotonic()-started)))
