import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;store=Path('/data2/zhangcy/sp-selector-20261014-v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic();procs=[]
for i in range(16):
 log=(store/('control-verify-%02d.log'%i)).open('wb');p=subprocess.Popen(['/usr/bin/nice','-n','19','/home/zhangcy/.conda/envs/leader/bin/python','-B','-u',str(root/'verify_controls_fast.py'),'--shard',str(i),'--of','16'],env=env,stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
(store/'control-verify-launch.json').write_text(json.dumps(dict(pid=os.getpid(),children=[p.pid for p,l in procs])));codes=[]
for p,l in procs:codes.append(p.wait());l.close()
(store/'control-verify-completed.json').write_text(json.dumps(dict(complete=not any(codes),codes=codes,seconds=time.monotonic()-start)))
