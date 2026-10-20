import json,os,subprocess,time,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;store=Path('/data2/zhangcy/sp-selector-20261014-v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic();procs=[]
for i in range(8):
 log=(store/('refinement-%02d.log'%i)).open('wb');p=subprocess.Popen(['/usr/bin/nice','-n','19','/home/zhangcy/.conda/envs/leader/bin/python','-B','-u',str(root/'refine_selected.py'),'--shard',str(i),'--of','8'],env=env,stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
(store/'refinement-launch.json').write_text(json.dumps(dict(pid=os.getpid(),children=[p.pid for p,l in procs])));codes=[]
for p,l in procs:codes.append(p.wait());l.close()
if not any(codes):
 rows=[]
 for i in range(8):rows.extend(json.loads((store/('refinement-shard-%02d.json'%i)).read_text())['rows'])
 rows.sort(key=lambda r:r['index']);assert len(rows)==2502;(store/'refinement-predictions.json').write_text(json.dumps(dict(rows=rows)));(store/'refinement-freeze.json').write_text(json.dumps(dict(predictions_sha=hashlib.sha256((store/'refinement-predictions.json').read_bytes()).hexdigest(),GT_online=False)))
(store/'refinement-completed.json').write_text(json.dumps(dict(complete=not any(codes),codes=codes,seconds=time.monotonic()-start)))
