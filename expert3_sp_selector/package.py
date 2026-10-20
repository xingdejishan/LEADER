from common import *
import zipfile
assert load(STORE/'refinement-score-completed.json')['complete'];assert load(STORE/'controller-completed.json')['complete'];assert load(STORE/'completed.json')['complete'];qs=load(STORE/'queries.json')['queries'];offsets=[0];features=[];poses=[];valid=[];names=[];files={}
for q in qs:
 f=STORE/'features'/('%06d.json'%q['index']);d=load(f);files[str(f)]=sha(f)
 for e in d['entries']:features.append(e['feature']);poses.append(e['pose']);valid.append(e['evidence']['valid']);names.append(e['name'])
 offsets.append(len(features))
np.savez_compressed(STORE/'feature-ledger.npz',features=features,poses=poses,valid=valid,names=names,offsets=offsets,keys=[q['key'] for q in qs]);save(STORE/'feature-freeze.json',dict(files=files,ledger_sha=sha(STORE/'feature-ledger.npz')))
files=['launch.json','queries.json','params.json','input-freeze.json','parent-candidate-freeze.json','train-labels.json','train-baselines.json','cls-freeze.json','geometry-gate.json','model-portability-gate.json','calibration.json','calibration-freeze.json','selector-portable.json','predictions.json','prediction-freeze.json','feature-ledger.npz','feature-freeze.json','report.json','scored.npz','completed.json','controller-completed.json','stages.json','prepare-read-cache-repair.json','hard-pool-parity.json','hard-feature-completed.json','match-cpu-resume.json','control-verify-completed.json','control-verify-resume.json','control-verify-fast-gate.json','extra-feature-completed.json','refinement-config.json','refinement-gate.json','refinement-completed.json','refinement-predictions.json','refinement-freeze.json','refinement-report.json','refinement-scored.npz','refinement-score-completed.json','reference-protocol-manifest.json'];source=load(STORE/'launch.json')['source_hashes']
for n,h in source.items():assert sha(ROOT/n)==h,(n,sha(ROOT/n),h)
with zipfile.ZipFile(STORE/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(STORE/n,n)
 z.write(Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/unselected-opportunities-20261014.json'),'parent-opportunities.json')
 for n in source:z.write(ROOT/n,'source/'+n)
 for f in STORE.glob('feature-map-bindings-*.json'):z.write(f,f.name)
print(json.dumps(dict(path=str(STORE/'evidence.zip'),bytes=(STORE/'evidence.zip').stat().st_size,sha256=sha(STORE/'evidence.zip'))))
