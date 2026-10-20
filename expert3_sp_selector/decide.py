from common import *
import joblib,time

denied=deny_GT();started=time.monotonic();freeze=load(STORE/'calibration-freeze.json');assert sha(STORE/'selector.joblib')==freeze['model_sha'] and sha(STORE/'calibration.json')==freeze['calibration_sha'];model=joblib.load(STORE/'selector.joblib');cal=load(STORE/'calibration.json');qs=load(STORE/'queries.json')['queries'];rows=[]
for q in qs:
 if q['group']=='train':continue
 f=STORE/'features'/('%06d.json'%q['index']);e=load(f)['entries'];b=e[0];valid=[a for a in e[1:] if a['evidence']['valid']];x=np.asarray([np.r_[a['feature'],b['feature'],np.asarray(a['feature'])-b['feature']] for a in valid]);prob=model.predict_proba(x)[:,1] if len(x) else np.zeros(0);k=int(np.argmax(prob)) if len(prob) else None;chosen=valid[k] if k is not None and prob[k]>cal['threshold'] else b
 def score(a):return min(float(np.mean(a['feature'][1:4])),float(np.mean(a['feature'][5:8])))
 cb=max(valid,key=score) if valid else b;continuous=cb if score(cb)>score(b) else b
 rows.append(dict(key=q['key'],index=q['index'],group=q['group'],pose=chosen['pose'],selected=chosen['name'],probability=float(prob[k]) if k is not None else 0.,threshold=cal['threshold'],continuous_pose=continuous['pose'],continuous_selected=continuous['name'],baseline=q['baseline_V7'],feature_sha=sha(f)))
assert not denied;save(STORE/'predictions.json',dict(rows=rows,GT_online_attempts=len(denied)));save(STORE/'prediction-freeze.json',dict(predictions_sha=sha(STORE/'predictions.json'),calibration_freeze_sha=sha(STORE/'calibration-freeze.json'),seconds=time.monotonic()-started));print(json.dumps(dict(complete=True,frames=len(rows),selected=sum(r['selected']!='baseline' for r in rows))))
