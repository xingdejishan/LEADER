from common import *
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib,time

def pair(candidate,baseline):
 a=np.asarray(candidate['feature']);b=np.asarray(baseline['feature']);return np.r_[a,b,a-b]
def errors(T,GT):
 T=np.asarray(T);GT=np.asarray(GT);return np.linalg.norm(T[:3,3]-GT[:3,3]),np.rad2deg(np.arccos(np.clip((np.trace(T[:3,:3].T@GT[:3,:3])-1)/2,-1,1)))
def beneficial(a,b):return (a[0]<b[0]-.02 and a[1]<=b[1]+.1) or (a[1]<b[1]-.2 and a[0]<=b[0]+.02)
started=time.monotonic();qs=load(STORE/'queries.json')['queries'];labels={r['key']:r for r in load(STORE/'train-labels.json')['rows']};X=[];y=[];fitkeys=[];calkeys=[]
for q in qs:
 if q['group']!='train':continue
 row=labels[q['key']]
 if row['split']!='fit':calkeys.append(q['key']);continue
 fitkeys.append(q['key']);e=load(STORE/'features'/('%06d.json'%q['index']))['entries'];c=[x for x in e if x['name'].startswith('candidate_')];baselines=[x for x in e if x['name'].startswith('train_baseline_')]
 for b in baselines:
  be=errors(b['pose'],row['pose'])
  for a in c:
   if not a['evidence']['valid']:continue
   X.append(pair(a,b));y.append(int(beneficial(errors(a['pose'],row['pose']),be)))
assert len(set(y))==2 and set(fitkeys).isdisjoint(calkeys);model=HistGradientBoostingClassifier(max_iter=100,max_depth=3,learning_rate=.05,l2_regularization=1.,random_state=20);model.fit(np.asarray(X),np.asarray(y));joblib.dump(model,STORE/'selector.joblib');portable=dict(bias=float(model._baseline_prediction[0,0]),trees=[[dict(value=float(n['value']),feature=int(n['feature_idx']),threshold=float(n['num_threshold']),left=int(n['left']),right=int(n['right']),leaf=bool(n['is_leaf'])) for n in p[0].nodes] for p in model._predictors]);save(STORE/'selector-portable.json',portable)
raw=np.full(len(X),portable['bias'])
for tree in portable['trees']:
 for i,x in enumerate(X):
  k=0
  while not tree[k]['leaf']:n=tree[k];k=n['left'] if x[n['feature']]<=n['threshold'] else n['right']
  raw[i]+=tree[k]['value']
portable_probability=1/(1+np.exp(-raw));diff=float(np.max(np.abs(portable_probability-model.predict_proba(np.asarray(X))[:,1])));assert diff<1e-10;save(STORE/'model-portability-gate.json',dict(complete=True,max_probability_difference=diff,training_pairs=len(X)));cal=[]
for q in qs:
 if q['key'] not in calkeys:continue
 row=labels[q['key']];e=load(STORE/'features'/('%06d.json'%q['index']))['entries'];c=[x for x in e if x['name'].startswith('candidate_') and x['evidence']['valid']];baselines=[x for x in e if x['name'].startswith('train_baseline_')]
 for b in baselines:
  probabilities=model.predict_proba(np.asarray([pair(a,b) for a in c]))[:,1] if c else np.zeros(0);be=errors(b['pose'],row['pose']);wrong=[float(probabilities[i]) for i,a in enumerate(c) if not beneficial(errors(a['pose'],row['pose']),be)];cal.append(dict(key=q['key'],baseline=b['name'],max_wrong_probability=max(wrong or [0.]),candidates=len(c)))
threshold=float(np.quantile([r['max_wrong_probability'] for r in cal],.99,interpolation='higher'))+1e-8
save(STORE/'calibration.json',dict(complete=True,fit_keys=fitkeys,calibration_keys=calkeys,fit_pairs=len(y),fit_positive=int(sum(y)),threshold=threshold,calibration=cal,empirical_false_acceptance=sum(r['max_wrong_probability']>threshold for r in cal),fit_source='Actual cross-date SP+LightGlue candidates, controlled Train GT-perturbed baselines; no actual Train V7 outputs. Normal controls remain necessary.',query_GT_used=False,seconds=time.monotonic()-started));save(STORE/'calibration-freeze.json',dict(model_sha=sha(STORE/'selector.joblib'),portable_sha=sha(STORE/'selector-portable.json'),calibration_sha=sha(STORE/'calibration.json'),params_sha=sha(STORE/'params.json')));print(json.dumps(dict(complete=True,threshold=threshold,pairs=len(y),positive=sum(y),seconds=time.monotonic()-started)))
