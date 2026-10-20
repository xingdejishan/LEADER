from common import *
import importlib.util,time

POLICY=Path('/home/zhangcy/projects/v7-moe-oneepoch-20261011-v3');SP=Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1');V7=Path('/data2/zhangcy/v7-full-20261006/epoch10-evaluation-20261009/evaluation-epoch10');assert sha(STORE/'predictions.json')==load(STORE/'prediction-freeze.json')['predictions_sha']
while not (STORE/'control-verify-completed.json').exists():time.sleep(2)
assert load(STORE/'control-verify-completed.json')['complete']
for n,h in PLAN['reference_assets'].items():assert sha(POLICY/n)==h
spec=importlib.util.spec_from_file_location('policy',POLICY/'nclt_reference_policy.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);rows=load(STORE/'predictions.json')['rows'];dates=np.array([r['key'].split('/')[0] for r in rows]);times=np.array([int(r['key'].split('/')[1]) for r in rows]);truth={};raw=[]
scratch_path=Path('/home/zhangcy/projects/LEADER-v7-scratch-full-20261008-v2/experiment-plan.json');assert sha(scratch_path)=='479f5652dba74628635f7a5c77be88f718b2863eac29b94bbca603cf20f585d8';scratch=load(scratch_path)
for date in np.unique(dates):
 truth_path=DATA/'evaluator-only-truth'/(date+'.json');assert sha(truth_path)==scratch['truth_hashes'][date];truth[date]=load(truth_path)
for r in rows:date,ut=r['key'].split('/');raw.append(truth[date][ut]['T_WB'])
raw=np.asarray(raw);old=np.load(SP/'scored-methods.npz');oldpos={str(k):i for i,k in enumerate(old['keys'])};report=dict(frames=len(rows),groups={},reference_policy_sha=sha(POLICY/'nclt-reference-policy.json'));arrays=dict(keys=np.array([r['key'] for r in rows]),dates=dates,timestamps_us=times,raw=raw)
for group in ['hard','normal']:
 ix=np.array([i for i,r in enumerate(rows) if r['group']==group]);methods={n:np.asarray([rows[i][field] for i in ix]) for n,field in [('V7','baseline'),('selector','pose'),('continuous_B','continuous_pose')]};summaries={};errors={}
 if group=='normal':
  oldB=[];oldC=[]
  for i in ix:
   record=load(STORE/'verify_controls'/('%06d.json'%rows[i]['index']))['poses'];c=[x for n,x in record.items() if n.startswith('candidate_')];best=max(c,key=lambda x:x['geometry_score'] or 0.) if c else None;bs=record['baseline_V7']['geometry_score'] or 0.;oldB.append(best['pose'] if best and (best['geometry_score'] or 0.)>bs else rows[i]['baseline']);oldC.append(best['pose'] if best and (best['geometry_score'] or 0.)>bs+.1 else rows[i]['baseline'])
  methods['rule_relative_v7']=np.asarray(oldB);methods['rule_relative_v7_margin_0.1']=np.asarray(oldC)
 for n,poses in methods.items():
  stats,t,r,replaced=helper.score_poses(dates[ix],times[ix],poses,raw[ix],POLICY/'nclt-reference-policy.json');errors[n]=(t,r);summaries[n]=dict(MPE=float(t.mean()),MOE=float(r.mean()),success=int(((t<1)&(r<2)).sum()),replaced=int(replaced.sum()));arrays[group+'_'+n+'_poses']=poses;arrays[group+'_'+n+'_T']=t;arrays[group+'_'+n+'_R']=r
 bt,br=errors['V7'];comparison={}
 for n in [n for n in methods if n!='V7']:
  t,r=errors[n];comparison[n]=dict(selected=int(np.any(np.abs(methods[n]-methods['V7'])>1e-12,axis=(1,2)).sum()),T_improved=int((t<bt-1e-6).sum()),T_worsened=int((t>bt+1e-6).sum()),R_improved=int((r<br-1e-6).sum()),R_worsened=int((r>br+1e-6).sum()),new_success=int((~((bt<1)&(br<2))&(t<1)&(r<2)).sum()),lost_success=int(((bt<1)&(br<2)&~((t<1)&(r<2))).sum()),new_10m=int(((bt<=10)&(t>10)).sum()))
 if group=='normal':assert np.all(bt<=.5+1e-8) and np.all(br<2)
 if group=='hard':
  oi=np.array([oldpos[rows[i]['key']] for i in ix]);assert np.max(np.abs(old['baseline_V7'][oi,0]-bt))<1e-8 and np.max(np.abs(old['baseline_V7'][oi,1]-br))<1e-8
  for n in ['rule_relative_v7','rule_relative_v7_margin_0.1']:t,r=old[n][oi].T;summaries[n]=dict(MPE=float(t.mean()),MOE=float(r.mean()),success=int(((t<1)&(r<2)).sum()))
 report['groups'][group]=dict(frames=len(ix),methods=summaries,comparisons=comparison);arrays[group+'_indices']=ix
np.savez_compressed(STORE/'scored.npz',**arrays);save(STORE/'report.json',report);save(STORE/'completed.json',dict(complete=True,report_sha=sha(STORE/'report.json'),scored_sha=sha(STORE/'scored.npz')));print(json.dumps(report))
