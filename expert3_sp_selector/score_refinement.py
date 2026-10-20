from common import *
import importlib.util

assert load(STORE/'refinement-completed.json')['complete'];assert sha(STORE/'refinement-predictions.json')==load(STORE/'refinement-freeze.json')['predictions_sha'];original=load(STORE/'completed.json');assert sha(STORE/'scored.npz')==original['scored_sha'];z=np.load(STORE/'scored.npz');rows=load(STORE/'refinement-predictions.json')['rows'];assert list(z['keys'])==[r['key'] for r in rows];A=Path('/home/zhangcy/projects/v7-moe-oneepoch-20261011-v3')
for n,h in PLAN['reference_assets'].items():assert sha(A/n)==h
spec=importlib.util.spec_from_file_location('policy',A/'nclt_reference_policy.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);result=dict(policy_sha=sha(A/'nclt-reference-policy.json'),groups={});arrays=dict(keys=z['keys'],dates=z['dates'],timestamps_us=z['timestamps_us'],raw=z['raw']);original_report=load(STORE/'report.json')
for group in ['hard','normal']:
 ix=z[group+'_indices'];methods={};bt=z[group+'_V7_T'];br=z[group+'_V7_R'];arrays[group+'_indices']=ix
 for name in ['selector','B','C']:
  record=[];errors=[]
  for field in ['initial','pose']:
   p=np.asarray([rows[int(i)]['methods'][name][field] for i in ix]);stats,t,r,replaced=helper.score_poses(z['dates'][ix],z['timestamps_us'][ix],p,z['raw'][ix],A/'nclt-reference-policy.json');record.append(dict(MPE=float(t.mean()),MOE=float(r.mean()),success=int(((t<1)&(r<2)).sum()),replaced=int(replaced.sum())));errors.append((t,r));suffix='before' if field=='initial' else 'after';arrays[group+'_'+name+'_'+suffix+'_poses']=p;arrays[group+'_'+name+'_'+suffix+'_T']=t;arrays[group+'_'+name+'_'+suffix+'_R']=r
  ref='selector' if name=='selector' else 'rule_relative_v7' if name=='B' else 'rule_relative_v7_margin_0.1';m=original_report['groups'][group]['methods'][ref];assert abs(record[0]['MPE']-m['MPE'])<1e-9 and abs(record[0]['MOE']-m['MOE'])<1e-9
  t0,r0=errors[0];t,r=errors[1];accepted=np.array([rows[int(i)]['methods'][name]['detail']['status']=='accepted' for i in ix]);methods[name]=dict(before=record[0],after=record[1],accepted=int(accepted.sum()),refinement_T_improved=int((t<t0-1e-6).sum()),refinement_T_worsened=int((t>t0+1e-6).sum()),refinement_R_improved=int((r<r0-1e-6).sum()),refinement_R_worsened=int((r>r0+1e-6).sum()),new_success_vs_before=int((~((t0<1)&(r0<2))&(t<1)&(r<2)).sum()),lost_success_vs_before=int(((t0<1)&(r0<2)&~((t<1)&(r<2))).sum()),T_improved_vs_V7=int((t<bt-1e-6).sum()),T_worsened_vs_V7=int((t>bt+1e-6).sum()),new_10m_vs_V7=int(((bt<=10)&(t>10)).sum()))
 result['groups'][group]=methods
np.savez_compressed(STORE/'refinement-scored.npz',**arrays);save(STORE/'refinement-report.json',result);save(STORE/'refinement-score-completed.json',dict(complete=True,report_sha=sha(STORE/'refinement-report.json'),scored_sha=sha(STORE/'refinement-scored.npz')));print(json.dumps(result))
