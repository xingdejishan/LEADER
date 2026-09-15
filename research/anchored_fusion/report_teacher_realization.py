import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/teacher_realization'
TAU=1e-7


def stats(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    if not len(a):
        return dict(count=0)
    return dict(count=len(a),mean=float(a.mean()),minimum=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def main():
    with np.load(OUT/'reference.npz') as f:
        ref={k:f[k] for k in f.files}
    correction=ref['kind']==2; zero=ref['kind']==1
    lab=ref['labels']; target=ref['target']; x=ref['x']; pos=lab==1; nong=lab!=1; negative=lab==-1
    assert correction.sum()==1500 and zero.sum()==2255
    assert (lab[correction,0]==-1).all() and (lab[zero,0]==1).all() and not target[zero].any()
    tn=np.linalg.norm(target,axis=1)
    indices=np.arange(len(lab)); selected=ref['selected_positive']
    def margin(s,competitors):
        return np.max(np.where(pos,s,-np.inf),axis=1)-np.max(np.where(competitors,s,-np.inf),axis=1)
    teacher_cos=ref['teacher_scores']/ref['teacher_norm'][:,None]
    tm=margin(teacher_cos,nong)
    teacher_fixed=ref['teacher_scores'][indices,selected]-np.max(np.where(nong,ref['teacher_scores'],-np.inf),axis=1)
    assert (tm[correction]>TAU).all()
    summary=dict(correction_count=1500,zero_count=2255,teacher_norm=stats(tn[correction]),teacher_margin=stats(tm[correction]),
        teacher_fixed_margin=stats(teacher_fixed[correction]),teacher_robustness_radius=stats(ref['robustness_radius'][correction]),runs={})
    query_meta=json.loads((OUT/'queries.json').read_text())
    for file in sorted(OUT.glob('aligned_*.npz'))+sorted(OUT.glob('shuffled_*.npz')):
        with np.load(file) as f:
            delta=f['delta']; raw=f['scores']
        assert np.isfinite(delta).all() and np.isfinite(raw).all()
        dn=np.linalg.norm(delta,axis=1); assert dn.max()<.050001
        score=raw/np.linalg.norm(x+delta,axis=1,keepdims=True)
        choice=(score>=score.max(1,keepdims=True)-TAU).argmax(1)
        out=lab[indices,choice]
        best=margin(score,nong); best_negative=margin(score,negative)
        fixed=raw[indices,selected]-np.max(np.where(nong,raw,-np.inf),axis=1)
        squared=np.sum((delta-target)**2,axis=1); target_sq=tn**2
        err=np.sqrt(squared)
        relative=err[correction]/tn[correction]
        amplitude=dn[correction]/tn[correction]
        cosine=np.sum(delta[correction]*target[correction],axis=1)/np.maximum(dn[correction]*tn[correction],1e-300)
        gain=np.sum(delta[correction]*target[correction],axis=1)/target_sq[correction]
        success=best[correction]>TAU; fail=~success
        R=float(squared[correction].sum()/target_sq[correction].sum())
        bins=[]
        edges=[0,.1,.25,.5,1,2,np.inf]
        for left,right in zip(edges[:-1],edges[1:]):
            use=(relative>=left)&(relative<right)
            bins.append(dict(lower=left,upper=None if np.isinf(right) else right,count=int(use.sum()),correct=int(success[use].sum())))
        result=dict(R=R,correct=int(success.sum()),correct_rate=float(success.mean()),top1_counts={str(v):int((out[correction]==v).sum()) for v in [-1,0,1]},
            numerical_boundary=int((np.abs(best[correction])<=TAU).sum()),student_norm=stats(dn[correction]),amplitude_ratio=stats(amplitude),
            relative_error=stats(relative),direction_cosine=stats(cosine),teacher_direction_coefficient=stats(gain),
            near_zero_count=int((amplitude<=.1).sum()),near_cap_count=int((dn[correction]>=.05-1e-6).sum()),
            student_margin=stats(best[correction]),student_fixed_teacher_positive_margin=stats(fixed[correction]),
            failure_margin=stats(best[correction][fail]),teacher_margin_on_failures=stats(tm[correction][fail]),
            error_over_teacher_safe_radius=stats(err[correction]/ref['robustness_radius'][correction]),
            within_sufficient_radius=int((err[correction]<ref['robustness_radius'][correction]).sum()),fit_bins=bins,
            zero=dict(count=2255,kept_positive=int((out[zero]==1).sum()),negative_top1=int((out[zero]==-1).sum()),gray_top1=int((out[zero]==0).sum()),
                boundary=int((np.abs(best[zero])<=TAU).sum()),residual_norm=stats(dn[zero]),squared_norm_mean=float(np.mean(dn[zero]**2))),
            deterministic_full_fit_direction_loss=.5*(float(np.mean(squared[correction]))+float(np.mean(squared[zero])))/.05**2,
            identity_full_fit_direction_loss=.5*float(np.mean(target_sq[correction]))/.05**2)
        result['by_date']={date:dict(count=int(use.sum()),R=float(squared[use].sum()/target_sq[use].sum()),correct=int((best[use]>TAU).sum()))
            for date in sorted({r['date'] for r in query_meta}) for use in [correction&np.array([r['date']==date for r in query_meta])]}
        summary['runs'][file.stem]=result
        np.savez_compressed(OUT/(file.stem+'_metrics.npz'),error=err,student_norm=dn,best_margin=best,negative_margin=best_negative,
            fixed_positive_margin=fixed,top1_label=out,top1_choice=choice,strict_correct=best>TAU)
        print(file.stem,'R',R,'correct',result['correct'],'zero harms',result['zero']['negative_top1'],result['zero']['gray_top1'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
