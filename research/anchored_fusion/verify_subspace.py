import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/subspace'
SOURCE=HERE/'results/reachability'


def classification(ln,un,lf,uf,tau):
    if lf>tau:
        return 'reachable'
    if un < -tau:
        return 'negative_blocked'
    if ln>tau and uf < -tau:
        return 'gray_blocked'
    return 'unresolved'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    tau,pad=protocol['tau'],protocol['pad']
    for file,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((SOURCE/file).read_bytes()).hexdigest()==digest
    original=json.loads((SOURCE/'records.json').read_text())
    labels=json.loads((SOURCE/'candidate_labels.json').read_text())
    expected=[i for i,r in enumerate(original) if 'bounds' in r]
    with np.load(SOURCE/'witnesses.npz') as z:
        vectors={i:(z['x'+str(i)],z['k'+str(i)]) for i in expected}
    summary={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        rows=json.loads((folder/'records.json').read_text())
        certs=json.loads((folder/'certificates.json').read_text())
        basis=json.loads((folder/'basis.json').read_text())
        with np.load(folder/'basis.npz') as f:
            q,matrix=f['Q'],f['matrix']
        sv=np.linalg.svd(matrix,compute_uv=False)
        assert q.shape==(512,33)
        assert sv[-1]>1e-12*sv[0] and basis['accepted']
        assert np.linalg.norm(q.T@q-np.eye(33),2)<1e-12
        assert np.linalg.norm(matrix-q@(q.T@matrix),2)/np.linalg.norm(matrix,2)<1e-12
        assert [r['record'] for r in rows]==expected
        grouped={}; gaps=[]; witness_rows=[]
        with np.load(folder/'witnesses.npz') as z:
            for c in certs:
                rid,key=c['record'],c['key']
                x,k=vectors[rid]; old=original[rid]
                lab=np.array(labels[old['frame_id']][old['candidate_row']])
                competitor=np.flatnonzero(lab=='N' if c['level']=='N' else lab!='P')
                assert lab[c['positive']]=='P' and competitor.tolist()==c['competitors']
                coefficients,w=z['c'+key],z['w'+key]
                delta=np.sum(q*coefficients[None],axis=1)
                assert np.isfinite(delta).all() and np.isfinite(w).all()
                assert np.linalg.norm(coefficients)<=c['rho']+1e-15 and np.linalg.norm(delta)<=c['rho']+1e-15
                assert w.min()>=0 and abs(w.sum()-1)<1e-14
                a=k[c['positive']]-k[competitor]
                lower=float(np.sum(a*(x+delta)[None],axis=1).min()-pad)
                average=np.sum(w[:,None]*a,axis=0)
                upper=float(np.sum(x*average)+c['rho']*np.linalg.norm(q.T@average)+pad)
                assert abs(lower-c['lower'])<1e-13 and abs(upper-c['upper'])<1e-13
                assert lower<=upper+1e-13
                grouped.setdefault((rid,c['level']),[]).append((c['positive'],lower,upper))
                gaps.append(upper-lower)
                scores=k@(x+delta)
                choice=int(np.flatnonzero(scores>=scores.max()-tau)[0])
                witness_rows.append(dict(key=key,record=rid,level=c['level'],choice=choice,label=str(lab[choice])))
            for r in rows:
                rid=r['record']; old=original[rid]; x,k=vectors[rid]
                lab=np.array(labels[old['frame_id']][old['candidate_row']])
                for level in ['N','NG']:
                    values=grouped[rid,level]
                    assert sorted(v[0] for v in values)==np.flatnonzero(lab=='P').tolist()
                    lo=max(v[1] for v in values); hi=max(v[2] for v in values)
                    assert np.allclose([lo,hi],r['bounds'][level],atol=1e-13,rtol=0)
                    assert lo<=old['bounds'][level][1]+1e-12
                assert classification(*r['bounds']['N'],*r['bounds']['NG'],tau)==r['classification']
                actual=z['actual'+str(rid)]; feasible=z['feasible'+str(rid)]
                rho=0. if old['protected'] else .05
                assert np.linalg.norm(feasible)<=rho+1e-15 and np.linalg.norm(q@feasible)<=rho+1e-15
                projection_error=np.linalg.norm(actual-q@(q.T@actual))
                assert projection_error<=1e-6 and np.linalg.norm(actual)<=rho+1e-6
                s=k@(x+actual)/np.linalg.norm(x+actual)
                margin=float(s[lab=='P'].max()-s[lab!='P'].max())
                assert abs(margin-r['actual_margin'])<1e-12
                assert (margin>tau)==old['models'][folder.name]['strict_correct']==r['actual_correct']
                s=k@(x+q@feasible)
                margin=float(s[lab=='P'].max()-s[lab!='P'].max()-pad)
                assert abs(margin-r['feasible_actual_margin'])<1e-12
                if r['actual_correct']:
                    assert margin>tau and r['classification']=='reachable'
        groups={}
        for label in ['N','G']:
            for group in ['all','editable','protected','full_reachable']:
                chosen=[r for r in rows if r['original']==label and (group=='all' or group=='editable' and not r['protected'] or group=='protected' and r['protected'] or group=='full_reachable' and r['full_classification']=='reachable')]
                groups[label+'_'+group]=dict(count=len(chosen),classification=dict(Counter(r['classification'] for r in chosen)),actual_correct=sum(r['actual_correct'] for r in chosen))
        by_date={date:dict(count=len(chosen),classification=dict(Counter(r['classification'] for r in chosen)),actual_correct=sum(r['actual_correct'] for r in chosen))
            for date in sorted({r['date'] for r in rows}) for chosen in [[r for r in rows if r['date']==date and r['original']=='N' and r['full_classification']=='reachable']]}
        summary[folder.name]=dict(verified=True,rank=33,minimum_singular_value=float(sv[-1]),condition=float(sv[0]/sv[-1]),certificates=len(certs),
            max_gap=max(gaps),median_gap=float(np.median(gaps)),max_actual_span_error=max(r['actual_span_error'] for r in rows),max_actual_norm_excess=max(r['actual_norm_excess'] for r in rows),groups=groups,by_date=by_date)
        (folder/'witness_rankings.json').write_text(json.dumps(witness_rows,indent=2))
        print(folder.name,json.dumps(groups['N_full_reachable']),flush=True)
    assert len(summary)==6
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
