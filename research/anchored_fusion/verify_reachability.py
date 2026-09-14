import json
from pathlib import Path
from collections import Counter
import numpy as np

OUT = Path(__file__).resolve().parent/'results/reachability'


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    records = json.loads((OUT/'records.json').read_text())
    certs = json.loads((OUT/'certificates.json').read_text())
    pad,tau = protocol['bound_padding'],protocol['tau']
    grouped = {}
    witness_labels = []
    gaps = []
    label_file=OUT/'candidate_labels.json'
    candidate_labels = {k:np.array(v) for k,v in json.loads(label_file.read_text()).items()} if label_file.exists() else {}
    for r in records:
        if r['frame_id'] not in candidate_labels:
            with np.load(Path('/home/zhang/anchored-contrastive-fusion/candidates')/(r['frame_id']+'.npz')) as f:
                candidate_labels[r['frame_id']] = np.where(f['positive'],'P',np.where(f['negative'],'N','G'))
    label_file.write_text(json.dumps({k:v.tolist() for k,v in candidate_labels.items()}))
    with np.load(OUT/'witnesses.npz') as z:
        for c in certs:
            rid,key = str(c['record']),c['key']
            x,k,delta,w = z['x'+rid],z['k'+rid],z['d'+key],z['w'+key]
            assert np.isfinite(delta).all() and np.isfinite(w).all()
            assert np.linalg.norm(delta)<=c['rho']+1e-15
            assert w.min()>=0 and abs(w.sum()-1)<1e-14
            assert abs(np.linalg.norm(x)-1)<1e-12
            a = k[c['positive']]-k[c['competitors']]
            margins = np.sum((x+delta)[None]*a,axis=1)
            av = np.sum(w[:,None]*a,axis=0)
            lo = float(margins.min()-pad)
            hi = float(np.sum(x*av)+c['rho']*np.sqrt(np.sum(av*av))+pad)
            assert abs(lo-c['lower'])<1e-13 and abs(hi-c['upper'])<1e-13
            assert lo<=hi+1e-13
            s = k @ (x+delta)
            choice = int(np.flatnonzero(s>=s.max()-tau)[0])
            rec = records[c['record']]
            labels = candidate_labels[rec['frame_id']][rec['candidate_row']]
            assert labels[c['positive']]=='P'
            assert c['competitors']==np.flatnonzero(labels=='N' if c['level']=='N' else labels!='P').tolist()
            witness_labels.append(dict(key=key,record=c['record'],level=c['level'],choice=choice,
                label=str(labels[choice]),positive_witness_margin=lo,global_classification=rec['classification']))
            grouped.setdefault((c['record'],c['level']),[]).append((lo,hi))
            gaps.append(hi-lo)
        for i,r in enumerate(records):
            if 'bounds' not in r:
                continue
            for level in ['N','NG']:
                values = np.array(grouped[i,level])
                assert np.allclose(values.max(0),r['bounds'][level],atol=1e-13,rtol=0)
                labels=candidate_labels[r['frame_id']][r['candidate_row']]
                assert len(values)==int((labels=='P').sum())
            ln,un=r['bounds']['N']; lf,uf=r['bounds']['NG']
            label='reachable' if lf>tau else 'negative_blocked' if un < -tau else 'gray_blocked' if ln>tau and uf < -tau else 'unresolved'
            assert label==r['classification']
    summary = dict(queries=len(records),certificates=len(certs),verified=True,max_gap=max(gaps),median_gap=float(np.median(gaps)),groups={})
    for original in ['N','G']:
        population=[r for r in records if r['original']==original and r['visual']=='P']
        for group in ['all','editable','protected']:
            subset=[r for r in population if group=='all' or r['protected']==(group=='protected')]
            reach=[r for r in subset if r['classification']=='reachable']
            summary['groups'][original+'_'+group]=dict(count=len(subset),classification=dict(Counter(r['classification'] for r in subset)),
                models={name:dict(strict_correct=sum(r['models'][name]['strict_correct'] for r in subset),
                    correct_on_reachable=sum(r['models'][name]['strict_correct'] for r in reach),
                    gray=sum(r['models'][name]['label']=='G' for r in subset),boundary=sum(r['models'][name]['boundary'] for r in subset)) for name in records[0]['models']})
    summary['by_date']={date:dict(count=len(subset),classification=dict(Counter(r['classification'] for r in subset))) for date in sorted({r['date'] for r in records})
        for subset in [[r for r in records if r['date']==date and r['original']=='N' and r['visual']=='P']]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'witness_rankings.json').write_text(json.dumps(witness_labels,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
