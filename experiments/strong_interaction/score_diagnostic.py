import json
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'worker.py').read_text().rsplit('\ntry:\n    main()',1)[0]
exec(compile(source,str(root/'worker.py'),'exec'),globals())

assert json.loads((STORE/'actual-exit.json').read_text())['exit']==0
assert not (STORE/'score-diagnostic.json').exists()
torch.cuda.set_per_process_memory_fraction(.32)
data=json.loads((STORE/'manifest.json').read_text())
thresholds=[0.,.001,.01,.03,.1]
results=[]
with torch.no_grad():
    for arm in ['v8','collaborative']:
        m=make_model()
        if arm=='collaborative':
            frozen=json.loads((STORE/'checkpoint-freeze.json').read_text())[arm]
            assert sha(frozen['path'])==frozen['sha']
            m.load_state_dict(torch.load(frozen['path'],map_location='cpu')['model'],strict=True)
        m.eval()
        for count in [3,10]:
            records=[]
            original={r['q']:r for r in json.loads((STORE/('%s-%d.json'%(arm,count))).read_text())['records']}
            for number,g in enumerate(data['test'],1):
                refs=g['refs'][:count]
                x=tensors(g['q'],refs,geometry(refs))
                scores=m(*x)[0] if arm=='collaborative' else independent_scores(m,x[0],x[1])
                matched,_,confidence,_=filter_matches(scores,0.)
                pair_records=[]
                for i,r in enumerate(refs):
                    qi=torch.where(matched[i]>=0)[0]
                    pairs=torch.stack([qi,matched[i,qi]],-1).cpu().numpy()
                    pair_records.append(dict(reference=r,**assess(g['q'],r,pairs,confidence[i,qi].cpu().numpy())))
                record=dict(q=g['q'],refs=refs,pairs=pair_records,thresholds={})
                for threshold in thresholds:
                    selected=[p for pair in pair_records for p in pair['predictions'] if p['confidence']>threshold]
                    counts=dict(matches=len(selected),evaluable=sum(p['evaluable'] for p in selected),
                                correct=sum(p['correct'] for p in selected),
                                unique_correct=len({p['qi'] for p in selected if p['correct']}))
                    record['thresholds'][str(threshold)]=counts
                for pair,previous in zip(pair_records,original[g['q']]['pairs']):
                    selected=[p for p in pair['predictions'] if p['confidence']>.1]
                    assert [(p['qi'],p['ri']) for p in selected]==[(p['qi'],p['ri']) for p in previous['predictions']]
                records.append(record)
                del scores,x
                if number%8==0:
                    progress('score_diagnostic',arm=arm,references=count,queries=number,total=48)
            summaries=[]
            for threshold in thresholds:
                key=str(threshold)
                total={k:sum(r['thresholds'][key][k] for r in records) for k in ['matches','evaluable','correct','unique_correct']}
                summaries.append(dict(threshold=threshold,**total,precision=total['correct']/max(1,total['evaluable'])))
            results.append(dict(arm=arm,references=count,summaries=summaries,records=records))
        del m
        torch.cuda.empty_cache()
save('score-diagnostic.json',dict(complete=True,results=results,thresholds=thresholds,
                                note='post-result calibration diagnostic only; primary .1 result unchanged; no model tuning',
                                primary_predictions_exact=True,elapsed=time.monotonic()-BEGIN))
