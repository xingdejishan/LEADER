import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/teacher_realization'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    with np.load(OUT/'reference.npz') as f:
        target=f['target']; kind=f['kind']; x=f['x']; labels=f['labels']
    correct=kind==2; zero=kind==1
    denominator=sum(float(v@v) for v in target[correct])
    assert denominator>0 and not target[zero].any()
    assert sum(float(v@v) for v in target[correct])/denominator==1.
    for name,r in summary['runs'].items():
        with np.load(OUT/(name+'.npz')) as f:
            delta=f['delta']; scores=f['scores']
        total=0.; hits=0; zero_counts={-1:0,0:0,1:0}
        for i in range(len(kind)):
            v=delta[i]-target[i]
            score=scores[i]/np.sqrt(np.sum((x[i]+delta[i])**2))
            winner=next(j for j in range(16) if score[j]>=max(score)-1e-7)
            if correct[i]:
                total+=float(v@v)
                p=max(score[labels[i]==1]); other=max(score[labels[i]!=1])
                hits+=int(p-other>1e-7)
            else:
                zero_counts[int(labels[i,winner])]+=1
        assert abs(total/denominator-r['R'])<1e-12
        assert hits==r['correct']
        assert zero_counts[-1]==r['zero']['negative_top1'] and zero_counts[0]==r['zero']['gray_top1'] and zero_counts[1]==r['zero']['kept_positive']
        assert sum(zero_counts.values())==2255
    (OUT/'verification.json').write_text(json.dumps(dict(verified=True,model_query_pairs=6*3755,correction_count=1500,zero_count=2255,
        identity_R=1.,description='Independent per-point loops recompute R, strict correction and separate zero-label outcomes from saved applied deltas and scores; no optimizer or new training'),indent=2))
    print('Independent verification passed')


if __name__=='__main__':
    main()
