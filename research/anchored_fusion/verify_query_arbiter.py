import json

import numpy as np
import torch

import query_arbiter as a


def main():
    p=json.loads((a.OUT/'protocol.json').read_text()); folds=json.loads((a.OUT/'folds.json').read_text()); rows={r['frame_id']:r for r in p['rows']}
    held=[]
    for fold in folds:
        train=set(fold['train_frames']); test=set(fold['held_frames']); assert not train&test
        assert all(rows[f]['role']=='fit' for f in train|test)
        assert all(p['fold'][f]==fold['fold'] for f in test)
        for t in train:
            assert all(rows[t]['session_id']!=rows[h]['session_id'] or abs(int(t)-int(h))>10000000 for h in test)
        held.extend(test)
    assert len(held)==len(set(held))==578
    assert set(held)=={f for f,r in rows.items() if r['role']=='fit'}
    model=a.Arbiter(); model.load_state_dict(torch.load(a.OUT/'best.pt',map_location='cpu')); assert sum(t.numel() for t in model.parameters())==129
    scores={}
    for name in ['oof','internal','development']:
        with np.load(a.OUT/(name+'.npz')) as f:
            l=np.exp(f['L'].astype(float)); v=np.exp(f['V'].astype(float)); ls=np.sort(l,axis=1); vs=np.sort(v,axis=1)
            x=np.column_stack((ls[:,-1]-ls[:,-2],-(l*f['L']).sum(1)/np.log(25),vs[:,-1]-vs[:,-2],-(v*f['V']).sum(1)/np.log(25),(l*v).sum(1),l.argmax(1)!=v.argmax(1))).astype(np.float32)
            assert np.array_equal(x,f['features'])
            benefit=(l.argmax(1)!=f['target'])&(v.argmax(1)==f['target']); harm=(l.argmax(1)==f['target'])&(v.argmax(1)!=f['target'])
            assert np.array_equal(benefit,f['benefit']) and np.array_equal(harm,f['harm'])
            with torch.no_grad(): score=model(torch.tensor(x)).sigmoid().numpy()
            if name=='development': assert np.allclose(score,f['score'],atol=1e-6)
            scores[name]=(score,x,benefit,harm)
    choices=json.loads((a.OUT/'internal_grid.json').read_text()); chosen=json.loads((a.OUT/'selection.json').read_text())
    expected=max(choices,key=lambda r:(r['net'],-r['overrides'],-r['epoch'],r['threshold']))
    assert chosen==expected
    score,x,b,h=scores['internal']; use=(score>=chosen['threshold'])&(x[:,-1]>0)
    assert int((use&b).sum())==chosen['rescue'] and int((use&h).sum())==chosen['damage']
    summary=json.loads((a.OUT/'summary.json').read_text())
    with np.load(a.OUT/'development.npz') as f:
        use=(f['score']>=chosen['threshold'])&(f['features'][:,-1]>0)
        assert np.array_equal(use,f['override'])
        for name in ['all','ambiguous','matcher','ambiguous_matcher']:
            r=summary['groups'][name]; rescue=int((use&f[name]&f['benefit']).sum()); damage=int((use&f[name]&f['harm']).sum())
            assert (rescue,damage,rescue-damage)==(r['rescue'],r['damage'],r['net'])
    logs=json.loads((a.OUT/'training.json').read_text()); assert [r['epoch'] for r in logs]==list(range(1,101))
    verification=dict(folds_disjoint=True,fit_frames_covered_once=578,temporal_embargo_verified=True,six_inputs_recomputed=True,labels_recomputed=True,parameters=129,epochs=100,selection_recomputed=True,development_counts_recomputed=True,best_sha256=a.q.e.run.digest(a.OUT/'best.pt'),protocol_sha256=a.q.e.run.digest(a.OUT/'protocol.json'))
    a.q.e.run.save_json(a.OUT/'verification.json',verification); print(verification,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); main()
