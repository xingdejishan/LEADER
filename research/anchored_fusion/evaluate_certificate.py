import json
from pathlib import Path
import numpy as np
import torch
import experiment as e
from certificate_fusion import OUT,load
from subspace import solve
from reachability import TAU,classify
from report import aggregate,block_interval


def main():
    torch.set_num_threads(4)
    p=json.loads((OUT/'protocol.json').read_text())
    for seed in e.SEEDS:
        for arm in ['aligned','shuffled']:
            assert json.loads((OUT/f'{arm}_{seed}'/'complete.json').read_text())==dict(epochs=100,updates=7300)
    decoder,data,anchors=load(p['rows'])
    dev=[d for d in data if d['row']['role']=='development']
    matcher=e.Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
    baseline=e.evaluate(None,decoder,dev,anchors,matcher,'aligned',2089,True)
    old=json.loads((e.OUT/'baseline_development.json').read_text())
    assert baseline==old
    e.run.save_json(OUT/'baseline_development.json',baseline)
    source=Path(__file__).resolve().parent/'results/reachability'
    records=json.loads((source/'records.json').read_text())
    labels=json.loads((source/'candidate_labels.json').read_text())
    primary=[(i,r) for i,r in enumerate(records) if r['original']=='N' and r.get('classification')=='reachable']
    assert len(primary)==161
    with np.load(source/'witnesses.npz') as f:
        vectors={i:(f['x'+str(i)],f['k'+str(i)]) for i,r in primary}
    anchor64=anchors.cpu().numpy().astype(float)
    summary=dict(baseline=aggregate(baseline),runs={},teacher=json.loads((OUT/'teacher_summary.json').read_text()))
    dates=sorted({r['date'] for r in baseline})
    summary['baseline_dates']={date:aggregate([r for r in baseline if r['date']==date]) for date in dates}
    values={}
    for seed in e.SEEDS:
        for arm in ['aligned','shuffled']:
            name=f'{arm}_{seed}'; folder=OUT/name
            head=e.AnchoredFusion().cuda(); head.load_state_dict(torch.load(folder/'best.pt'))
            measured=e.evaluate(head,decoder,dev,anchors,matcher,arm,seed,True)
            full=[]; feature_map={}
            with torch.no_grad():
                for d in dev:
                    fused=head(d['f'],e.visual(d,arm,seed),d['editable'])
                    query=fused[d['query']].cpu().numpy().astype(float)
                    query/=np.linalg.norm(query,axis=-1,keepdims=True)
                    score=np.einsum('nd,nkd->nk',query,anchor64[d['candidates'].cpu().numpy()])
                    choice=(score>=score.max(1,keepdims=True)-TAU).argmax(1)
                    unchanged=(fused[d['query']]==d['f'][d['query']]).all(-1).cpu().numpy()
                    choice[unchanged]=0
                    pp=d['positive'].cpu().numpy(); nn=d['negative'].cpu().numpy()
                    lab=np.where(pp,'P',np.where(nn,'N','G'))
                    original=lab[:,0]; predicted=lab[np.arange(len(lab)),choice]
                    full.append(dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],count=len(lab),
                        transitions={a+b:int(((original==a)&(predicted==b)).sum()) for a in ['P','N','G'] for b in ['P','N','G']}))
                    feature_map[d['row']['frame_id']]=fused.cpu().numpy()
            e.run.save_json(folder/'development.json',measured)
            e.run.save_json(folder/'full_ranking.json',full)
            matrix=np.column_stack([head.net[-1].weight.detach().cpu().numpy(),head.net[-1].bias.detach().cpu().numpy()]).astype(float)
            sv=np.linalg.svd(matrix,compute_uv=False)
            if not matrix.any():
                q=np.zeros((512,0)); accepted=True
            else:
                q,_=np.linalg.qr(matrix,mode='reduced'); accepted=bool(sv[-1]>1e-12*sv[0])
            np.savez_compressed(folder/'coverage_basis.npz',Q=q,matrix=matrix)
            coverage=[]; certs=[]; witness={}
            for rid,r in primary:
                x,k=vectors[rid]; lab=np.array(labels[r['frame_id']][r['candidate_row']])
                levels={}
                for level,comp in [('N',lab=='N'),('NG',lab!='P')]:
                    lows,highs=[],[]
                    for pos in np.flatnonzero(lab=='P'):
                        c,w,lo,hi,meta=solve(x,k[pos]-k[comp],q,.05 if q.shape[1] else 0.)
                        key=str(len(certs)); witness['c'+key]=c; witness['w'+key]=w
                        certs.append(dict(key=key,record=rid,positive=int(pos),level=level,lower=lo,upper=hi,**meta))
                        lows.append(lo); highs.append(hi)
                    levels[level]=[max(lows),max(highs)]
                f=feature_map[r['frame_id']][r['voxel']].astype(float); s=k@(f/np.linalg.norm(f))
                margin=float(s[lab=='P'].max()-s[lab!='P'].max())
                coverage.append(dict(record=rid,bounds=levels,classification=classify(*levels['N'],*levels['NG']) if accepted else 'basis_unresolved',actual_correct=margin>TAU))
            np.savez_compressed(folder/'coverage_witnesses.npz',**witness)
            e.run.save_json(folder/'coverage_certificates.json',certs)
            e.run.save_json(folder/'coverage.json',coverage)
            result=aggregate(measured)
            result['selection']=json.loads((folder/'selection.json').read_text())
            result['dates']={date:aggregate([r for r in measured if r['date']==date]) for date in dates}
            result['full_ranking']={key:sum(r['transitions'][key] for r in full) for key in full[0]['transitions']}
            result['coverage']=dict(total=161,reachable=sum(r['classification']=='reachable' for r in coverage),actual_correct=sum(r['actual_correct'] for r in coverage),rank=q.shape[1])
            summary['runs'][name]=result
            values[name]=np.array([r['standard'] for r in measured])
            print('evaluated',name,result['standard']['mean'],result['coverage'],flush=True)
    a=np.mean([values[f'aligned_{s}'] for s in e.SEEDS],axis=0)
    s=np.mean([values[f'shuffled_{s}'] for s in e.SEEDS],axis=0)
    b=np.array([r['standard'] for r in baseline])
    summary['paired_translation']=dict(baseline=block_interval(a[:,0]-b[:,0],baseline),shuffled=block_interval(a[:,0]-s[:,0],baseline))
    passed=True
    for seed in e.SEEDS:
        a,s=[summary['runs'][f'{arm}_{seed}'] for arm in ['aligned','shuffled']]
        passed &= a['standard']['mean'][0]<min(summary['baseline']['standard']['mean'][0],s['standard']['mean'][0])
        passed &= a['mechanism']['accuracy']>max(summary['baseline']['mechanism']['accuracy'],s['mechanism']['accuracy'])
        for date in dates:
            passed &= a['dates'][date]['standard']['mean'][0]<min(summary['baseline_dates'][date]['standard']['mean'][0],s['dates'][date]['standard']['mean'][0])
    for key,index in [('mean',1),('p95',0)]:
        passed &= not all(summary['runs'][f'aligned_{seed}']['standard'][key][index]>summary['baseline']['standard'][key][index] for seed in e.SEEDS)
    passed &= all(v['ci95'][1]<0 for v in summary['paired_translation'].values())
    summary['passed']=bool(passed)
    e.run.save_json(OUT/'summary.json',summary)


if __name__=='__main__':
    main()
