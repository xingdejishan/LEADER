import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import torch
import experiment as e

OUT=Path('/home/zhang/certificate-supervised-fusion')
GAMMA=1e-4
GUARD=1e-8
PAD=1e-10


def minimum(x,a):
    q,_=np.linalg.qr(a.T,mode='reduced')
    h=a@q
    need=GAMMA-a@x
    result=minimize(lambda c:.5*c@c,np.zeros(q.shape[1]),jac=lambda c:c,
        constraints=[{'type':'ineq','fun':lambda c:h@c-need-GUARD,'jac':lambda c:h}],
        method='SLSQP',options={'ftol':1e-13,'maxiter':500})
    delta=q@result.x
    gram=a@a.T
    dual=minimize(lambda w:.5*w@gram@w-need@w,np.zeros(len(a)),jac=lambda w:gram@w-need,
        bounds=[(0,None)]*len(a),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':1000,'maxls':50})
    w=np.maximum(dual.x,0)
    lower=float(need@w-.5*np.linalg.norm(w@a)**2-PAD)
    upper=float(.5*delta@delta+PAD)
    margin=float(np.min(a@(x+delta))-PAD)
    valid=bool(np.isfinite(delta).all() and np.isfinite(w).all() and margin>=GAMMA and np.linalg.norm(delta)<=.05 and upper-lower<=1e-8)
    status='certified' if valid else 'radius_blocked' if lower>.5*.05**2+PAD else 'uncertain'
    return delta,w,dict(status=status,lower=lower,upper=upper,margin=margin,norm=float(np.linalg.norm(delta)),
        solver=str(result.message),dual_solver=str(dual.message),primal_violation=float(max(0,np.max(need+GUARD-h@result.x))),
        dual_violation=float(max(0,-dual.x.min())),iterations=int(result.nit),dual_iterations=int(dual.nit))


def protocol():
    OUT.mkdir(exist_ok=True)
    p=json.loads((e.OUT/'protocol.json').read_text())
    p.pop('contrastive_weight'); p.pop('contrastive'); p.pop('temperature')
    p.update(direction_weight=.01,gamma=GAMMA,teacher_guard=GUARD,teacher_pad=PAD,teacher_optimality_gap=1e-8,
        teacher='Fit578 only, same fit anchors and candidate16; no visual-ranking filter; all editable original-P zero; original-N with P solve all positives against N+G; original-G/no-P/unresolved skip direction only',
        teacher_selection='All positives require certified feasible or radius-blocked result; any uncertain positive skips query. Lowest certified norm, tolerance1e-12 then earliest candidate index.',
        teacher_filter='For original Matcher editable candidates, frozen decoder coordinate error on original coarse target must not increase (delta<=0); rejected targets are skipped, never zero-labelled.',
        direction_loss='Mean squared L2 norm of (applied residual / original feature norm - teacher_delta)/.05; separately average correction and zero classes in each batch, equal weight across classes present. No contrastive term.',
        initialization='Same-seed paired from-scratch 37408-parameter heads; zero output layer; train both layers and biases; all original modules frozen',
        source_protocol_sha256=e.run.digest(e.OUT/'protocol.json'))
    e.run.save_json(OUT/'protocol.json',p)
    return p


def load(rows):
    decoder=e.run.load_leader(SimpleNamespace(checkpoint=e.run.WORKSPACE/'research/image_gate_checkpoint')).decoder
    data=e.load_data(decoder,rows)
    fit=[d for d in data if d['row']['role']=='fit']
    anchors=torch.nn.functional.normalize(torch.cat([d['f'][d['valid']] for d in fit]),dim=-1)
    for d in data:
        path=e.OUT/'candidates'/(d['row']['frame_id']+'.npz')
        assert path.exists()
        with np.load(path) as f:
            for dest,source in [('query','indices'),('candidates','candidates'),('positive','positive'),('negative','negative')]:
                d[dest]=torch.tensor(f[source],device='cuda')
            d['ambiguous']=d['positive'].any(1)&d['negative'].any(1)&torch.tensor(f['gap']<=.02,device='cuda')
    return decoder,data,anchors


def prepare(p):
    rows=[r for r in p['rows'] if r['role']=='fit']
    assert len(rows)==578
    decoder,data,_=load(rows)
    raw=np.concatenate([d['f'][d['valid']].cpu().numpy() for d in data]).astype(float)
    anchors=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    folder=OUT/'teacher'; folder.mkdir(exist_ok=True)
    stats=[]
    for frame,d in enumerate(data):
        path=folder/(d['row']['frame_id']+'.npz')
        if path.exists() and path.with_suffix('.json').exists():
            stats.append(json.loads(path.with_suffix('.json').read_text())); continue
        indices=d['query'].cpu().numpy(); candidates=d['candidates'].cpu().numpy()
        positive=d['positive'].cpu().numpy(); negative=d['negative'].cpu().numpy()
        features=d['f'].cpu().numpy().astype(float)
        x=features/np.linalg.norm(features,axis=1,keepdims=True)
        editable=d['editable'].cpu().numpy()
        selected=np.zeros(len(features),bool); selected[d['indices'].cpu().numpy()]=True
        targets=np.zeros_like(features,dtype=np.float32); kind=np.zeros(len(features),np.int8)
        record=dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],points=len(features),zero=0,attempted=0,certified=0,
            kept=0,coordinate_rejected=0,radius_blocked=0,uncertain=0,gray_skipped=0,no_positive=0,protected_skipped=0,checks=[])
        witness={}
        for i,voxel in enumerate(indices):
            if not editable[voxel]:
                record['protected_skipped']+=1; continue
            if positive[i,0]:
                kind[voxel]=1; record['zero']+=1; continue
            if not negative[i,0]:
                record['gray_skipped']+=1; continue
            if not positive[i].any():
                record['no_positive']+=1; continue
            record['attempted']+=1
            k=anchors[candidates[i]]
            witness['x'+str(voxel)]=x[voxel]
            witness['k'+str(voxel)]=k
            comp=~positive[i]
            options=[]; checks=[]
            for pos in np.flatnonzero(positive[i]):
                delta,w,info=minimum(x[voxel],k[pos]-k[comp])
                key=f'{voxel}_{pos}'
                witness['d'+key]=delta; witness['w'+key]=w
                info.update(voxel=int(voxel),candidate_row=int(i),positive=int(pos),key=key)
                checks.append(info)
                if info['status']=='certified':
                    options.append((float(delta@delta),int(pos),delta))
            record['checks'].extend(checks)
            if any(c['status']=='uncertain' for c in checks):
                record['uncertain']+=1; continue
            if not options:
                record['radius_blocked']+=1; continue
            best=min(options,key=lambda t:(round(t[0],12),t[1]))
            targets[voxel]=best[2]
            kind[voxel]=2; record['certified']+=1
        corrected=np.flatnonzero(kind==2)
        if len(corrected):
            proposed=d['f'].clone()
            ids=torch.tensor(corrected,device='cuda')
            proposed[ids]+=torch.tensor(targets[corrected],device='cuda')*d['f'][ids].norm(dim=-1,keepdim=True)
            with torch.no_grad():
                prediction=decoder(proposed)
            difference=((prediction[:,:3]-d['target']).norm(dim=-1)-(d['base'][:,:3]-d['target']).norm(dim=-1)).cpu().numpy()
            reject=(kind==2)&selected&(difference>0)
            record['coordinate_rejected']=int(reject.sum())
            kind[reject]=0
            record['coordinate_deltas']={str(i):float(difference[i]) for i in corrected}
        record['kept']=int((kind==2).sum())
        np.savez_compressed(path,targets=targets,kind=kind,indices=indices,candidates=candidates,positive=positive,negative=negative,
            selected=selected,editable=editable,**witness)
        e.run.save_json(path.with_suffix('.json'),record)
        stats.append(record)
        if frame%50==0:
            print('teacher',frame+1,'kept',sum(s['kept'] for s in stats),'zero',sum(s['zero'] for s in stats),flush=True)
    keys=['zero','attempted','certified','kept','coordinate_rejected','radius_blocked','uncertain','gray_skipped','no_positive','protected_skipped']
    e.run.save_json(OUT/'teacher_summary.json',dict(frames=len(stats),totals={k:sum(s[k] for s in stats) for k in keys},
        frames_with_correction=sum(s['kept']>0 for s in stats),frames_with_zero=sum(s['zero']>0 for s in stats),
        dates={date:{k:sum(s[k] for s in stats if s['date']==date) for k in keys} for date in sorted({s['date'] for s in stats})}))
    print('TEACHERS COMPLETE',flush=True)


def direction(fused,features,targets,kinds):
    residual=(fused-features)/features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    errors=((residual-targets)/.05).square().sum(-1)
    values=[errors[kinds==kind].mean() for kind in [1,2] if (kinds==kind).any()]
    return torch.stack(values).mean() if values else fused.sum()*0


def train(p):
    assert (OUT/'teacher_verified.json').exists()
    rows=[r for r in p['rows'] if r['role']!='development']
    decoder,data,anchors=load(rows)
    fit=[d for d in data if d['row']['role']=='fit']
    internal=[d for d in data if d['row']['role']=='internal']
    for d in fit:
        with np.load(OUT/'teacher'/(d['row']['frame_id']+'.npz')) as f:
            d['direction']=torch.tensor(f['targets'])
            d['kind']=torch.tensor(f['kind'],device='cuda')
            assert not ((d['kind']>0)&~d['editable']).any()
    matcher=e.Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
    baseline=e.evaluate(None,decoder,internal,anchors,matcher,'aligned',2089)
    old=json.loads((e.OUT/'baseline_internal.json').read_text())
    assert [r['standard'] for r in baseline]==[r['standard'] for r in old]
    e.run.save_json(OUT/'baseline_internal.json',baseline)
    base_score=float(np.mean([r['standard'][0] for r in baseline]))
    trr=e.run.official_trr()
    for seed in e.SEEDS:
        for arm in ['aligned','shuffled']:
            folder=OUT/f'{arm}_{seed}'; folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():
                continue
            torch.manual_seed(seed)
            head=e.AnchoredFusion().cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed)
            logs=[]; best=base_score
            if (folder/'resume.pt').exists():
                checkpoint=torch.load(folder/'resume.pt'); head.load_state_dict(checkpoint['head']); optimizer.load_state_dict(checkpoint['optimizer'])
                rng.bit_generator.state=checkpoint['rng']; logs=checkpoint['logs']; best=checkpoint['best']
            else:
                torch.save(head.state_dict(),folder/'best.pt'); torch.save(head.state_dict(),folder/'epoch0.pt')
                e.run.save_json(folder/'selection.json',dict(epoch=0,mean_translation=best))
            for epoch in range(len(logs)+1,101):
                start=time.perf_counter()
                lr=.001*epoch/5 if epoch<=5 else .00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/95))/2
                optimizer.param_groups[0]['lr']=lr
                order=rng.permutation(len(fit)); totals=[]
                for offset in range(0,len(fit),8):
                    batch=[fit[i] for i in order[offset:offset+8]]
                    features=torch.cat([d['f'] for d in batch]); images=torch.cat([e.visual(d,arm,seed) for d in batch])
                    editable=torch.cat([d['editable'] for d in batch])
                    optimizer.zero_grad(set_to_none=True)
                    fused=head(features,images,editable); pred=decoder(fused)
                    batch_idx=torch.cat([torch.full((len(d['f']),),i,device='cuda',dtype=torch.long) for i,d in enumerate(batch)])
                    regression=trr(torch.cat([d['target'] for d in batch]),pred[:,:3],pred[:,3],batch_idx)[0].mean()
                    auxiliary=direction(fused,features,torch.cat([d['direction'] for d in batch]).to(features.device),torch.cat([d['kind'] for d in batch]))
                    loss=regression+.01*auxiliary
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is None for p in decoder.parameters())
                    optimizer.step()
                    totals.append([loss.item(),regression.item(),auxiliary.item()])
                log=dict(epoch=epoch,lr=lr,loss=np.mean(totals,axis=0).tolist(),seconds=time.perf_counter()-start)
                if epoch%10==0:
                    values=e.evaluate(head,decoder,internal,anchors,matcher,arm,seed)
                    score=float(np.mean([r['standard'][0] for r in values])); log['internal']=e.run.metrics([r['standard'] for r in values])
                    if score<best:
                        best=score; torch.save(head.state_dict(),folder/'best.pt')
                        e.run.save_json(folder/'selection.json',dict(epoch=epoch,mean_translation=best))
                    e.run.save_json(folder/f'internal_{epoch}.json',values)
                    print(arm,seed,epoch,'internal',score,flush=True)
                logs.append(log); e.run.save_json(folder/'training.json',logs)
                torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,logs=logs,best=best),folder/'resume.pt')
            torch.save(head.state_dict(),folder/'last.pt')
            e.run.save_json(folder/'complete.json',dict(epochs=100,updates=7300))
    print('TRAINING COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train']); args=parser.parse_args()
    torch.set_num_threads(4)
    p=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else protocol()
    if args.stage=='prepare':
        prepare(p)
    else:
        train(p)


if __name__=='__main__':
    main()
