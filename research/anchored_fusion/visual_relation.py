import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import certificate_fusion as c
import experiment as e
from report import block_interval

OUT = Path('/home/zhang/visual-relation-supervision')


def hidden(decoder, features):
    result = []
    handle = decoder.pred_out[2].register_forward_hook(lambda module, inputs, output: result.append(output))
    prediction = decoder(features)
    handle.remove()
    return prediction, result[0]


def protocol():
    source = json.loads((e.OUT/'protocol.json').read_text())
    p = dict(rows=source['rows'], seed=2089, epochs=100, batch_frames=8, relations_per_update=256,
        lr=1e-5, final_lr=1e-6, warmup=5, weight_decay=1e-4, margin=.05, relation_weight=.1,
        hidden='Existing pred_out[2] activation, directly before final Linear(512,4); no new parameters',
        trainable='All original MMRegressor parameters, initialized from original checkpoint; RPGE and all caches frozen',
        pool='All P x N combinations within fixed top16 for fit queries having P and N and original top1-top2 gap<=.02; no core protection or original-error-only filter',
        labels='Cartesian representative GT: P<=.5m, N>=2m; gray ignored in loss and retained in full ranking',
        visual='Float64 cosine(query image,positive image)>cosine(query image,negative image), strict, no confidence threshold',
        shuffle='Permute selection flags within query-date and deciles of original LiDAR cosine(n)-cosine(p); seed2089; preserve each stratum selected count; never change GT labels',
        sampling='256 selected relations per update using a uniform cyclic permutation of each arm selected list; same index schedule in V/S, separate RNG from frame order; all selected relations receive floor/ceil equal exposure over run',
        checkpoints=list(range(0,101,10)), selection='Minimum internal mean final position error; earliest ties; epoch0 eligible; all arms complete100 epochs',
        evaluation='182 touched development frames only after all training; same current decoder on query and fit references; tau1e-7 earliest candidate tie; gray separate; original Matcher intersection',
        pass_rule='V mean translation below B0,B1,S, paired block intervals reported; rotation and P95 checked; hidden ranking improves versus B1/S; one-seed local evidence only',
        source_protocol_sha256=e.run.digest(e.OUT/'protocol.json'))
    e.run.save_json(OUT/'protocol.json',p)
    return p


def load(p, development=False):
    rows=p['rows'] if development else [r for r in p['rows'] if r['role']!='development']
    decoder,data,_=c.load(rows)
    fit=[d for d in data if d['row']['role']=='fit']
    raw=torch.cat([d['f'][d['valid']] for d in fit])
    return decoder,data,raw


def prepare(p):
    decoder,data,raw=load(p)
    fit=[d for d in data if d['row']['role']=='fit']
    anchors=F.normalize(raw.double(),dim=-1)
    visual=F.normalize(torch.cat([d['image'][d['valid']] for d in fit]).double(),dim=-1)
    triples=[]; difficulties=[]; flags=[]; dates=[]
    offset=0
    for d in fit:
        eligible=torch.where(d['ambiguous'])[0].tolist()
        for i in eligible:
            voxel=int(d['query'][i]); ids=d['candidates'][i]
            pos=torch.where(d['positive'][i])[0]
            neg=torch.where(d['negative'][i])[0]
            pp,nn=torch.meshgrid(pos,neg,indexing='ij'); pp=pp.flatten(); nn=nn.flatten()
            query=F.normalize(d['f'][voxel].double(),dim=-1)
            image=F.normalize(d['image'][voxel].double(),dim=-1)
            scores=anchors[ids]@query; vscores=visual[ids]@image
            count=len(pp)
            triples.extend(zip([offset+voxel]*count,ids[pp].tolist(),ids[nn].tolist()))
            difficulties.extend((scores[nn]-scores[pp]).tolist())
            flags.extend((vscores[pp]>vscores[nn]).tolist())
            dates.extend([d['row']['session_id']]*count)
        offset+=len(d['f'])
    triples=np.asarray(triples,np.int64); difficulty=np.asarray(difficulties); aligned=np.asarray(flags,bool); dates=np.asarray(dates)
    shuffled=aligned.copy(); strata=np.zeros(len(flags),np.int32); stats=[]; rng=np.random.default_rng(2089)
    for di,date in enumerate(sorted(set(dates))):
        use=np.flatnonzero(dates==date)
        edges=np.quantile(difficulty[use],np.linspace(0,1,11))
        bins=np.searchsorted(edges[1:-1],difficulty[use],side='right')
        for bi in range(10):
            ids=use[bins==bi]; strata[ids]=di*10+bi
            shuffled[ids]=aligned[ids][rng.permutation(len(ids))]
            assert aligned[ids].sum()==shuffled[ids].sum()
            stats.append(dict(date=str(date),bin=bi,count=len(ids),selected=int(aligned[ids].sum()),intersection=int((aligned[ids]&shuffled[ids]).sum())))
    np.savez_compressed(OUT/'relations.npz',triples=triples,difficulty=difficulty,V=aligned,S=shuffled,strata=strata)
    files={}
    for d in data:
        fid=d['row']['frame_id']
        files[fid]={kind:e.run.digest(root/(fid+'.npz')) for kind,root in [('lidar',e.CACHE/'lidar'),('visual',e.CACHE/'visual_raw'),('candidates',e.OUT/'candidates')]}
    e.run.save_json(OUT/'inputs.json',dict(files=files,checkpoint=e.run.digest(e.run.WORKSPACE/'research/image_gate_checkpoint/model.safetensors')))
    e.run.save_json(OUT/'relations.json',dict(pool=len(flags),selected=int(aligned.sum()),overlap=int((aligned&shuffled).sum()),overlap_fraction=float((aligned&shuffled).sum()/aligned.sum()),strata=stats,sha256=e.run.digest(OUT/'relations.npz')))
    with torch.no_grad():
        pred,h=hidden(decoder,data[0]['f'])
        assert torch.equal(pred,decoder(data[0]['f']))
        assert torch.equal(pred,decoder.pred_out[3](h))
    print('PREPARED',len(flags),int(aligned.sum()),flush=True)


def matcher():
    return e.Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)


@torch.no_grad()
def evaluate(decoder,data,solver,raw=None):
    reference=None
    if raw is not None:
        reference=torch.cat([F.normalize(hidden(decoder,b)[1].double(),dim=-1) for b in raw.split(2048)])
    records=[]; labels=[]; choices=[]
    for d in data:
        pred,h=hidden(decoder,d['f'])
        record=dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],standard=e.pose(d,pred,solver))
        if reference is not None:
            score=(F.normalize(h[d['query']].double(),dim=-1)[:,None]*reference[d['candidates']]).sum(-1)
            choice=(score>=score.max(1,keepdim=True).values-1e-7).long().argmax(1)
            label=torch.where(d['positive'].gather(1,choice[:,None])[:,0],1,torch.where(d['negative'].gather(1,choice[:,None])[:,0],-1,0))
            original=torch.zeros_like(d['valid']); original[d['indices']]=True
            record['groups']={}
            for name,mask in [('all',torch.ones_like(d['ambiguous'])),('ambiguous',d['ambiguous']),('matcher',original[d['query']]),('ambiguous_matcher',d['ambiguous']&original[d['query']])]:
                record['groups'][name]=dict(count=int(mask.sum()),P=int(((label==1)&mask).sum()),N=int(((label==-1)&mask).sum()),G=int(((label==0)&mask).sum()))
            labels.extend(label.cpu().tolist()); choices.extend(choice.cpu().tolist())
        records.append(record)
    return records,np.asarray(labels,np.int8),np.asarray(choices,np.int8)


def train(p):
    decoder,data,raw=load(p)
    initial=copy.deepcopy(decoder.state_dict())
    decoder.requires_grad_(True)
    fit=[d for d in data if d['row']['role']=='fit']; internal=[d for d in data if d['row']['role']=='internal']
    queries=torch.cat([d['f'] for d in fit])
    with np.load(OUT/'relations.npz') as a:
        pools={arm:a['triples'][a[arm]] for arm in ['V','S']}
    assert len(pools['V'])==len(pools['S'])>0
    solver=matcher(); trr=e.run.official_trr()
    baseline,_,_=evaluate(decoder,internal,solver)
    old=json.loads((e.OUT/'baseline_internal.json').read_text())
    assert [r['standard'] for r in old]==[r['standard'] for r in baseline]
    e.run.save_json(OUT/'baseline_internal.json',baseline)
    base=float(np.mean([r['standard'][0] for r in baseline]))
    for arm in ['B1','V','S']:
        folder=OUT/arm; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): continue
        decoder.load_state_dict(initial); torch.manual_seed(2089)
        optimizer=torch.optim.AdamW(decoder.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        rng=np.random.default_rng(2089); rrng=np.random.default_rng(3096)
        schedule=rrng.permutation(len(pools['V'])); cursor=0
        logs=[]; best=base
        if (folder/'resume.pt').exists():
            state=torch.load(folder/'resume.pt'); decoder.load_state_dict(state['decoder']); optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng']; rrng.bit_generator.state=state['rrng']; schedule=state['schedule']; cursor=state['cursor']; logs=state['logs']; best=state['best']
        else:
            torch.save(initial,folder/'best.pt')
            e.run.save_json(folder/'selection.json',dict(epoch=0,mean_translation=base))
        for epoch in range(len(logs)+1,101):
            start=time.perf_counter(); totals=[]; gradient=None
            lr=p['lr']*epoch/5 if epoch<=5 else p['final_lr']+(p['lr']-p['final_lr'])*(1+math.cos(math.pi*(epoch-5)/95))/2
            optimizer.param_groups[0]['lr']=lr
            order=rng.permutation(len(fit))
            for offset in range(0,len(fit),8):
                batch=[fit[i] for i in order[offset:offset+8]]
                optimizer.zero_grad(set_to_none=True)
                pred=decoder(torch.cat([d['f'] for d in batch]))
                ids=torch.cat([torch.full((len(d['f']),),i,device='cuda',dtype=torch.long) for i,d in enumerate(batch)])
                regression=trr(torch.cat([d['target'] for d in batch]),pred[:,:3],pred[:,3],ids)[0].mean()
                regression.backward()
                auxiliary=torch.zeros((),device='cuda')
                selected=[]
                while len(selected)<p['relations_per_update']:
                    take=min(p['relations_per_update']-len(selected),len(schedule)-cursor)
                    selected.extend(schedule[cursor:cursor+take]); cursor+=take
                    if cursor==len(schedule): schedule=rrng.permutation(len(schedule)); cursor=0
                if arm!='B1':
                    triples=torch.tensor(pools[arm][selected],device='cuda')
                    features=torch.cat([queries[triples[:,0]],raw[triples[:,1]],raw[triples[:,2]]])
                    _,h=hidden(decoder,features); q,positive,negative=F.normalize(h,dim=-1).chunk(3)
                    auxiliary=(p['margin']-(q*positive).sum(-1)+(q*negative).sum(-1)).relu().mean()
                    if offset==0:
                        g=torch.autograd.grad(auxiliary,decoder.pred_out[0].weight,retain_graph=True)[0]
                        gradient=g.norm().item()
                        assert gradient>0 and np.isfinite(gradient)
                    (p['relation_weight']*auxiliary).backward()
                assert torch.isfinite(regression) and torch.isfinite(auxiliary)
                optimizer.step()
                totals.append([regression.item(),auxiliary.item()])
            log=dict(epoch=epoch,lr=lr,loss=np.mean(totals,axis=0).tolist(),seconds=time.perf_counter()-start,relation_gradient=gradient)
            if epoch%10==0:
                values,_,_=evaluate(decoder,internal,solver)
                score=float(np.mean([r['standard'][0] for r in values])); log['internal']=e.run.metrics([r['standard'] for r in values])
                if score<best:
                    best=score; torch.save(decoder.state_dict(),folder/'best.pt'); e.run.save_json(folder/'selection.json',dict(epoch=epoch,mean_translation=best))
                e.run.save_json(folder/f'internal_{epoch}.json',values)
            logs.append(log); e.run.save_json(folder/'training.json',logs)
            torch.save(dict(decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,rrng=rrng.bit_generator.state,schedule=schedule,cursor=cursor,logs=logs,best=best),folder/'resume.pt')
            print(arm,epoch,'loss',log['loss'],'seconds',round(log['seconds'],2),'best',best,flush=True)
        torch.save(decoder.state_dict(),folder/'last.pt'); e.run.save_json(folder/'complete.json',dict(epochs=100,updates=7300,relation_exposures=0 if arm=='B1' else 7300*p['relations_per_update']))


def assess(p):
    assert all((OUT/arm/'complete.json').exists() for arm in ['B1','V','S'])
    decoder,data,raw=load(p,True); initial=copy.deepcopy(decoder.state_dict())
    dev=[d for d in data if d['row']['role']=='development']; solver=matcher()
    results={}; arrays={}
    for arm in ['B0','B1','V','S']:
        decoder.load_state_dict(initial if arm=='B0' else torch.load(OUT/arm/'best.pt'))
        records,labels,choices=evaluate(decoder,dev,solver,raw)
        e.run.save_json(OUT/f'{arm}_development.json',records)
        np.savez_compressed(OUT/f'{arm}_ranking.npz',labels=labels,choices=choices)
        results[arm]=records; arrays[arm]=labels
    old=json.loads((e.OUT/'baseline_development.json').read_text())
    assert [r['standard'] for r in old]==[r['standard'] for r in results['B0']]
    summary={}
    masks={'all':np.ones(len(arrays['B0']),bool),'ambiguous':np.concatenate([d['ambiguous'].cpu().numpy() for d in dev])}
    masks['matcher']=np.concatenate([torch.isin(d['query'],d['indices']).cpu().numpy() for d in dev]); masks['ambiguous_matcher']=masks['ambiguous']&masks['matcher']
    np.savez_compressed(OUT/'evaluation_masks.npz',**masks)
    for arm,records in results.items():
        values=np.asarray([r['standard'] for r in records])
        summary[arm]=dict(metrics=e.run.metrics(values),dates={date:e.run.metrics([r['standard'] for r in records if r['date']==date]) for date in sorted({r['date'] for r in records})},mechanism={})
        if arm!='B0': summary[arm]['selection']=json.loads((OUT/arm/'selection.json').read_text())
        for name,mask in masks.items():
            a=arrays['B0'][mask]; b=arrays[arm][mask]
            summary[arm]['mechanism'][name]=dict(count=int(mask.sum()),P=int((b==1).sum()),N=int((b==-1).sum()),G=int((b==0).sum()),transitions={str(i)+':'+str(j):int(((a==i)&(b==j)).sum()) for i in [1,0,-1] for j in [1,0,-1]})
    summary['paired']={}
    for other in ['B0','B1','S']:
        v=np.asarray([r['standard'] for r in results['V']]); o=np.asarray([r['standard'] for r in results[other]])
        summary['paired']['V-'+other]=block_interval(v[:,0]-o[:,0],results['V'])
    e.run.save_json(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','assess']); args=parser.parse_args()
    torch.set_num_threads(4); OUT.mkdir(exist_ok=True)
    p=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else protocol()
    globals()[args.stage](p)
