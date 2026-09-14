import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'image_gate'))
import run
from model import ProtectedVisualReliability, boundary_ranking_loss, selection_state, select

SOURCE=Path('/home/zhang/leader-image-gate-raw')
OUTPUT=Path('/home/zhang/protected-visual-reliability')
SEEDS=[2089,2090,2091]


def config():
    return SimpleNamespace(output=SOURCE,checkpoint=run.WORKSPACE/'research/image_gate_checkpoint')


def load_data():
    rows=json.loads((SOURCE/'manifest.json').read_text())
    data=[]
    for index,row in enumerate(rows):
        item=run.frame(config(),row)
        item.update(row=row,index=index)
        item['state']=selection_state(item['prediction'],item['valid'])
        item['coordinate_error']=(item['prediction'][:,:3]-item['target']).norm(dim=-1)
        data.append(item)
    return data


def visual(item,arm,seed):
    if arm=='aligned':return item['image']
    image=item['image'].clone()
    indices=torch.where(item['valid'])[0]
    generator=torch.Generator(device='cuda').manual_seed(seed*10000+item['index'])
    image[indices]=image[indices[torch.randperm(len(indices),device='cuda',generator=generator)]]
    return image


def infer(head,item,arm,seed):
    out=head(item['features'],visual(item,arm,seed),item['prediction'],item['valid'])
    assert torch.equal(out.pred[:,:3],item['prediction'][:,:3])
    assert torch.equal(out.pred[out.protected,3],item['prediction'][out.protected,3])
    assert torch.equal(out.pred[~item['valid'],3],item['prediction'][~item['valid'],3])
    selected=select(out.pred,item['state'])
    if out.gap>1e-8 and out.k<len(out.pred):
        membership=torch.zeros_like(item['valid'])
        membership[selected]=True
        assert membership[out.protected].all()
    return out,selected


class Evaluator:
    def __init__(self):
        from models.sc2pcr import Matcher
        self.matcher=Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
        self.cache={}
        self.hits=0

    def pose(self,item,selected):
        key=(item['row']['frame_id'],selected.cpu().numpy().tobytes())
        if key in self.cache:
            self.hits+=1
            return self.cache[key]
        torch.manual_seed(2089)
        prediction=item['prediction']
        estimate=self.matcher.estimator(item['source'][selected][None],prediction[selected,:3][None])[0]
        estimate[:3,3]+=item['center']
        translation=(estimate[:3,3]-item['GT'][:3,3]).norm().item()
        cosine=((estimate[:3,:3].T@item['GT'][:3,:3]).trace()-1)/2
        error=[translation,torch.rad2deg(cosine.clamp(-1,1).acos()).item()]
        self.cache[key]=error
        return error

    @torch.no_grad()
    def evaluate(self,head,data,arm,seed):
        records=[]
        for item in data:
            out,selected=infer(head,item,arm,seed)
            original=item['state']['original']
            entered=selected[~torch.isin(selected,original)]
            left=original[~torch.isin(original,selected)]
            assert len(entered)==len(left)
            same=not len(entered)
            if same:assert torch.equal(selected,original)
            records.append(dict(frame_id=item['row']['frame_id'],error=self.pose(item,selected),replaced=len(entered),
                entered_error_sum=float(item['coordinate_error'][entered].sum()),left_error_sum=float(item['coordinate_error'][left].sum()),
                selected_original_error_mean=float(item['coordinate_error'][selected].mean()),
                delta_max=float(out.delta.abs().max()),gap=float(out.gap),editable=int(out.editable.sum()),
                core_retained=True,coordinates_exact=True,unchanged_candidate_order=same))
        return dict(metrics=run.metrics([r['error'] for r in records]),records=records)


def prepare(data):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    training=[d for d in data if d['row']['split']=='train']
    assert len(training)==64 and len(data)==96
    protocol=dict(source_commit='807c9ff',head='User-supplied prototype.py unchanged; 641->32->1, 20577 parameters',
        objective='Frozen-coordinate visual candidate selection with protected core; no coordinate correction',
        fit_ids=[d['row']['frame_id'] for d in training[:48]],internal_ids=[d['row']['frame_id'] for d in training[48:]],
        development_ids=[d['row']['frame_id'] for d in data if d['row']['split']=='val'],
        pilot_seed=2089,seeds=SEEDS,pilot_epochs=100,batch_frames=4,optimizer='AdamW',weight_decay=.0001,
        lr='epoch1..5 warmup from 0.0002 to 0.001; epoch6..100 cosine to 0.00001; same original schedule truncated at selected E during refit',
        epoch_selection='aligned pilot only, epoch0 and every10 through100; lowest internal16 mean final translation, earliest exact tie; one E shared by all seeds and arms',
        refit='Reset head and optimizer; fit all64 for E epochs; if E=0 preserve baseline for all arms',
        loss='Original TRR per frame +0.1 prototype boundary_ranking_loss, then average four frames',
        pairs='Prototype: original ranks25-50% vs50-75%, draw1024 keep up to256, >=one editable, error gap>1cm, score difference divided by original gap',
        shuffled='Fixed seeded independent within-frame valid descriptor permutation for each seed; same during train/eval; mask and descriptor set unchanged',
        matcher='Original coordinates only; common RNG2089; order candidates by fixed original-score ranking, preserving exact original topk order',
        pass_rule='All3 seeds aligned mean translation better than both original and shuffled; average reduction>=1% vs original; paired four-contiguous-block bootstrap95% upper bound<0 vs both; mean rotation<=original, each seed rotation and translationP95<=105% of original, zero additional1m5deg failures',
        bootstrap='Fixed four consecutive blocks of8 development frames, 2000 paired block resamples, seed918; average seed differences first',
        stop='No expansion if pilot selects epoch0 or final rule fails; no architecture/threshold tuning on development',
        scope='Previously touched same-day local data, original LEADER pretraining date, not blind/full NCLT',
        source_protocol_sha256=run.digest(SOURCE/'protocol.json'),prototype_sha256=run.digest(HERE/'prototype.py'))
    path=OUTPUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    run.save_json(path,protocol)
    original=run.load_leader(config())
    head=ProtectedVisualReliability().cuda()
    audit=[]
    max_parity=0.
    with torch.no_grad():
        for item in data:
            parity=float((original.decoder(item['features'])-item['prediction']).abs().max())
            max_parity=max(max_parity,parity)
            assert parity<=1e-4
            out,indices=infer(head,item,'aligned',2089)
            assert torch.equal(out.pred,item['prediction']) and torch.equal(indices,item['state']['original'])
            cutoff=item['prediction'][item['state']['original'][-1],3]
            possible=out.editable&(item['prediction'][:,3]>=cutoff-.5*out.gap)
            possible&=~torch.isin(torch.arange(len(possible),device='cuda'),item['state']['original'])
            audit.append(dict(frame_id=item['row']['frame_id'],split=item['row']['split'],points=len(out.pred),valid=int(item['valid'].sum()),
                editable=int(out.editable.sum()),gap=float(out.gap),potential_outside_points=int(possible.sum())))
    del original
    evaluator=Evaluator()
    inner=training[48:]
    with torch.no_grad():
        before=evaluator.pose(inner[0],inner[0]['state']['original'])
        evaluator.cache.clear()
        after=evaluator.pose(inner[0],inner[0]['state']['original'])
        assert np.allclose(before,after,atol=1e-6)
    run.save_json(OUTPUT/'preflight.json',dict(parameters=sum(p.numel() for p in head.parameters()),baseline_prediction_max_error=max_parity,
        matcher_repeat_error=abs(np.array(before)-after).max().item(),records=audit))
    print(json.dumps(dict(frames=len(audit),editable=sum(r['editable'] for r in audit),potential_outside=sum(r['potential_outside_points'] for r in audit),parity=max_parity)),flush=True)


def learning_rate(epoch):
    if epoch<=5:return .001*epoch/5
    return .00001+(.001-.00001)*.5*(1+math.cos(math.pi*(epoch-5)/95))


def train_epochs(head,data,epochs,arm,seed,callback=None):
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    pair_rng=torch.Generator(device='cuda').manual_seed(seed+100000)
    trr=run.official_trr()
    logs=[]
    updates=0
    first_gradient=0.
    started=time.perf_counter()
    for epoch in range(1,epochs+1):
        lr=learning_rate(epoch)
        for group in optimizer.param_groups:group['lr']=lr
        order=rng.permutation(len(data))
        losses=[]
        for start in range(0,len(order),4):
            optimizer.zero_grad(set_to_none=True)
            tasks=[]
            ranks=[]
            for index in order[start:start+4]:
                item=data[index]
                out=head(item['features'],visual(item,arm,seed),item['prediction'],item['valid'])
                task=trr(item['target'],out.pred[:,:3],out.pred[:,3],torch.zeros(len(out.pred),dtype=torch.long,device='cuda'))[0].mean()
                rank=boundary_ranking_loss(out,item['prediction'],item['target'],generator=pair_rng)
                tasks.append(task)
                ranks.append(rank)
            task=torch.stack(tasks).mean()
            ranking=torch.stack(ranks).mean()
            loss=task+.1*ranking
            assert torch.isfinite(loss)
            if loss.requires_grad:
                loss.backward()
                if not first_gradient:first_gradient=float(head.head[-1].weight.grad.norm())
                optimizer.step()
                updates+=1
            losses.append([float(task.detach()),float(ranking.detach())])
        if epoch%10==0 or epoch==epochs:
            values=np.mean(losses,axis=0)
            logs.append(dict(epoch=epoch,trr=float(values[0]),ranking=float(values[1]),lr=lr))
            print(f'{arm} seed{seed} epoch{epoch}/{epochs} TRR={values[0]:.5f}',flush=True)
        if callback and epoch%10==0:callback(epoch,head)
    return dict(logs=logs,optimizer_updates=updates,first_gradient_norm=first_gradient,seconds=time.perf_counter()-started)


def new_head(seed):
    torch.manual_seed(seed)
    return ProtectedVisualReliability().cuda()


def pilot(data):
    training=[d for d in data if d['row']['split']=='train']
    fit,internal=training[:48],training[48:]
    head=new_head(2089)
    evaluator=Evaluator()
    evaluations=[]
    best=dict(epoch=0,mean=float('inf'))
    def check(epoch,head):
        result=evaluator.evaluate(head,internal,'aligned',2089)
        evaluations.append(dict(epoch=epoch,**result))
        value=result['metrics']['mean'][0]
        if value<best['mean']-1e-12:
            best.update(epoch=epoch,mean=value)
            torch.save(head.state_dict(),OUTPUT/'pilot_best.pt')
        run.save_json(OUTPUT/'pilot_evaluation.json',evaluations)
        print(f'internal epoch{epoch} mean={value:.6f} bestE={best["epoch"]}',flush=True)
    check(0,head)
    stats=train_epochs(head,fit,100,'aligned',2089,check)
    torch.save(head.state_dict(),OUTPUT/'pilot_last.pt')
    run.save_json(OUTPUT/'pilot_training.json',stats)
    run.save_json(OUTPUT/'selected_epoch.json',dict(**best,criterion='internal16 final mean translation only',epoch0=evaluations[0]['metrics'],
        model_selection_finished_before_development=True,matcher_cache_hits=evaluator.hits))
    print(json.dumps(best),flush=True)


def refit(data):
    epochs=json.loads((OUTPUT/'selected_epoch.json').read_text())['epoch']
    training=[d for d in data if d['row']['split']=='train']
    for seed in SEEDS:
        for arm in ['aligned','shuffled']:
            head=new_head(seed)
            stats=train_epochs(head,training,epochs,arm,seed)
            prefix=f'{arm}_{seed}'
            torch.save(head.state_dict(),OUTPUT/(prefix+'.pt'))
            run.save_json(OUTPUT/(prefix+'_training.json'),stats)


def block_interval(delta):
    means=np.asarray(delta).reshape(4,8).mean(1)
    rng=np.random.default_rng(918)
    samples=means[rng.integers(4,size=(2000,4))].mean(1)
    return dict(mean=float(means.mean()),blocks=means.tolist(),ci95=np.percentile(samples,[2.5,97.5]).tolist())


def evaluate(data):
    development=[d for d in data if d['row']['split']=='val']
    evaluator=Evaluator()
    baseline=evaluator.evaluate(new_head(2089),development,'aligned',2089)
    expected=json.loads((SOURCE/'validation_frames.json').read_text())
    assert np.allclose([r['error'] for r in baseline['records']],[r['errors']['baseline'] for r in expected],atol=1e-6)
    results=dict(baseline=baseline)
    for seed in SEEDS:
        for arm in ['aligned','shuffled']:
            prefix=f'{arm}_{seed}'
            head=new_head(seed)
            head.load_state_dict(torch.load(OUTPUT/(prefix+'.pt'),map_location='cuda'))
            result=evaluator.evaluate(head,development,arm,seed)
            results[prefix]=result
            print(prefix,json.dumps(result['metrics']),flush=True)
    base=np.array([r['error'] for r in baseline['records']])
    aligned=np.array([[r['error'] for r in results[f'aligned_{s}']['records']] for s in SEEDS])
    shuffled=np.array([[r['error'] for r in results[f'shuffled_{s}']['records']] for s in SEEDS])
    ci_base=block_interval((aligned[:,:,0]-base[:,0]).mean(0))
    ci_shuffled=block_interval((aligned[:,:,0]-shuffled[:,:,0]).mean(0))
    means=aligned.mean(1)
    before=(base[:,0]<1)&(base[:,1]<5)
    after=(aligned[:,:,0]<1)&(aligned[:,:,1]<5)
    passed=bool((means[:,0]<base[:,0].mean()).all() and (means[:,0]<shuffled[:,:,0].mean(1)).all())
    passed&=bool(means[:,0].mean()<=.99*base[:,0].mean() and ci_base['ci95'][1]<0 and ci_shuffled['ci95'][1]<0)
    passed&=bool(means[:,1].mean()<=base[:,1].mean() and (means[:,1]<=1.05*base[:,1].mean()).all())
    passed&=bool((np.percentile(aligned[:,:,0],95,axis=1)<=1.05*np.percentile(base[:,0],95)).all() and not ((~after)&before).any())
    results.update(passed=passed,paired_blocks_vs_baseline=ci_base,paired_blocks_vs_shuffled=ci_shuffled,
        rescue_per_seed=((after)&(~before)).sum(1).tolist(),damage_per_seed=((~after)&before).sum(1).tolist(),
        selected_epoch=json.loads((OUTPUT/'selected_epoch.json').read_text())['epoch'],matcher_cache_hits=evaluator.hits)
    run.save_json(OUTPUT/'result.json',results)
    print(json.dumps(dict(passed=passed,vs_baseline=ci_base,vs_shuffled=ci_shuffled)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','pilot','refit','evaluate'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=load_data()
    globals()[args.stage](data)


if __name__=='__main__':main()
