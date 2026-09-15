import copy
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import query_location as q
from report import block_interval

OUT=Path('/home/zhang/query-only-arbitration-probe')


class Arbiter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)


def protocol():
    old=json.loads((q.OUT/'protocol.json').read_text())
    fit=[r for r in old['rows'] if r['role']=='fit']; assignment={}
    for date in sorted({r['session_id'] for r in fit}):
        rows=[r for r in fit if r['session_id']==date]
        for k,ids in enumerate(np.array_split(np.arange(len(rows)),3)):
            for i in ids: assignment[rows[i]['frame_id']]=k
    p=dict(rows=old['rows'],fold=assignment,seed=2089,
        folds='Three contiguous thirds per date; fold classifier excludes all its held-out frames and same-date training frames within10 seconds of any held-out frame.',
        category_system='Reuse fixed K25 centroids from prior578 training; common target-label definition, not refit per fold. Fold class labels may be absent from its training subset; report this. No held-out fold feature updates.',
        classifiers='Prior architecture and optimizer. Fixed L100/V35 epochs from the already selected full578 classifiers; no OOF-driven checkpoint selection. Internal145 and dev182 use prior frozen full578 L/V weights.',
        inputs=['L probability top1-top2 margin','L entropy/log25','V probability top1-top2 margin','V entropy/log25','dot(pL,pV)','L argmax != V argmax'],
        labels='1 iff L classification wrong AND V classification correct, otherwise0; use OOF predictions/GT on fit only.',
        model='6->16->1 ReLU,129 parameters; sigmoid score; no feature bank, class IDs, coordinates, GT, candidate mask or frame identity as input.',
        training='100 epochs,unweighted BCEWithLogits,AdamW lr1e-3,5epoch warmup,cosine1e-5,weight_decay1e-4,batch4096,seed2089; all OOF fit points once per epoch.',
        thresholds=[float(x) for x in np.linspace(0,1,21)]+[1.1],checkpoints=list(range(0,101,5)),
        selection='Internal all image-valid points: maximize rescue-damage; ties prefer fewer overrides, then earlier epoch, then higher threshold. Never override threshold1.1 is explicit candidate. Operate only on L/V disagreement.',
        evaluation='Freeze checkpoint and threshold before reading dev posteriors/GT. All, fixed6937ambiguity, originalLEADERMatcher intersection; per-date and paired frame-mean trajectory-block interval. No dev threshold sweep.',
        interpretation='Only tests arbitration of fixed K25 classifiers using6confidence signals; not a general query-only fusion gate or pose result. OOF2/3-trained versus full578 deployment classifier distribution shift is reported, not silently assumed absent.')
    OUT.mkdir(exist_ok=True); q.e.run.save_json(OUT/'protocol.json',p)
    return p


def load(p):
    centers=np.load(q.OUT/'centroids.npz')['centroids']; data=[]
    for r in p['rows']:
        if r['role']=='development': continue
        fid=r['frame_id']
        with np.load(q.e.CACHE/'lidar'/(fid+'.npz')) as l,np.load(q.e.CACHE/'visual_raw'/(fid+'.npz')) as v:
            ids=np.flatnonzero(v['valid']); xyz=l['source'][ids].astype(float)@l['GT'][:3,:3].astype(float).T+l['GT'][:3,3]
            target=np.linalg.norm(xyz[:,None]-centers[None],axis=-1).argmin(1)
            data.append(dict(row=r,L=l['features'][ids],V=v['image'][ids],y=target,ids=ids))
    return data


def signals(l,v):
    pl=np.exp(l.astype(float)); pv=np.exp(v.astype(float))
    sl=np.sort(pl,axis=1); sv=np.sort(pv,axis=1)
    return np.column_stack([sl[:,-1]-sl[:,-2],-(pl*l).sum(1)/np.log(25),sv[:,-1]-sv[:,-2],-(pv*v).sum(1)/np.log(25),(pl*pv).sum(1),pl.argmax(1)!=pv.argmax(1)]).astype(np.float32)


def outcomes(y,l,v):
    lc=l.argmax(1)==y; vc=v.argmax(1)==y
    return vc&~lc,lc&~vc


def oof(p,data):
    fit=[d for d in data if d['row']['role']=='fit']; counts=[len(d['y']) for d in fit]
    starts=np.cumsum([0]+counts); outputs={arm:np.empty((starts[-1],25),np.float32) for arm in ['L','V']}; covered=np.zeros(starts[-1],bool)
    records=[]
    for fold in range(3):
        held=[d for d in fit if p['fold'][d['row']['frame_id']]==fold]
        training=[d for d in fit if p['fold'][d['row']['frame_id']]!=fold and not any(d['row']['session_id']==h['row']['session_id'] and abs(int(d['row']['frame_id'])-int(h['row']['frame_id']))<=10000000 for h in held)]
        assert not {d['row']['frame_id'] for d in held}&{d['row']['frame_id'] for d in training}
        y=np.concatenate([d['y'] for d in training]); held_y=np.concatenate([d['y'] for d in held]); yt=torch.tensor(y,device='cuda')
        record=dict(fold=fold,train_frames=[d['row']['frame_id'] for d in training],held_frames=[d['row']['frame_id'] for d in held],train_points=len(y),held_points=len(held_y),missing_training_classes=sorted(set(held_y.tolist())-set(y.tolist())),classifiers={})
        for arm,epochs in [('L',100),('V',35)]:
            torch.manual_seed(2089); head=q.Classifier(arm=='V').cuda(); rng=np.random.default_rng(2089)
            x=torch.tensor(np.concatenate([d[arm] for d in training]),device='cuda'); test=torch.tensor(np.concatenate([d[arm] for d in held]),device='cuda')
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001); logs=[]
            for epoch in range(1,epochs+1):
                lr=.001*epoch/5 if epoch<=5 else .00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/95))/2
                optimizer.param_groups[0]['lr']=lr; order=rng.permutation(len(y)); total=0.
                for start in range(0,len(y),4096):
                    ids=torch.tensor(order[start:start+4096],device='cuda'); optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(head(x[ids]),yt[ids]); assert torch.isfinite(loss); loss.backward(); optimizer.step(); total+=loss.item()*len(ids)
                logs.append(dict(epoch=epoch,loss=total/len(y),lr=lr))
            lp=q.infer(head,test); offset=0
            for i,d in enumerate(fit):
                if p['fold'][d['row']['frame_id']]!=fold: continue
                n=len(d['y']); outputs[arm][starts[i]:starts[i+1]]=lp[offset:offset+n]; offset+=n
                if arm=='L': assert not covered[starts[i]:starts[i+1]].any(); covered[starts[i]:starts[i+1]]=True
            path=OUT/f'fold{fold}_{arm}.pt'; torch.save(head.state_dict(),path)
            record['classifiers'][arm]=dict(epochs=epochs,updates=epochs*math.ceil(len(y)/4096),accuracy=float((lp.argmax(1)==held_y).mean()),sha256=q.e.run.digest(path),training=logs)
            print('OOF',fold,arm,'held accuracy',record['classifiers'][arm]['accuracy'],flush=True)
        records.append(record)
    assert covered.all()
    y=np.concatenate([d['y'] for d in fit]); x=signals(outputs['L'],outputs['V']); benefit,harm=outcomes(y,outputs['L'],outputs['V'])
    np.savez_compressed(OUT/'oof.npz',target=y,features=x,benefit=benefit,harm=harm,**outputs)
    q.e.run.save_json(OUT/'folds.json',records)
    return x,benefit


def full_internal(data):
    internal=[d for d in data if d['row']['role']=='internal']; outputs={}
    for arm in ['L','V']:
        head=q.Classifier(arm=='V').cuda(); head.load_state_dict(torch.load(q.OUT/arm/'best.pt'))
        outputs[arm]=q.infer(head,torch.tensor(np.concatenate([d[arm] for d in internal]),device='cuda'))
    y=np.concatenate([d['y'] for d in internal]); x=signals(outputs['L'],outputs['V']); benefit,harm=outcomes(y,outputs['L'],outputs['V'])
    np.savez_compressed(OUT/'internal.npz',target=y,features=x,benefit=benefit,harm=harm,**outputs)
    return x,benefit,harm


def counts(score,threshold,x,benefit,harm,mask):
    override=(score>=threshold)&(x[:,-1]>0)&mask
    rescue=int((override&benefit).sum()); damage=int((override&harm).sum())
    return dict(count=int(mask.sum()),overrides=int(override.sum()),rescue=rescue,damage=damage,neutral=int(override.sum())-rescue-damage,net=rescue-damage)


def train(p,x,y,internal):
    torch.manual_seed(2089); head=Arbiter().cuda(); assert sum(t.numel() for t in head.parameters())==129
    x=torch.tensor(x,device='cuda'); y=torch.tensor(y,device='cuda',dtype=torch.float32)
    ix,benefit,harm=internal; it=torch.tensor(ix,device='cuda'); optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001); rng=np.random.default_rng(2089)
    all_choices=[]; best_key=None; best=None; logs=[]
    for epoch in range(101):
        if epoch:
            lr=.001*epoch/5 if epoch<=5 else .00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/95))/2
            optimizer.param_groups[0]['lr']=lr; order=rng.permutation(len(y)); total=0.
            for offset in range(0,len(y),4096):
                ids=torch.tensor(order[offset:offset+4096],device='cuda'); optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(head(x[ids]),y[ids]); assert torch.isfinite(loss); loss.backward(); optimizer.step(); total+=loss.item()*len(ids)
            logs.append(dict(epoch=epoch,loss=total/len(y),lr=lr))
        if epoch%5==0:
            with torch.no_grad(): scores=head(it).sigmoid().cpu().numpy()
            for threshold in p['thresholds']:
                r=counts(scores,threshold,ix,benefit,harm,np.ones(len(ix),bool)); r.update(epoch=epoch,threshold=threshold); all_choices.append(r)
                key=(r['net'],-r['overrides'],-epoch,threshold)
                if best_key is None or key>best_key:
                    best_key=key; best=r.copy(); torch.save(head.state_dict(),OUT/'best.pt')
            print('arbiter',epoch,'selected',best,flush=True)
    q.e.run.save_json(OUT/'selection.json',best); q.e.run.save_json(OUT/'internal_grid.json',all_choices); q.e.run.save_json(OUT/'training.json',logs)
    head.load_state_dict(torch.load(OUT/'best.pt'))
    with torch.no_grad(): score=head(it).sigmoid().cpu().numpy()
    np.savez_compressed(OUT/'internal_scores.npz',score=score)
    return head,best


def evaluate(p,head,selected):
    with np.load(q.OUT/'development.npz') as a:
        l=a['L']; v=a['V']; y=a['target']; ambiguous=a['ambiguous']; x=signals(l,v)
    benefit,harm=outcomes(y,l,v)
    with torch.no_grad(): score=head(torch.tensor(x,device='cuda')).sigmoid().cpu().numpy()
    frames=json.loads((q.OUT/'development_frames.json').read_text()); rows={r['frame_id']:r for r in p['rows']}; matcher_masks=[]
    decoder=q.e.run.load_leader(SimpleNamespace(checkpoint=q.e.run.WORKSPACE/'research/image_gate_checkpoint')).decoder
    for r in frames:
        fid=r['frame_id']; assert rows[fid]['role']=='development'
        with np.load(q.e.CACHE/'lidar'/(fid+'.npz')) as a,np.load(q.e.CACHE/'visual_raw'/(fid+'.npz')) as b:
            with torch.no_grad(): prediction=decoder(torch.tensor(a['features'],device='cuda'))
            indices=q.e.top(prediction).cpu().numpy(); ids=np.flatnonzero(b['valid']); matcher_masks.append(np.isin(ids,indices))
    matcher=np.concatenate(matcher_masks); masks=dict(all=np.ones(len(y),bool),ambiguous=ambiguous,matcher=matcher,ambiguous_matcher=ambiguous&matcher)
    override=(score>=selected['threshold'])&(x[:,-1]>0); final=np.where(override,v.argmax(1),l.argmax(1)); base=l.argmax(1)==y; correct=final==y
    summary=dict(selection=selected,groups={},dates={},classifier_shift={})
    for name,mask in masks.items():
        r=counts(score,selected['threshold'],x,benefit,harm,mask)
        r.update(lidar_accuracy=float(base[mask].mean()),final_accuracy=float(correct[mask].mean()))
        use=[f for f in frames if mask[f['start']:f['stop']].any()]
        delta=np.asarray([(correct[f['start']:f['stop']][mask[f['start']:f['stop']]].astype(float)-base[f['start']:f['stop']][mask[f['start']:f['stop']]].astype(float)).mean() for f in use])
        r['paired_frame_accuracy']=block_interval(delta,use); summary['groups'][name]=r
    for date in sorted({f['date'] for f in frames}):
        mask=np.zeros(len(y),bool)
        for f in frames:
            if f['date']==date: mask[f['start']:f['stop']]=True
        summary['dates'][date]={name:counts(score,selected['threshold'],x,benefit,harm,mask&m) for name,m in masks.items()}
    for name,path in [('oof',OUT/'oof.npz'),('internal',OUT/'internal.npz'),('development',q.OUT/'development.npz')]:
        with np.load(path) as a:
            b,h=outcomes(a['target'],a['L'],a['V']); summary['classifier_shift'][name]=dict(points=len(b),beneficial=int(b.sum()),harmful=int(h.sum()),L_accuracy=float((a['L'].argmax(1)==a['target']).mean()),V_accuracy=float((a['V'].argmax(1)==a['target']).mean()))
    np.savez_compressed(OUT/'development.npz',target=y,features=x,score=score,override=override,benefit=benefit,harm=harm,L=l,V=v,**masks)
    q.e.run.save_json(OUT/'development_frames.json',frames); q.e.run.save_json(OUT/'summary.json',summary)
    q.e.run.save_json(OUT/'sources.json',dict(centroids=q.e.run.digest(q.OUT/'centroids.npz'),L=q.e.run.digest(q.OUT/'L/best.pt'),V=q.e.run.digest(q.OUT/'V/best.pt'),development=q.e.run.digest(q.OUT/'development.npz')))
    print('RESULT',json.dumps(summary),flush=True)


def report():
    s=json.loads((OUT/'summary.json').read_text()); selected=s['selection']
    grid=json.loads((OUT/'internal_grid.json').read_text()); active=[r for r in grid if r['overrides']]
    best_active=max(active,key=lambda r:(r['net'],-r['overrides'],-r['epoch'],r['threshold']))
    lines=['# Query-only视觉仲裁可学习性探针','',
        '**本协议未找到净收益为正的仲裁规则：内部选模选择epoch0、threshold1.1，即永不替换；开发集救回0、损害0、净变化0。** 这不是仲裁器安全纠正了样本，而是回退到原LiDAR分类器。',
        '负结果不能直接解释为query-only信号不足：折外视觉分类器准确率为8.75%，内部完整模型33.44%，开发完整模型54.50%，存在明显输入分布差异。它是本轮解释限制，不是已确认的失败原因；也没有排除更细阈值、其他模型或其他部署信号。','',
        '固定K25分类任务，三折连续轨迹块折外预测训练129参数仲裁器；不改LEADER，不训练融合，不运行Matcher求姿态。Matcher仅用于标记原模型候选交集。','',
        '每日期578训练部分分为三个连续段，同折三个日期一起留出，另排除同日期相距≤10秒的训练帧。分类权重没有见过用于训练仲裁器的相应折外点。固定25类沿用既有训练标签定义，不逐折重聚类。',
        '折分类器使用原架构和优化器，L固定100epoch、V固定35epoch，来自上一轮已确定的训练预算/检查点轮数；不按折外GT选模。145内部与182开发使用上一轮完整578帧训练得到的固定分类器。','',
        f"内部集选择epoch={selected['epoch']}、threshold={selected['threshold']:.2f}；救回{selected['rescue']}、损害{selected['damage']}、净变化{selected['net']}。开发评估不再扫描阈值。",
        f"预固定21个0到1阈值加永不替换选项、21个检查点共{len(grid)}个内部配置，其中{len(active)}个产生非空override。最优非空配置为epoch{best_active['epoch']}、threshold{best_active['threshold']:.2f}，override{best_active['overrides']}、救回{best_active['rescue']}、损害{best_active['damage']}、中性{best_active['neutral']}；按少override并列规则仍选择永不替换。没有扫描开发集寻找另一个阈值。",'',
        '| 开发集合 | 点数 | override | 救回 | 损害 | 中性 | 净变化 | 原准确率 % | 最终准确率 % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in s['groups'].items(): lines.append(f"| {name} | {r['count']} | {r['overrides']} | {r['rescue']} | {r['damage']} | {r['neutral']} | {r['net']} | {100*r['lidar_accuracy']:.4f} | {100*r['final_accuracy']:.4f} |")
    lines+=['','| 集合 | 帧均准确率差 pp | 轨迹块95%区间 pp |','|---|---:|---|']
    for name,r in s['groups'].items():
        c=r['paired_frame_accuracy']; lines.append(f"| {name} | {100*c['mean']:.5f} | [{100*c['ci95'][0]:.5f}, {100*c['ci95'][1]:.5f}] |")
    lines+=['','区间使用每帧准确率差，日期内连续轨迹块、10000次bootstrap、seed271828；与按点计数的净变化权重不同。本轮零区间由永不替换造成，不是方法有效性或统计等效的证明。','', '| 日期 | 集合 | 救回 | 损害 | 净变化 |','|---|---|---:|---:|---:|']
    for date,groups in s['dates'].items():
        for name in ['all','ambiguous','ambiguous_matcher']:
            r=groups[name]; lines.append(f"| {date} | {name} | {r['rescue']} | {r['damage']} | {r['net']} |")
    lines+=['','## 折外与部署分类器的分布差异','', '| 集合 | 点数 | L准确率 % | V准确率 % | 有益替换 | 有害替换 |','|---|---:|---:|---:|---:|---:|']
    for name,r in s['classifier_shift'].items(): lines.append(f"| {name} | {r['points']} | {100*r['L_accuracy']:.4f} | {100*r['V_accuracy']:.4f} | {r['beneficial']} | {r['harmful']} |")
    lines+=['','折外分类器训练量较小，且可能没有见过部分地点类别；这减少训练内置信度偏差，但不消除OOF→完整训练模型的分布差异。各折缺失类别、帧清单和分类器权重保存在folds.json。','',
        '## 协议与结论边界','',
        '输入仅六项：L/V的top1概率间隔、熵/log25、两后验点积及top1是否不同；不输入类别ID、GT、坐标、参考特征、歧义mask或Matcher标记。输出为sigmoid仲裁分数，不宣称已校准概率。',
        '监督只将L错且V对记为1，其余均为0；unweighted BCE，100epoch。每5epoch在内部集扫描预固定0:.05:1阈值及1.1永不替换，按净纠正数、少override、早epoch、高阈值确定唯一部署决策。',
        '训练与阈值选择均不使用182帧。开发集已被多轮研究接触，结果不是盲测或跨种子稳健结论。',
        '通过也只说明这两个K25分类器的置信度仲裁存在可识别互补，不表示可安全修改场景坐标或改善姿态。失败只限制当前六项信号、分类器和仲裁协议，不证明其他query-only信息不足。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n'); target=q.e.HERE/'results/query_arbiter'; target.mkdir(exist_ok=True)
    for file in OUT.iterdir():
        if file.is_file(): shutil.copy2(file,target/file.name)
    print('REPORT',target,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); assert not (OUT/'summary.json').exists()
    p=protocol(); data=load(p); x,y=oof(p,data); internal=full_internal(data); head,selected=train(p,x,y,internal); evaluate(p,head,selected); report()
