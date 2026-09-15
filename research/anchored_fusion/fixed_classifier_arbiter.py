import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import query_arbiter as a

OUT=Path('/home/zhang/fixed-classifier-arbitration')


def protocol():
    old=json.loads((a.OUT/'protocol.json').read_text()); rows=[r for r in old['rows'] if r['role']=='internal']; fold={}
    for date in sorted({r['session_id'] for r in rows}):
        same=[r for r in rows if r['session_id']==date]
        for f,idx in enumerate(np.array_split(np.arange(len(same)),3)):
            for i in idx: fold[same[i]['frame_id']]=f
    p=dict(rows=rows,fold=fold,seed=2089,epochs=100,checkpoints=list(range(0,101,5)),thresholds=old['thresholds'],
        classifier='Frozen prior full578 L/V classifiers, identical weights for arbiter training, OOF selection and development. Reuse exact saved posterior arrays; no classifier training.',
        crossfit='Only arbiter: three contiguous thirds per date among145 internal frames, train other2/3; no extra embargo, no pointwise splitting. Classifiers previously selected on these145 frames; not a fresh independent validation set.',
        features=old['inputs'],labels=old['labels'],model=old['model'],training=old['training'],
        selection='Merge all145 frame OOF scores at each checkpoint. Maximize rescue-damage on all valid points over fixed epoch/threshold grid; ties fewer overrides, earlier epoch, higher threshold. Never override remains eligible.',
        refit='Train same arbiter from same seed on all145 for complete100epochs; retain weights at OOF-selected epoch and fixed threshold. No refit-in-sample or development selection.',
        evaluation='182 development after refit and threshold lock; net>0 required overall and originalMatcher intersection. Report ambiguity, dates and trajectory-block intervals. No development threshold scan.',
        limitations='Same classifier mapping does not imply identical trajectory or confidence distribution; OOF arbiters and final arbiter have different training sample sizes. Success does not isolate classifier shift as sole previous cause, or establish pose gain. Failure limited to this6signal,K25,MLP,BCE,grid protocol.')
    OUT.mkdir(exist_ok=True); a.q.e.run.save_json(OUT/'protocol.json',p)
    return p


def prepare(p):
    sources=json.loads((a.OUT/'sources.json').read_text())
    assert all(a.q.e.run.digest(a.q.OUT/arm/'best.pt')==sources[arm] for arm in ['L','V'])
    with np.load(a.OUT/'internal.npz') as f:
        data={k:f[k] for k in f.files}
    assert np.array_equal(a.signals(data['L'],data['V']),data['features'])
    stats={r['frame_id']:r for r in json.loads((a.q.OUT/'data_stats.json').read_text())}; frames=[]; start=0
    for r in p['rows']:
        fid=r['frame_id']; stop=start+stats[fid]['points']; frames.append(dict(frame_id=fid,date=r['session_id'],start=start,stop=stop,fold=p['fold'][fid])); start=stop
    assert len(frames)==145 and start==len(data['features'])==19030
    fold=np.concatenate([np.full(r['stop']-r['start'],r['fold'],np.int8) for r in frames]); np.savez_compressed(OUT/'internal.npz',**data,fold=fold)
    a.q.e.run.save_json(OUT/'internal_frames.json',frames)
    a.q.e.run.save_json(OUT/'sources.json',dict(**sources,internal=a.q.e.run.digest(a.OUT/'internal.npz')))
    return data,fold


def fit(x,y,p,predict=None,selected_epoch=None):
    torch.manual_seed(2089); model=a.Arbiter().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    assert sum(v.numel() for v in model.parameters())==129
    rng=np.random.default_rng(2089); x=torch.tensor(x,device='cuda'); y=torch.tensor(y,dtype=torch.float32,device='cuda'); logs=[]; scores={}; states={}
    px=torch.tensor(predict,device='cuda') if predict is not None else None
    for epoch in range(101):
        if epoch:
            lr=.001*epoch/5 if epoch<=5 else .00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/95))/2
            optimizer.param_groups[0]['lr']=lr; order=rng.permutation(len(y)); total=0.
            for start in range(0,len(y),4096):
                idx=torch.tensor(order[start:start+4096],device='cuda'); optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(x[idx]),y[idx]); assert torch.isfinite(loss); loss.backward(); optimizer.step(); total+=len(idx)*loss.item()
            logs.append(dict(epoch=epoch,loss=total/len(y),lr=lr))
        if epoch in p['checkpoints']:
            if px is not None:
                with torch.no_grad(): scores[epoch]=model(px).sigmoid().cpu().numpy()
            states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if selected_epoch is not None: model.load_state_dict(states[selected_epoch])
    return model,scores,states,logs


def crossfit(p,data,fold):
    x=data['features']; y=data['benefit']; merged=np.full((len(p['checkpoints']),len(x)),np.nan,np.float32); metadata=[]
    for f in range(3):
        train=fold!=f; held=~train
        _,score,states,logs=fit(x[train],y[train],p,predict=x[held])
        for i,epoch in enumerate(p['checkpoints']): merged[i,held]=score[epoch]
        torch.save(states,OUT/f'fold{f}_checkpoints.pt'); a.q.e.run.save_json(OUT/f'fold{f}_training.json',logs)
        metadata.append(dict(fold=f,train_points=int(train.sum()),held_points=int(held.sum()),train_frames=[r['frame_id'] for r in p['rows'] if p['fold'][r['frame_id']]!=f],held_frames=[r['frame_id'] for r in p['rows'] if p['fold'][r['frame_id']]==f]))
        print('FOLD COMPLETE',metadata[-1]['fold'],metadata[-1]['train_points'],metadata[-1]['held_points'],flush=True)
    assert np.isfinite(merged).all()
    np.savez_compressed(OUT/'oof_scores.npz',scores=merged,epochs=p['checkpoints'])
    a.q.e.run.save_json(OUT/'folds.json',metadata)
    grid=[]
    for i,epoch in enumerate(p['checkpoints']):
        for threshold in p['thresholds']:
            r=a.counts(merged[i],threshold,x,data['benefit'],data['harm'],np.ones(len(x),bool)); r.update(epoch=epoch,threshold=threshold); grid.append(r)
    selected=max(grid,key=lambda r:(r['net'],-r['overrides'],-r['epoch'],r['threshold']))
    a.q.e.run.save_json(OUT/'oof_grid.json',grid); a.q.e.run.save_json(OUT/'selection.json',selected)
    print('SELECTED',selected,flush=True)
    model,_,states,logs=fit(x,y,p,selected_epoch=selected['epoch'])
    torch.save(states[selected['epoch']],OUT/'best.pt'); torch.save(states[100],OUT/'last.pt'); a.q.e.run.save_json(OUT/'final_training.json',logs)
    return model,selected


def evaluate(p,model,selected):
    with np.load(a.OUT/'development.npz') as f:
        y=f['target']; x=f['features']; l=f['L']; v=f['V']; benefit=f['benefit']; harm=f['harm']
        masks={k:f[k] for k in ['all','ambiguous','matcher','ambiguous_matcher']}
    assert np.array_equal(a.signals(l,v),x)
    with torch.no_grad(): score=model(torch.tensor(x,device='cuda')).sigmoid().cpu().numpy()
    override=(score>=selected['threshold'])&(x[:,-1]>0); base=l.argmax(1)==y; final=np.where(override,v.argmax(1),l.argmax(1))==y
    frames=json.loads((a.OUT/'development_frames.json').read_text()); summary=dict(selection=selected,groups={},dates={})
    for name,mask in masks.items():
        r=a.counts(score,selected['threshold'],x,benefit,harm,mask); r.update(lidar_accuracy=float(base[mask].mean()),final_accuracy=float(final[mask].mean()))
        use=[f for f in frames if mask[f['start']:f['stop']].any()]
        delta=np.array([(final[f['start']:f['stop']][mask[f['start']:f['stop']]].astype(float)-base[f['start']:f['stop']][mask[f['start']:f['stop']]].astype(float)).mean() for f in use])
        r['paired_frame_accuracy']=a.block_interval(delta,use); summary['groups'][name]=r
    for date in sorted({r['date'] for r in frames}):
        mask=np.zeros(len(y),bool)
        for f in frames:
            if f['date']==date: mask[f['start']:f['stop']]=True
        summary['dates'][date]={name:a.counts(score,selected['threshold'],x,benefit,harm,mask&m) for name,m in masks.items()}
    summary['net_criterion_passed']=summary['groups']['all']['net']>0 and summary['groups']['matcher']['net']>0
    np.savez_compressed(OUT/'development.npz',target=y,features=x,L=l,V=v,score=score,override=override,benefit=benefit,harm=harm,**masks)
    a.q.e.run.save_json(OUT/'development_frames.json',frames); a.q.e.run.save_json(OUT/'summary.json',summary)
    print('RESULT',json.dumps(summary),flush=True)


def verify(p):
    with np.load(OUT/'internal.npz') as f: x=f['features']; benefit=f['benefit']; harm=f['harm']; fold=f['fold']; l=f['L']; v=f['V']; y=f['target']
    assert np.array_equal(x,a.signals(l,v)); b,h=a.outcomes(y,l,v); assert np.array_equal(b,benefit) and np.array_equal(h,harm)
    assert np.array_equal(x,np.load(a.OUT/'internal.npz')['features'])
    scores=np.load(OUT/'oof_scores.npz')['scores']; seen=[]
    for f in json.loads((OUT/'folds.json').read_text()):
        assert not set(f['train_frames'])&set(f['held_frames']); seen+=f['held_frames']; held=fold==f['fold']
        states=torch.load(OUT/f"fold{f['fold']}_checkpoints.pt",map_location='cpu'); model=a.Arbiter()
        for i,epoch in enumerate(p['checkpoints']):
            model.load_state_dict(states[epoch])
            with torch.no_grad(): predicted=model(torch.tensor(x[held])).sigmoid().numpy()
            assert np.allclose(predicted,scores[i,held],atol=1e-6)
        logs=json.loads((OUT/f"fold{f['fold']}_training.json").read_text()); assert [r['epoch'] for r in logs]==list(range(1,101))
    assert len(seen)==len(set(seen))==145
    grid=[]
    for i,epoch in enumerate(p['checkpoints']):
        for threshold in p['thresholds']:
            use=(scores[i]>=threshold)&(x[:,-1]>0); rescue=int((use&benefit).sum()); damage=int((use&harm).sum())
            grid.append((rescue-damage,-int(use.sum()),-epoch,threshold))
    selected=json.loads((OUT/'selection.json').read_text()); assert max(grid)==(selected['net'],-selected['overrides'],-selected['epoch'],selected['threshold'])
    model=a.Arbiter(); model.load_state_dict(torch.load(OUT/'best.pt',map_location='cpu'))
    with np.load(OUT/'development.npz') as f:
        with torch.no_grad(): s=model(torch.tensor(f['features'])).sigmoid().numpy()
        assert np.allclose(s,f['score'],atol=1e-6)
        use=(f['score']>=selected['threshold'])&(f['features'][:,-1]>0); assert np.array_equal(use,f['override'])
        summary=json.loads((OUT/'summary.json').read_text())
        for name in ['all','ambiguous','matcher','ambiguous_matcher']:
            rescue=int((use&f[name]&f['benefit']).sum()); damage=int((use&f[name]&f['harm']).sum()); r=summary['groups'][name]
            assert (rescue,damage,rescue-damage)==(r['rescue'],r['damage'],r['net'])
    logs=json.loads((OUT/'final_training.json').read_text()); assert [r['epoch'] for r in logs]==list(range(1,101))
    a.q.e.run.save_json(OUT/'verification.json',dict(same_frozen_classifier_posteriors=True,frames_held_once=145,all63OOFcheckpoint_predictions_recomputed=True,selection_recomputed=True,development_counts_recomputed=True,fold_and_final_epochs=100,parameters=129,best_sha256=a.q.e.run.digest(OUT/'best.pt')))


def report():
    s=json.loads((OUT/'summary.json').read_text()); selected=s['selection']; grid=json.loads((OUT/'oof_grid.json').read_text())
    active=[r for r in grid if r['overrides']]; best_active=max(active,key=lambda r:(r['net'],-r['overrides'],-r['epoch'],r['threshold']))
    lines=['# 固定分类器的仲裁层交叉拟合','',
        '**本轮仍未通过：固定同一套完整分类器后，折外选择仍为永不替换；开发集整体及Matcher交集救回0、损害0、净变化0。** 这是回退到原LiDAR分类，而不是安全纠正成功。',
        f"在预固定{len(grid)}个轮数/阈值组合中，{len(active)}个产生非空替换，其中净收益为正的配置数为{sum(r['net']>0 for r in active)}，净收益为零的配置数为{sum(r['net']==0 for r in active)}。没有据开发集重新挑阈值，也没有扩展网格追求正结果。",'',
        '本轮仅对129参数仲裁器进行交叉拟合和重新训练。训练、OOF选模及开发评估使用同一套既有完整578帧L/V分类器的原始后验；六个输入、标签、BCE、网络结构和优化器均与上一轮相同。没有重训分类器、没有改变类别权重或扫描开发集阈值。','',
        f"145帧分为每日期三个连续轨迹段，三折各训练100epoch，合并OOF分数选择epoch={selected['epoch']}、threshold={selected['threshold']:.2f}；OOF救回{selected['rescue']}、损害{selected['damage']}、净变化{selected['net']}。随后全部145帧从同一初始化训练100epoch，仅保留OOF选定轮数的权重。",'',
        f"OOF预固定462个轮数/阈值组合；最优非空策略：epoch{best_active['epoch']}，threshold{best_active['threshold']:.2f}，override{best_active['overrides']}，救回{best_active['rescue']}，损害{best_active['damage']}，净变化{best_active['net']}。平局依次偏好少替换、早轮数、高阈值；1.1表示永不替换。",'',
        '| 开发集合 | 点数 | override | 救回 | 损害 | 中性 | 净变化 | 原准确率 % | 最终准确率 % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in s['groups'].items(): lines.append(f"| {name} | {r['count']} | {r['overrides']} | {r['rescue']} | {r['damage']} | {r['neutral']} | {r['net']} | {100*r['lidar_accuracy']:.4f} | {100*r['final_accuracy']:.4f} |")
    lines+=['','| 集合 | 帧均准确率差 pp | 轨迹块95%区间 pp |','|---|---:|---|']
    for name,r in s['groups'].items():
        c=r['paired_frame_accuracy']; lines.append(f"| {name} | {100*c['mean']:.5f} | [{100*c['ci95'][0]:.5f}, {100*c['ci95'][1]:.5f}] |")
    lines+=['','帧均区间与按点加权净变化不同；日期内轨迹块bootstrap10000次，seed271828。永不替换时零区间为恒等决策结果，不代表安全纠正或统计等效。','',
        '| 日期 | 集合 | 救回 | 损害 | 净变化 |','|---|---|---:|---:|---:|']
    for date,groups in s['dates'].items():
        for name in ['all','matcher','ambiguous','ambiguous_matcher']:
            r=groups[name]; lines.append(f"| {date} | {name} | {r['rescue']} | {r['damage']} | {r['net']} |")
    lines+=['','## 判据和解释边界','',
        '整体及原Matcher交集的净变化均大于零，满足本轮计数判据；仍需结合区间和逐日期一致性，不能直接宣布稳健或姿态收益。' if s['net_criterion_passed'] else '整体和原Matcher交集没有同时取得正净变化，本轮预定判据未通过。',
        '同一分类器消除了上一轮更换分类权重的问题，但不等于145帧与182帧具有相同数据/置信度分布。交叉拟合仲裁器与全量重训仲裁器的训练样本量也不同，不能忽略此差异。',
        '145帧此前用于分类器选模，本轮又用于仲裁训练和OOF选模；182帧已在多轮研究中接触，均不是新盲测。本轮不再只使用578帧标签训练最终系统，额外使用145帧标签训练仲裁器。',
        '只在既定21个分数阈值加永不替换、21个检查点范围内判断；未评估的阈值或其他仲裁模型不在否定范围。若成功，也不能将前后差异唯一归因于旧OOF分类器的分布失配，因为仲裁训练样本来源同时改变。',
        '本轮不改变LEADER，不运行Matcher求位姿；交集复用已核验的原候选mask，仅作评估。任何结论均限于K25分类器的六项置信度仲裁，不外推所有query-only融合。','',
        '## 复现','',
        'fixed_classifier_arbiter.py保存协议、固定后验、三折全部63个检查点与OOF分数、合并网格、最终权重、开发逐点决策和验证。验证重新计算全部折外预测、轮数/阈值选择及开发净纠正数。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n'); target=a.q.e.HERE/'results/fixed_classifier_arbiter'; target.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        if f.is_file(): shutil.copy2(f,target/f.name)
    print('REPORT',target,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); assert not (OUT/'summary.json').exists()
    p=protocol(); data,fold=prepare(p); model,selected=crossfit(p,data,fold); evaluate(p,model,selected); verify(p); report()
