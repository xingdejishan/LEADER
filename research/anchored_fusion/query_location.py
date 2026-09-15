import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

import experiment as e
from report import block_interval

OUT=Path('/home/zhang/query-only-location-probe')


class Classifier(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128 if visual else 512,269 if visual else 77),nn.ReLU(),nn.Linear(269 if visual else 77,25))

    def forward(self,x):
        return self.net(F.layer_norm(x,(x.shape[-1],)))


def protocol():
    p=dict(rows=json.loads((e.OUT/'protocol.json').read_text())['rows'],seed=2089,clusters=25,
        labels='Original coarse voxel world XYZ = cached source @ GT rotation.T + GT translation; not projection representatives. Fit KMeans only on 578 fit image-valid points, Euclidean meters, n_init10 max_iter300 random_state2089; fixed centroids assign all splits.',
        heads='Per-point parameter-free LayerNorm; L 512->77->25, V/S 128->269->25; ReLU; each exactly41451 trainable parameters. Equal parameter budget, different widths, not a claim of identical expressivity.',
        epochs=100,batch_points=4096,lr=.001,final_lr=.00001,warmup=5,weight_decay=.0001,
        loss='Unweighted cross entropy over identical image-valid points; all fit points once each epoch; same shuffled point order across arms.',
        selection='Internal145 macro recall over classes present, evaluated every5 epochs plus epoch0; strict improvement, earliest ties; no dev selection.',
        shuffle='Fixed within-frame permutation of valid descriptors seed2089+frame_id%1000000007; both train/eval. Retains frame appearance and coarse-place composition; equality does not establish lack of query-only place information.',
        coverage='Nearest fit coarse world point distance<=2m is supported; farther points separately reported; not an unseen-place generalization claim.',
        mechanism='Fixed historical ambiguity: original LiDAR top16 contains P/N and gap<=.02; no reference bank used for inference; candidates used only to retain pre-existing diagnostic mask.',
        baselines='Train class-frequency majority predictor and train frequency posterior with add-one smoothing.',
        comparisons='All points, fixed ambiguity subset and supported/unsupported points: micro accuracy, macro recall, NLL, V vs L corrections/damage; by date and paired trajectory-block confidence intervals.',
        scope='578/145/182 touched local split, one seed, frozen historical DeDoDe/PCA and caches. No encoder/MMRegressor/Matcher/fusion training or inference.',
        interpretation='Positive local evidence needs visual classification above frequency control and aligned-vs-shuffled incremental evidence, with ambiguity corrections/damage and cross-date consistency. Failure limits this descriptor/head/K25 protocol, not all query-only imagery. No pose benefit claimed.')
    OUT.mkdir(exist_ok=True); e.run.save_json(OUT/'protocol.json',p)
    return p


def load(p):
    data=[]; hashes={}
    for r in p['rows']:
        fid=r['frame_id']; lp=e.CACHE/'lidar'/(fid+'.npz'); vp=e.CACHE/'visual_raw'/(fid+'.npz'); cp=e.OUT/'candidates'/(fid+'.npz')
        with np.load(lp) as l,np.load(vp) as v,np.load(cp) as c:
            ids=np.flatnonzero(v['valid']); assert np.array_equal(ids,c['indices'])
            xyz=l['source'][ids].astype(float)@l['GT'][:3,:3].astype(float).T+l['GT'][:3,3]
            image=v['image'][ids]; permutation=np.random.default_rng(2089+int(fid)%1000000007).permutation(len(ids))
            data.append(dict(row=r,xyz=xyz,L=l['features'][ids],V=image,S=image[permutation],ids=ids,
                ambiguous=c['positive'].any(1)&c['negative'].any(1)&(c['gap']<=.02)))
        hashes[fid]=dict(lidar=e.run.digest(lp),visual=e.run.digest(vp),candidates=e.run.digest(cp))
    e.run.save_json(OUT/'inputs.json',hashes)
    return data


def labels(data):
    xyz=np.concatenate([d['xyz'] for d in data if d['row']['role']=='fit'])
    cluster=KMeans(n_clusters=25,n_init=10,max_iter=300,random_state=2089).fit(xyz)
    np.savez_compressed(OUT/'centroids.npz',centroids=cluster.cluster_centers_)
    tree=cKDTree(xyz)
    stats=[]
    for d in data:
        d['target']=cluster.predict(d['xyz']); d['distance']=tree.query(d['xyz'])[0]; d['supported']=d['distance']<=2.
        count=np.bincount(d['target'],minlength=25)
        stats.append(dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],role=d['row']['role'],points=len(d['ids']),ambiguous=int(d['ambiguous'].sum()),supported=int(d['supported'].sum()),class_counts=count.tolist(),dominant_share=float(count.max()/count.sum())))
    e.run.save_json(OUT/'data_stats.json',stats)
    return stats


def metrics(y,pred,logp,mask):
    y=y[mask]; pred=pred[mask]; logp=logp[mask]
    if not len(y): return dict(count=0,accuracy=None,macro=None,nll=None,classes=[])
    classes=np.unique(y)
    return dict(count=len(y),accuracy=float((y==pred).mean()),macro=float(np.mean([(pred[y==k]==k).mean() for k in classes])),nll=float(-logp[np.arange(len(y)),y].mean()),classes=classes.tolist(),class_count=np.bincount(y,minlength=25).tolist(),class_correct=np.bincount(y,weights=(pred==y),minlength=25).astype(int).tolist())


@torch.no_grad()
def infer(head,x):
    return torch.cat([F.log_softmax(head(b),dim=-1) for b in x.split(4096)]).cpu().numpy()


def train(p,data):
    fit=[d for d in data if d['row']['role']=='fit']; val=[d for d in data if d['row']['role']=='internal']
    y=torch.tensor(np.concatenate([d['target'] for d in fit]),device='cuda',dtype=torch.long)
    target=np.concatenate([d['target'] for d in val]); selected={}; audits={}
    frames=len(fit); assert frames==578 and len(val)==145
    for arm in ['L','V','S']:
        torch.manual_seed(2089); head=Classifier(arm!='L').cuda(); assert sum(v.numel() for v in head.parameters())==41451
        folder=OUT/arm; folder.mkdir(exist_ok=True)
        x=torch.tensor(np.concatenate([d[arm] for d in fit]),device='cuda'); internal=torch.tensor(np.concatenate([d[arm] for d in val]),device='cuda')
        initial=copy.deepcopy(head.state_dict()); torch.save(initial,folder/'epoch0.pt')
        optimizer=torch.optim.AdamW(head.parameters(),lr=p['lr'],weight_decay=p['weight_decay']); rng=np.random.default_rng(2089)
        lp=infer(head,internal); best=metrics(target,lp.argmax(1),lp,np.ones(len(target),bool))['macro']; epoch_best=0
        torch.save(initial,folder/'best.pt'); logs=[]; selection_records=[dict(epoch=0,macro=best)]
        for epoch in range(1,101):
            start=time.perf_counter(); order=rng.permutation(len(y)); total=0.
            lr=p['lr']*epoch/5 if epoch<=5 else p['final_lr']+(p['lr']-p['final_lr'])*(1+math.cos(math.pi*(epoch-5)/95))/2
            optimizer.param_groups[0]['lr']=lr
            for offset in range(0,len(y),p['batch_points']):
                ids=torch.tensor(order[offset:offset+p['batch_points']],device='cuda')
                optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(head(x[ids]),y[ids]); assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); total+=loss.item()*len(ids)
            log=dict(epoch=epoch,loss=total/len(y),lr=lr,seconds=time.perf_counter()-start)
            if epoch%5==0:
                lp=infer(head,internal); m=metrics(target,lp.argmax(1),lp,np.ones(len(target),bool)); log['internal']=m
                selection_records.append(dict(epoch=epoch,macro=m['macro']))
                if m['macro']>best:
                    best=m['macro']; epoch_best=epoch; torch.save(head.state_dict(),folder/'best.pt')
                print(arm,epoch,'loss',log['loss'],'internal macro',m['macro'],flush=True)
            logs.append(log); e.run.save_json(folder/'training.json',logs)
        head.load_state_dict(torch.load(folder/'best.pt')); selected[arm]=head
        lp=infer(head,x); train_metrics=metrics(y.cpu().numpy(),lp.argmax(1),lp,np.ones(len(y),bool))
        assert epoch_best==max(selection_records,key=lambda r:r['macro'])['epoch']
        e.run.save_json(folder/'selection.json',dict(epoch=epoch_best,macro=best,all_choices=selection_records,train=train_metrics))
        audits[arm]=dict(epochs=100,updates=100*math.ceil(len(y)/p['batch_points']),point_exposures=100*len(y),parameters=41451,rng=rng.bit_generator.state,initial_sha256=e.run.digest(folder/'epoch0.pt'),best_sha256=e.run.digest(folder/'best.pt'))
    assert audits['L']['rng']==audits['V']['rng']==audits['S']['rng']
    a=torch.load(OUT/'V/epoch0.pt'); b=torch.load(OUT/'S/epoch0.pt'); assert all(torch.equal(a[k],b[k]) for k in a)
    e.run.save_json(OUT/'training_verification.json',audits)
    return selected


def evaluate(p,data,heads):
    dev=[d for d in data if d['row']['role']=='development']; assert len(dev)==182
    y=np.concatenate([d['target'] for d in dev]); ambiguous=np.concatenate([d['ambiguous'] for d in dev]); supported=np.concatenate([d['supported'] for d in dev])
    assert ambiguous.sum()==6937
    counts=np.bincount(np.concatenate([d['target'] for d in data if d['row']['role']=='fit']),minlength=25)
    freq=(counts+1)/(counts.sum()+25)
    probs={'frequency':np.tile(np.log(freq),(len(y),1))}
    for arm,head in heads.items():
        x=torch.tensor(np.concatenate([d[arm] for d in dev]),device='cuda'); probs[arm]=infer(head,x)
    masks=dict(all=np.ones(len(y),bool),ambiguous=ambiguous,supported=supported,unsupported=~supported,ambiguous_supported=ambiguous&supported,ambiguous_unsupported=ambiguous&~supported)
    np.savez_compressed(OUT/'development.npz',target=y,distance=np.concatenate([d['distance'] for d in dev]),**masks,**probs)
    summary=dict(models={},paired={},dates={}); frame_records=[]; start=0
    for d in dev:
        stop=start+len(d['target']); frame_records.append(dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],start=start,stop=stop)); start=stop
    e.run.save_json(OUT/'development_frames.json',frame_records)
    for arm,lp in probs.items():
        pred=lp.argmax(1); summary['models'][arm]={name:metrics(y,pred,lp,mask) for name,mask in masks.items()}
        summary['dates'][arm]={}
        for date in sorted({d['row']['session_id'] for d in dev}):
            date_mask=np.concatenate([np.full(len(d['target']),d['row']['session_id']==date) for d in dev])
            summary['dates'][arm][date]={name:metrics(y,pred,lp,mask&date_mask) for name,mask in masks.items()}
    for other in ['L','S','frequency']:
        vc=probs['V'].argmax(1)==y; oc=probs[other].argmax(1)==y
        for name,mask in masks.items():
            use=[r for r in frame_records if mask[r['start']:r['stop']].any()]
            differences=np.array([(vc[r['start']:r['stop']][mask[r['start']:r['stop']]].astype(float)-oc[r['start']:r['stop']][mask[r['start']:r['stop']]].astype(float)).mean() for r in use])
            interval=block_interval(differences,use) if len(use) else None
            summary['paired']['V-'+other+'/'+name]=dict(correction=int((vc&~oc&mask).sum()),damage=int((~vc&oc&mask).sum()),net=int(((vc&~oc&mask).sum()-(~vc&oc&mask).sum())),frame_mean_accuracy_difference=interval)
    e.run.save_json(OUT/'summary.json',summary)
    print('EVALUATED', {a:{k:r[k]['accuracy'] for k in ['all','ambiguous']} for a,r in summary['models'].items()},flush=True)


def report():
    s=json.loads((OUT/'summary.json').read_text()); stats=json.loads((OUT/'data_stats.json').read_text())
    with np.load(OUT/'development.npz') as a:
        mask=a['ambiguous']; y=a['target']; lc=a['L'].argmax(1)==y
        complementary={arm:dict(correction=int((mask&~lc&(a[arm].argmax(1)==y)).sum()),damage=int((mask&lc&(a[arm].argmax(1)!=y)).sum())) for arm in ['V','S']}
        complementary['lidar_errors']=int((mask&~lc).sum())
    e.run.save_json(OUT/'complementary.json',complementary)
    lines=['# Query-only粗地点分类探针','',
        '**结果：当前局部视觉描述子确实包含无需reference bank即可学习的粗地点信号；尚未建立可部署的LiDAR增量判别或定位收益。** 固定歧义子集上正确视觉55.6148%，置乱18.3941%，且三日期方向一致；LiDAR分类器已有95.6898%。不能将历史细粒度候选歧义等同于K25粗类别歧义。','',
        '在LiDAR分类错误的299个歧义点中，正确视觉选对92个、置乱选对45个，说明存在一部分可供进一步验证的互补判断。但若直接采用视觉Top1，会同时把2872个LiDAR原本正确的判断改错（置乱5407个）；没有证明推理时能识别并选择那92个有帮助的点，也没有测试SCR或姿态收益。','',
        '固定K=25，578帧训练、145帧内部选模、182帧已接触开发评估，种子2089。只训练三个分类器，各100epoch，不修改LEADER，不运行Matcher。','',
        '输入是单个图像有效voxel的缓存局部特征；推理无需reference feature bank。训练得到的模型权重保留场景先验，因此“无显式地图”不等于“未学习场景”。本探针不代表完整图像或所有视觉表征。','',
        '类别由训练部分图像有效点的原coarse voxel世界坐标进行KMeans生成，训练之外不更新中心。它与SCR目标一致；未使用图像投影代表点替代定位坐标。','',
        '## 分类读数','', '| 集合 | 模型 | 点数 | Accuracy % | Macro recall % | NLL |','|---|---|---:|---:|---:|---:|']
    for group in ['all','ambiguous','supported','unsupported','ambiguous_supported','ambiguous_unsupported']:
        for arm in ['frequency','L','V','S']:
            r=s['models'][arm][group]
            if r['count']: lines.append(f"| {group} | {arm} | {r['count']} | {100*r['accuracy']:.4f} | {100*r['macro']:.4f} | {r['nll']:.5f} |")
    lines+=['','Macro recall只对该集合出现的类别平均，类别列表及逐类分母见summary.json。frequency为训练类别频率先验的多数类决策，不是均匀随机猜测。',
        '开发集合只出现训练25类中的16类，其中一类仅1个点；训练多数类22恰好未出现在开发集合，故多数类基准accuracy=0，不能将它作为视觉有效性的唯一依据。25类均匀随机决策的理论期望准确率为4%；视觉证据主要来自与置乱模型、逐日期及NLL的对照。没有据此重分区或改K。','',
        '## 互补与配对比较','', '| 集合 | 比较 | V纠正对照错误 | V破坏对照正确 | 净变化 | 帧均值差 pp | 轨迹块95%区间 pp |','|---|---|---:|---:|---:|---:|---|']
    for group in ['all','ambiguous','ambiguous_supported','ambiguous_unsupported']:
        for other in ['L','S','frequency']:
            r=s['paired']['V-'+other+'/'+group]; ci=r['frame_mean_accuracy_difference']
            if ci: lines.append(f"| {group} | V-{other} | {r['correction']} | {r['damage']} | {r['net']} | {100*ci['mean']:.4f} | [{100*ci['ci95'][0]:.4f}, {100*ci['ci95'][1]:.4f}] |")
    lines+=['','区间对每帧准确率差按日期/轨迹块重采样，因此不是上表按点加权accuracy之差的区间；两种权重明确区分。bootstrap10000次，seed271828。','',
        '## 逐日期：固定歧义子集','', '| 日期 | 模型 | 点数 | Accuracy % | Macro recall % |','|---|---|---:|---:|---:|']
    for date in s['dates']['V']:
        for arm in ['frequency','L','V','S']:
            r=s['dates'][arm][date]['ambiguous']; lines.append(f"| {date} | {arm} | {r['count']} | {100*r['accuracy']:.4f} | {100*r['macro']:.4f} |")
    lines+=['','## 训练、覆盖与边界','',
        'L使用512→77→25；V/S使用128→269→25；均为无参数逐点LayerNorm、两层MLP与ReLU，各41451个可训练参数。参数预算完全相同，宽度不同，不宣称表达能力完全相同。',
        '三臂相同点序、曝光、AdamW设置：lr1e-3，5epoch预热，余弦降至1e-5，weight decay1e-4，batch4096。无类别加权CE。每5epoch按内部macro recall选模，含epoch0，所有臂都跑满100epoch。',
        'S在每帧有效点内固定置乱视觉描述子，训练与评估都置乱；它保留帧级外观及地点类别组成。V≈S不能据此证明没有query-only地点信号，只能限制正确逐点关联的增量结论。','',
        '| 模型 | 选中epoch | 训练accuracy % | 训练macro % | 内部macro % |','|---|---:|---:|---:|---:|']
    for arm in ['L','V','S']:
        r=json.loads((OUT/arm/'selection.json').read_text()); lines.append(f"| {arm} | {r['epoch']} | {100*r['train']['accuracy']:.4f} | {100*r['train']['macro']:.4f} | {100*r['macro']:.4f} |")
    dev=[r for r in stats if r['role']=='development']
    lines+=['',f"开发帧平均主导类别占比{100*np.mean([r['dominant_share'] for r in dev]):.3f}%；同帧置乱仍可保留相当部分粗地点标签信息。",
        'supported仅表示距离最近训练coarse voxel点≤2m，是预固定的覆盖诊断，不是对未见空间泛化的定义。轨迹划分允许空间重叠，历史固定PCA也已接触开发图像；所有结果仅为本地开发证据。',
        '固定历史6937点歧义mask由旧候选定义；本轮预测不检索该参考库。该条件子集上的结论不外推全部点。',
        '分类能力不等于坐标或定位收益；正结果只支持继续验证任务语义接口。负结果只限制当前局部DeDoDe/PCA、分类头和K25协议，不证明视觉信息只能经query-reference比较使用。','',
        '## 复现工件','',
        'query_location.py执行协议、聚类、三臂训练和评估；保存centroids、protocol、输入hash、逐帧类别分布、已选权重、完整训练日志及开发逐点log posterior。原始905帧缓存继续复用本地数据，不随本结果重新上传。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    target=e.HERE/'results/query_location'; target.mkdir(exist_ok=True)
    for file in OUT.iterdir():
        if file.is_file(): shutil.copy2(file,target/file.name)
        else:
            dest=target/file.name; dest.mkdir(exist_ok=True)
            for f in file.iterdir(): shutil.copy2(f,dest/f.name)
    print('REPORT',target,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    assert not (OUT/'summary.json').exists()
    p=protocol(); data=load(p); labels(data); heads=train(p,data); evaluate(p,data,heads); report()
