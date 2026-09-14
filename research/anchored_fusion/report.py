import json
import shutil
import numpy as np
from experiment import OUT, HERE, CACHE, SEEDS, run


def read(path):
    return json.loads(path.read_text())


def aggregate(records):
    result = dict(standard=run.metrics([r['standard'] for r in records]),fixed=run.metrics([r['fixed'] for r in records]))
    if 'mechanism' in records[0]:
        m = {k:sum(r['mechanism'][k] for r in records) for k in ['count','correct','rescue','damage']}
        m['accuracy'] = m['correct']/max(m['count'],1)
        result['mechanism'] = m
        result['groups'] = {}
        for name in ['matcher','protected','editable']:
            g = {k:sum(r['mechanism_groups'][name][k] for r in records) for k in ['count','correct','rescue','damage']}
            g['accuracy'] = g['correct']/max(g['count'],1)
            result['groups'][name] = g
        protected = sum(r['protected_count'] for r in records)
        result['protected_retention'] = sum(r['protected_retained'] for r in records)/protected
        result['matcher_count'] = sum(r['matcher_count'] for r in records)
        result['protected_count'] = protected
        count = sum(r['selected_visible']['count'] for r in records)
        result['selected_visible'] = dict(count=count,mean_error_delta=sum(r['selected_visible']['delta_sum'] for r in records)/count,
                                         harm_rate=sum(r['selected_visible']['harmed'] for r in records)/count)
    return result


def block_interval(delta, records):
    by_date = {}
    for date in sorted({r['date'] for r in records}):
        indices = [i for i,r in enumerate(records) if r['date']==date]
        blocks, current = [], []
        for index in indices:
            if current and (len(current)==10 or int(records[index]['frame_id'])-int(records[current[-1]]['frame_id'])>10000000):
                blocks.append(current)
                current = []
            current.append(index)
        if current:
            blocks.append(current)
        by_date[date] = blocks
    rng = np.random.default_rng(271828)
    totals, counts = np.zeros(10000),np.zeros(10000)
    for blocks in by_date.values():
        sums = np.array([delta[b].sum() for b in blocks])
        sizes = np.array([len(b) for b in blocks])
        indices = rng.integers(len(blocks),size=(10000,len(blocks)))
        totals += sums[indices].sum(1)
        counts += sizes[indices].sum(1)
    return dict(mean=float(delta.mean()),ci95=np.percentile(totals/counts,[2.5,97.5]).tolist(),blocks={d:len(b) for d,b in by_date.items()})


def main():
    baseline = read(OUT/'baseline_development.json')
    line1 = read(OUT/'line1_development.json')
    summary = dict(baseline=aggregate(baseline),line1=aggregate(line1),runs={})
    dates = sorted({r['date'] for r in baseline})
    summary['baseline_dates'] = {d:aggregate([r for r in baseline if r['date']==d]) for d in dates}
    values = {}
    for seed in SEEDS:
        for arm in ['aligned','shuffled']:
            name = f'{arm}_{seed}'
            folder = OUT/name
            records = read(folder/'development.json')
            assert [r['frame_id'] for r in records]==[r['frame_id'] for r in baseline]
            logs = read(folder/'training.json')
            assert [r['epoch'] for r in logs]==list(range(1,101))
            assert read(folder/'complete.json')==dict(epochs=100,updates=7300)
            choices = [(0,np.mean([r['standard'][0] for r in read(OUT/'baseline_internal.json')]))]
            choices += [(r['epoch'],r['internal']['mean'][0]) for r in logs if 'internal' in r]
            selection = read(folder/'selection.json')
            assert selection['epoch']==min(choices,key=lambda x:x[1])[0]
            result = aggregate(records)
            result['checkpoint_sha256'] = run.digest(folder/'best.pt')
            result['last_checkpoint_sha256'] = run.digest(folder/'last.pt')
            result['selection'] = selection
            current_errors = np.asarray([r['standard'] for r in records])
            baseline_errors = np.asarray([r['standard'] for r in baseline])
            current_success = (current_errors[:,0]<1)&(current_errors[:,1]<5)
            baseline_success = (baseline_errors[:,0]<1)&(baseline_errors[:,1]<5)
            result['pose_rescue'] = int((current_success&~baseline_success).sum())
            result['pose_damage'] = int((~current_success&baseline_success).sum())
            result['dates'] = {d:aggregate([r for r in records if r['date']==d]) for d in dates}
            if selection['epoch']==0:
                assert [r['standard'] for r in records]==[r['standard'] for r in baseline]
                assert result['mechanism']['rescue']==0 and result['mechanism']['damage']==0
            assert result['groups']['protected']['rescue']==0 and result['groups']['protected']['damage']==0
            summary['runs'][name] = result
            values[name] = np.array([r['standard'] for r in records])
    a = np.mean([values[f'aligned_{s}'] for s in SEEDS],axis=0)
    s = np.mean([values[f'shuffled_{s}'] for s in SEEDS],axis=0)
    b = np.array([r['standard'] for r in baseline])
    summary['paired_translation'] = dict(baseline=block_interval(a[:,0]-b[:,0],baseline),
                                         shuffled=block_interval(a[:,0]-s[:,0],baseline))
    passed = True
    for seed in SEEDS:
        a,s = [summary['runs'][f'{arm}_{seed}'] for arm in ['aligned','shuffled']]
        b = summary['baseline']
        passed &= a['standard']['mean'][0]<min(b['standard']['mean'][0],s['standard']['mean'][0])
        passed &= a['mechanism']['accuracy']>max(b['mechanism']['accuracy'],s['mechanism']['accuracy'])
        for date in dates:
            passed &= a['dates'][date]['standard']['mean'][0]<min(summary['baseline_dates'][date]['standard']['mean'][0],s['dates'][date]['standard']['mean'][0])
    for key,index in [('mean',1),('p95',0)]:
        passed &= not all(summary['runs'][f'aligned_{seed}']['standard'][key][index]>summary['baseline']['standard'][key][index] for seed in SEEDS)
    passed &= all(x['ci95'][1]<0 for x in summary['paired_translation'].values())
    summary['passed'] = bool(passed)
    run.save_json(OUT/'summary.json',summary)
    run.save_json(OUT/'source_provenance.json',dict(source={p.name:run.digest(p) for p in HERE.glob('*.py')},
                                                  protocol_sha256=run.digest(OUT/'protocol.json'),audit_sha256=run.digest(OUT/'audit.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    base_internal = np.mean([r['standard'][0] for r in read(OUT/'baseline_internal.json')])
    axes[0].axhline(base_internal,color='black',linestyle=':',label='Original LEADER')
    for color,seed in zip(['tab:blue','tab:orange','tab:green'],SEEDS):
        for arm,style in [('aligned','-'),('shuffled','--')]:
            logs = read(OUT/f'{arm}_{seed}'/'training.json')
            checkpoints = [(0,base_internal)]+[(r['epoch'],r['internal']['mean'][0]) for r in logs if 'internal' in r]
            axes[0].plot(*zip(*checkpoints),color=color,linestyle=style,label=f'{arm} {seed}',linewidth=1.2)
            selected = summary['runs'][f'{arm}_{seed}']['selection']
            axes[0].scatter([selected['epoch']],[selected['mean_translation']],color=color,marker='*',s=65)
            axes[1].plot([r['epoch'] for r in logs],[r['loss'][2] for r in logs],color=color,linestyle=style,linewidth=1.2)
    axes[0].set(ylabel='Internal mean translation error (m)',xlabel='Epoch',title='Internal pose selection (* = selected)')
    axes[1].set(ylabel='Training contrastive loss',xlabel='Epoch',title='Full 100-epoch training')
    axes[0].legend(fontsize=7,ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'learning.png',dpi=160)
    plt.close(fig)
    lines = ['# 跨帧对比监督的锚定残差融合', '',
             f'**固定联合判据：{"通过" if passed else "未通过"}。** 排序读数与最终定位分别报告，不把匹配改善当作定位提升。', '',
             '## 协议与核验', '',
             '使用统一版本 905 帧缓存：578 帧训练，145 帧内部选模，182 帧固定开发评估；对比参考库仅来自训练部分。三个种子 × 正确／置乱两组，每组完整 100 epoch、7300 次更新。检查点只在内部最终平均位置误差上从 0、10、…、100 轮选择；零初始化可以获选，没有挑最佳种子。', '',
             '37,408 参数的 640→32→512 小 MLP，只加范数不超过原特征 5% 的残差。原 top-K 最高可靠的一半及缺图点保持特征与回归输出严格不变；其他点可能改变可靠度排序，所以另记保护点最终保留率。没有收益门控、额外匹配头或回归头解冻。', '',
             '训练目标为原 TRR + 0.01 倍温度 0.1 的多正例对比损失。查询是融合特征，参考是冻结原 LiDAR 特征；模糊距离不作负例。训练既含原排序正确点，也含错误点。原 coarse voxel 监督目标与真实代表点空间标签分开。结构、梯度、距离模糊标签忽略及 905 帧零初始化恒等性检查通过。', '',
             '![内部选模与完整训练曲线](learning.png)', '',
             '## 182 帧最终定位', '',
             '| 方法 | 内部选中 epoch | 平均位置 m | 平均旋转 ° | 位置 P95 m | 固定原候选位置 m | 成功帧 1m/5° |',
             '|---|---:|---:|---:|---:|---:|---:|']
    entries = [('纯 LEADER',summary['baseline'],'—'),('历史①',summary['line1'],'历史权重')]
    entries += [(n,r,str(r['selection']['epoch'])) for n,r in summary['runs'].items()]
    for name,r,epoch in entries:
        m = r['standard']
        lines.append(f'| {name} | {epoch} | {m["mean"][0]:.6f} | {m["mean"][1]:.6f} | {m["p95"][0]:.6f} | {r["fixed"]["mean"][0]:.6f} | {m["successes"]}/182 |')
    lines += ['', '纯 LEADER 和①均在当前 182 帧上重新计算，未沿用旧 32 帧数值。①为旧训练权重的历史参考，其训练数据和预算不同。', '',
              '## 固定候选消歧及保护', '',
              '| 方法 | Top-1 | 纠正 | 损害 | 保护点最终保留率 | 原入选可见点坐标误差变化 m |', '|---|---:|---:|---:|---:|---:|']
    for name,r in [('纯 LEADER',summary['baseline'])]+list(summary['runs'].items()):
        m = r['mechanism']
        lines.append(f'| {name} | {m["accuracy"]:.2%} | {m["rescue"]} | {m["damage"]} | {r["protected_retention"]:.2%} | {r["selected_visible"]["mean_error_delta"]:+.6f} |')
    b = summary['baseline']
    lines += ['', f'歧义点共 {b["mechanism"]["count"]} 个，其中 {b["groups"]["matcher"]["count"]} 个属于原 Matcher 候选、{b["groups"]["protected"]["count"]} 个属于保护集合。原 Matcher 候选总数 {b["matcher_count"]}，故歧义点占其 {b["groups"]["matcher"]["count"]/b["matcher_count"]:.2%}。分组排序结果详见 summary.json。', '',
              f'允许改变的原 Matcher 歧义点为 {b["groups"]["matcher"]["count"]-b["groups"]["protected"]["count"]} 个，占原 Matcher 候选 {(b["groups"]["matcher"]["count"]-b["groups"]["protected"]["count"])/b["matcher_count"]:.2%}；这个交集不能等同于全部图像有效点上的歧义比例。', '',
              '这里参考库从探针的 723 帧缩为仅训练的 578 帧，因此重新固定 LiDAR 前 16 候选及排序基线，不能直接沿用上一轮 7200 点的正确率。特征完全不变的查询保留原候选顺序，避免矩阵乘法与逐项乘加的舍入差异造成假纠正。', '',
              '## 各日期平均位置误差 m', '', '| 方法 | '+ ' | '.join(dates)+' |', '|---|'+'---:|'*len(dates)]
    for name,by_date in [('纯 LEADER',summary['baseline_dates'])]+[(n,r['dates']) for n,r in summary['runs'].items()]:
        lines.append('| '+name+' | '+' | '.join(f'{by_date[d]["standard"]["mean"][0]:.6f}' for d in dates)+' |')
    lines += ['', '## 轨迹块配对不确定性', '',
              '三个种子取逐帧均值，在每日期内对连续最多 10 帧的轨迹块重采样，超过 10 秒断开；10000 次、种子271828。负的位置差表示正确视觉更好。', '',
              '```json',json.dumps(summary['paired_translation'],indent=2),'```', '',
              '判据要求三个正确视觉种子均优于纯 LEADER 与配对置乱，并在每日期同向；配对区间上界低于零；旋转均值和位置 P95 不出现三个种子一致退化；排序也须胜过原 LiDAR 与置乱。', '',
              '## 本轮解读', '',
              '正确视觉只有种子2089的平均位置误差优于纯 LEADER，另两个种子略差；三种子均值相对 baseline 只低约 0.10 mm，轨迹块区间包含零。相对置乱，正确视觉三种子均值反而高约 1.24 mm，也未建立稳定优势。三个正确视觉模型的平均旋转误差均略高于 baseline。', '',
              '匹配排序净纠正分别只有 +4、+11、+9，置乱为 +4、+15、+6；因此不能把这次小幅排序变化归因于正确的逐点视觉关联。前一轮直接视觉排序的阳性证据仍然成立，但当前锚定残差未将其转成稳定的融合定位收益。', '',
              '35,502 个保护点在六组评估中均保持特征、回归输出不变，且最终候选保留率实际均为 100%；该结构性质成立，但不等于其余对应或整个定位流程不退化。', '',
              '历史①在新182帧上平均位置为 0.119545 m，确实低于纯 LEADER 的 0.122615 m；不能把旧32帧的负结果直接外推到这个集合。但本轮没有为①补齐同预算三种子置乱对照，这个历史参考结果不构成稳健视觉贡献的证明。', '',
              '本轮保留纯 LEADER 作为基线，不继续调整当前方案的幅度、温度或损失系数来追逐这182帧。', '',
              '## 边界', '',
              '182 帧已参与前一轮消歧探针和设计选择，不是盲测。PCA 复用训练日期图像拟合的版本。本实验不使用检索库做推理定位，参考库只用于训练监督及事后机制评价。即使排序改善，也不能据此宣布定位提升或完整 NCLT 改善。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    target = HERE/'results'
    for p in OUT.iterdir():
        if p.name=='candidates':
            continue
        if p.is_dir():
            shutil.copytree(p,target/p.name,dirs_exist_ok=True)
        else:
            shutil.copy2(p,target/p.name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
