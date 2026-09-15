import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import experiment as e
from certificate_fusion import OUT


def read(p):
    return json.loads(p.read_text())


def main():
    summary=read(OUT/'summary.json'); baseline=summary['baseline']; primary_source=e.HERE/'results/reachability'
    original=read(primary_source/'records.json'); labels=read(primary_source/'candidate_labels.json')
    selected=[i for i,r in enumerate(original) if r['original']=='N' and r.get('classification')=='reachable']
    with np.load(primary_source/'witnesses.npz') as z:
        vectors={i:(z['x'+str(i)],z['k'+str(i)]) for i in selected}
    dependencies=['certificate_fusion.py','check_certificate.py','verify_teacher.py','evaluate_certificate.py','report_certificate.py','model.py','experiment.py','subspace.py']
    audit=dict(runs={},teacher=read(OUT/'teacher_verified.json'),source_hashes={name:e.run.digest(e.HERE/name) for name in dependencies})
    internal=read(OUT/'baseline_internal.json')
    for seed in e.SEEDS:
        init=[]
        for arm in ['aligned','shuffled']:
            name=f'{arm}_{seed}'; folder=OUT/name
            logs=read(folder/'training.json'); selection=read(folder/'selection.json')
            assert [r['epoch'] for r in logs]==list(range(1,101))
            assert read(folder/'complete.json')==dict(epochs=100,updates=7300)
            choices=[(0,float(np.mean([r['standard'][0] for r in internal])))]+[(r['epoch'],r['internal']['mean'][0]) for r in logs if 'internal' in r]
            assert min(choices,key=lambda x:x[1])[0]==selection['epoch']
            init.append(torch.load(folder/'epoch0.pt',map_location='cpu'))
            cov=read(folder/'coverage.json'); certs=read(folder/'coverage_certificates.json')
            with np.load(folder/'coverage_basis.npz') as z:
                q,matrix=z['Q'],z['matrix']
            assert q.shape[1] in [0,33]
            if q.shape[1]:
                assert np.linalg.norm(q.T@q-np.eye(33),2)<1e-12
                assert np.linalg.norm(matrix-q@(q.T@matrix))/np.linalg.norm(matrix)<1e-12
            else:
                assert not matrix.any()
            grouped={}; gaps=[]
            with np.load(folder/'coverage_witnesses.npz') as z:
                for c in certs:
                    rid=c['record']; x,k=vectors[rid]; r=original[rid]
                    lab=np.array(labels[r['frame_id']][r['candidate_row']]); comp=lab=='N' if c['level']=='N' else lab!='P'
                    a=k[c['positive']]-k[comp]
                    coeff,w=z['c'+c['key']],z['w'+c['key']]
                    rho=.05 if q.shape[1] else 0.
                    assert np.linalg.norm(coeff)<=rho+1e-15 and np.linalg.norm(q@coeff)<=rho+1e-15
                    assert w.min()>=0 and abs(w.sum()-1)<1e-14
                    lo=float(np.min(np.sum(a*(x+q@coeff),axis=1))-1e-10)
                    av=np.sum(a*w[:,None],axis=0)
                    hi=float(x@av+rho*np.linalg.norm(q.T@av)+1e-10)
                    assert abs(lo-c['lower'])<1e-12 and abs(hi-c['upper'])<1e-12 and lo<=hi+1e-12
                    grouped.setdefault((rid,c['level']),[]).append((c['positive'],lo,hi)); gaps.append(hi-lo)
                for row in cov:
                    rid=row['record']; r=original[rid]; lab=np.array(labels[r['frame_id']][r['candidate_row']])
                    for level in ['N','NG']:
                        items=grouped[rid,level]
                        assert sorted(v[0] for v in items)==np.flatnonzero(lab=='P').tolist()
                        assert np.allclose([max(v[1] for v in items),max(v[2] for v in items)],row['bounds'][level],atol=1e-12,rtol=0)
                    ln,un=row['bounds']['N']; lf,uf=row['bounds']['NG']
                    classification='reachable' if lf>1e-7 else 'negative_blocked' if un < -1e-7 else 'gray_blocked' if ln>1e-7 and uf < -1e-7 else 'unresolved'
                    assert classification==row['classification']
            audit['runs'][name]=dict(epochs=100,updates=7300,selected=selection['epoch'],checkpoint_sha256=e.run.digest(folder/'best.pt'),
                certificate_count=len(certs),maximum_certificate_gap=max(gaps),coverage_classifications=dict(Counter(r['classification'] for r in cov)))
        assert all(torch.equal(init[0][k],init[1][k]) for k in init[0])
    e.run.save_json(OUT/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    base_internal=float(np.mean([r['standard'][0] for r in internal]))
    axes[0].axhline(base_internal,color='black',linestyle=':',label='Pure LEADER')
    for seed,color in zip(e.SEEDS,['tab:blue','tab:orange','tab:green']):
        for arm,style in [('aligned','-'),('shuffled','--')]:
            logs=read(OUT/f'{arm}_{seed}'/'training.json')
            points=[(0,base_internal)]+[(r['epoch'],r['internal']['mean'][0]) for r in logs if 'internal' in r]
            axes[0].plot(*zip(*points),color=color,linestyle=style,label=f'{arm} {seed}')
            axes[1].plot([r['epoch'] for r in logs],[r['loss'][2] for r in logs],color=color,linestyle=style)
    axes[0].set(xlabel='Epoch',ylabel='Mean translation error (m)',title='Internal checkpoint selection')
    axes[1].set(xlabel='Epoch',ylabel='Balanced direction loss',title='Full training budget')
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'learning.png',dpi=150); plt.close(fig)
    lines=['# 训练集证书监督的残差融合','',f'固定联合判据：**{"通过" if summary["passed"] else "未通过"}**。完整执行六组100 epoch，不按开发集调参。','',
        '## 固定设计与训练目标','',
        '同一 640→32→512 小头（37,408 参数），两层和偏置从头学习，末层零初始化；没有新门控或扩容。原保护集合、5%范数限制、MMRegressor、TRR、Matcher、特征缓存和投影不变。LEADER与视觉提取器冻结。推理不读取参考库或GT。','',
        '仅用578帧训练部分生成教师，参考同样只来自这578帧；145帧内部选模，182帧已接触开发集只做固定评估。沿用原固定16候选及近重复排除；无视觉选对筛选。P≤0.5m，N≥2m，G为中间距离。','',
        '教师间隔gamma=1e-4，求解可行裕量1e-8；固定正例下最小化半平方残差范数，再用原始/对偶证据核验最小性（间隙≤1e-8）与半径≤0.05。多个正例都求解，按最小范数及固定索引选取；有未决正例则跳过该查询。明确错误且获证点得到修正目标，原正确可调整点得到零目标；灰区、无正例、不可达或未决点只参加TRR。灰区只作为确保已知正例登顶的竞争者，不重标成负例。','',
        '教师送入冻结回归头，对原Matcher候选按原coarse voxel GT检查；场景坐标误差增大的修正被剔除，不改标为零。未入选Matcher的修正不做此筛选。最小范数和逐点坐标筛选均不保证姿态改善。','',
        '损失为原TRR + 0.01*Ldir；Ldir对实际施加残差按原特征范数和0.05归一化后计算平方L2误差。修正/保持两类在每批内分别平均，再对存在的类等权平均；没有跨帧对比损失。三个种子2089/2090/2091，配对初始化与样本顺序完全相同；每组100轮、7300更新。AdamW初始1e-3，5轮预热后余弦至1e-5，weight decay1e-4。内部最终位置误差从0、10、…、100轮选模，最早并列优先；不回灌182帧。','',
        '运行中出现过WSL实例暂时不可用；后段显存接近上限时，将固定教师张量改为CPU存放并按批传入GPU，已核验与GPU拼接数值逐位相同。没有更改目标、梯度公式、批顺序或有效训练预算；断点恢复使用上一个完整epoch的模型、优化器与样本随机状态，未完成epoch的临时更新丢弃后重放。细节见operational_notes.json。','',
        '![内部选模与方向损失](learning.png)','', '## 教师可用性','',
        '```json',json.dumps(summary['teacher'],ensure_ascii=False,indent=2),'```','',
        '6299份逐正例证书通过独立复算；保留修正最小间隔0.0001000097，最大范数0.04999025，平均范数0.02694831。单约束解析解、半径不可达、分类等权、跳过标签、零初始化、梯度流和保护检查通过。教师目标文件完整保存，可用verify_teacher.py --folder指定归档目录独立核验。','',
        '## 182帧最终定位','',
        '| 条件 | 选中epoch | 位置均值cm | 旋转均值° | 位置P95 cm | 固定候选位置cm |','|---|---:|---:|---:|---:|---:|']
    for name,result in [('纯LEADER',baseline)]+list(summary['runs'].items()):
        m=result['standard']; ep=result.get('selection',{}).get('epoch','—')
        lines.append(f'| {name} | {ep} | {m["mean"][0]*100:.4f} | {m["mean"][1]:.4f} | {m["p95"][0]*100:.4f} | {result["fixed"]["mean"][0]*100:.4f} |')
    lines+=['','## 方向覆盖与实际纠正','','| 条件 | 原全空间可达 | 新子空间可达 | 实际纠正 |','|---|---:|---:|---:|']
    for name,r in summary['runs'].items():
        c=r['coverage']; lines.append(f'| {name} | 161 | {c["reachable"]} | {c["actual_correct"]} |')
    lines+=['','上一轮正确视觉子空间/实际纠正为46/3、52/6、46/4；置乱为48/5、50/5、49/4。这里161点仅用于机制衔接，既不生成训练教师，也不用于选模。子空间可达仍是自由GT系数的松弛，不保证共享网络能表达或定位受益。','',
        '## 完整候选排序与损害','','| 条件 | 歧义Top1 | 歧义纠正 | 歧义损害 | 全可见 N→P | 全可见 P→N | 全可见 P→G | 全可见 G→P |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in summary['runs'].items():
        m=r['mechanism']; f=r['full_ranking']; lines.append(f'| {name} | {m["accuracy"]:.2%} | {m["rescue"]} | {m["damage"]} | {f["NP"]} | {f["PN"]} | {f["PG"]} | {f["GP"]} |')
    lines+=['','完整30160个图像有效查询均计入P/N/G转移；G保留标签不确定，不并入明确错误。历史可比的6937歧义点另报原评价口径，原Top1为13.9542%。评价使用同一候选和固定排序；位姿Matcher配对随机状态与基线一致。','',
        '## 轨迹块配对比较','','负值表示正确视觉位置误差更低。按日期内连续最多10帧、超过10秒断块，10000次重采样，种子271828；先计算三个种子的逐帧均值。','',
        '```json',json.dumps(summary['paired_translation'],indent=2),'```','',
        'summary.json保留逐日期位置、旋转、P95、成功率、保护点保留率和固定候选结果。历史同结构对比损失六组结果保留在上级results，不作为本轮选模或调参依据。','',
        '## 结果解读','',
        '本轮未建立优于纯LEADER的稳定平均位置收益，继续保留纯LEADER。正确视觉三种子平均位置12.2885cm，相对纯LEADER的12.2615cm高0.270mm；轨迹块95%区间为[-1.548,+2.147]mm，包含零。一个种子退化、两个略有改善，且1月22日三个种子的位置均值都高于基线。','',
        '同时不能把所有对照都描述为没有差别：正确视觉在三个种子的平均位置上都优于对应置乱，种子均值差为-1.990mm，轨迹块95%区间[-3.462,-0.543]mm。这个结果支持当前开发集上正确与置乱条件存在位置差异，但跨日期并非处处同向，也没有转成超过纯LEADER的稳定平均位置收益；不能概括为图像完全未被使用。','',
        '方向覆盖从旧正确模型46/52/46增至53/55/55，但原161点上的实际纠正从3/6/4降为2/1/3；置乱为3/3/2。明确方向监督没有建立预期的实际纠正增益。完整6937歧义点的正确视觉Top1仍只有14.01%–14.10%，相对13.9542%的基线变化很小，且未在全部配对种子中优于置乱。不能只用子空间覆盖略增宣布监督有效。','',
        '尾部读数有改善：正确视觉三个位置P95均低于纯LEADER；旋转均值两组更好、一组更差。全部六组182/182通过1m/5度条件，保护点最终保留率均为100%。这些读数保留，但不替代预先固定的平均位置与视觉对照判据。','',
        '训练日志中的方向损失在第一轮约0.211–0.215，最后一轮约0.227–0.242，没有呈现期望的下降；总损失下降并不等于方向教师被拟合好。日志本身不足以区分输入信息、共享表达约束与联合损失冲突，不能据此确认某一个失败原因，也不据此调整lambda或继续延长训练。','',
        '所有1500个保留教师在实际float32特征加法后仍满足正间隔gamma：最小间隔0.00010000537、最大归一化残差0.04999025；零个点跌破gamma。新子空间证书、配对初始化和选模记录均通过独立审计。','',
        '## 适用边界','','182帧是已经参与机制研究的开发集，不是盲测；没有完整NCLT提升结论。只增加方向覆盖不算成功，排序改善但定位不改善也不算成功，正确视觉未稳定优于置乱时不归因于逐点视觉贡献。训练方向标签仅来自578帧；先前开发集161点证书没有用于蒸馏。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    target=e.HERE/'results/certificate_fusion'; target.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.name=='teacher':
            continue
        if path.is_dir():
            dest=target/path.name; dest.mkdir(exist_ok=True)
            for f in path.iterdir():
                if f.name!='resume.pt':
                    shutil.copy2(f,dest/f.name)
        else:
            shutil.copy2(path,target/path.name)
    print('REPORT AND AUDIT COMPLETE',flush=True)


if __name__=='__main__':
    main()
