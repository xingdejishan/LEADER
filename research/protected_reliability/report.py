import json
import shutil
import numpy as np
import torch
import experiment as ex
from model import ProtectedVisualReliability


def main():
    torch.set_num_threads(4)
    data=ex.load_data()
    head=ex.new_head(2089)
    head.load_state_dict(torch.load(ex.OUTPUT/'pilot_last.pt',map_location='cuda'))
    training=[d for d in data if d['row']['split']=='train']
    mechanism={}
    with torch.no_grad():
        for split,items in [('fit48',training[:48]),('internal16',training[48:])]:
            ratios=[]
            entered_errors=[]
            left_errors=[]
            changed=0
            counts=[]
            for item in items:
                out,selected=ex.infer(head,item,'aligned',2089)
                original=item['state']['original']
                entered=selected[~torch.isin(selected,original)]
                left=original[~torch.isin(original,selected)]
                changed+=int(len(entered)>0)
                counts.append(len(entered))
                entered_errors.extend(item['coordinate_error'][entered].cpu().tolist())
                left_errors.extend(item['coordinate_error'][left].cpu().tolist())
                if out.gap>1e-8 and out.editable.any():
                    ratios.extend((out.delta[out.editable]/(.25*out.gap)).cpu().tolist())
            ratio=np.asarray(ratios)
            mechanism[split]=dict(frames=len(items),changed_frames=changed,replacements=sum(counts),
                entered_mean_error=float(np.mean(entered_errors)) if entered_errors else None,
                left_mean_error=float(np.mean(left_errors)) if left_errors else None,
                editable_count=len(ratio),saturated_fraction=float((abs(ratio)>.95).mean()),
                positive_fraction=float((ratio>0).mean()),normalized_delta_quantiles=np.percentile(ratio,[0,25,50,75,100]).tolist())
    ex.run.save_json(ex.OUTPUT/'pilot_mechanism.json',mechanism)
    pilot=json.loads((ex.OUTPUT/'pilot_evaluation.json').read_text())
    stats=json.loads((ex.OUTPUT/'pilot_training.json').read_text())
    result=json.loads((ex.OUTPUT/'result.json').read_text())
    lines=['| 内部选模epoch | 平移均值m | 旋转均值° | 平移P95m | 替换点总数 |', '|---|---:|---:|---:|---:|']
    for row in pilot:
        metric=row['metrics']
        lines.append(f"| {row['epoch']} | {metric['mean'][0]:.6f} | {metric['mean'][1]:.6f} | {metric['p95'][0]:.6f} | {sum(r['replaced'] for r in row['records'])} |")
    selected=result['selected_epoch']
    status='未通过本轮预设验证' if not result['passed'] else '通过本地开发验证，需冻结后扩大验证'
    text='\n\n'.join([
        '# 坐标锁定、核心候选保护的视觉可靠度残差头',
        f'结论：{status}。内部48帧拟合、16帧选模跑满100 epoch，最终选中E={selected}；epoch0是合法候选，未因训练过的模型更复杂而强行采用它。',
        '## 实现与来源',
        '用户提供的protected_visual_reliability.py原样保存为prototype.py，直接导入其ProtectedVisualReliability和boundary_ranking_loss。未改评分上下文归一化、核心并列分数处理、缺图/小帧回退或排序损失。新增仅为数据加载、原TRR/Matcher连接、固定候选排序、训练选模和评估。',
        'head为641→32→1，20577参数，最后一层零初始化，delta限制在±0.25g；所有坐标和原编码器/回归头、DeDoDe/PCA冻结。原型在核心阈值处保护所有同分点；gap<=1e-8时直接回退。真实数据的原始预测与旧缓存最大差为0。',
        '## 训练和选模',
        '按原64帧训练清单时间顺序取前48拟合、后16内部选模，原32帧开发清单不参与选择E。每批4帧，AdamW，weight_decay=1e-4，100 epoch；前5 epoch预热到1e-3，随后余弦下降到1e-5。损失为逐帧原TRR+0.1×原型排序损失，再跨帧平均。每10 epoch评估内部16帧，按最终平均平移选E，精确并列取更早epoch。',
        f"实际完成{stats['optimizer_updates']}次优化更新，首个非零输出层梯度范数{stats['first_gradient_norm']:.8f}，训练与内部选模耗时{stats['seconds']:.2f}秒；这不是仅跑合成输入自检，也不是训练未开始。",
        '\n'.join(lines),
        '## 候选替换机制',
        f"epoch100：48帧拟合集有{mechanism['fit48']['changed_frames']}帧发生替换，共{mechanism['fit48']['replacements']}点；内部16帧有{mechanism['internal16']['changed_frames']}帧发生替换，共{mechanism['internal16']['replacements']}点。",
        f"内部换入点原坐标误差均值{mechanism['internal16']['entered_mean_error']}m，换出点{mechanism['internal16']['left_mean_error']}m；可调整点中归一化残差绝对值>0.95的比例{mechanism['internal16']['saturated_fraction']*100:.2f}%。这些是解释当前头行为的诊断，不是单独的最终定位成功判据。",
        '内部epoch10到100最终指标相同，应结合候选集合是否相同理解：评分连续变化不一定跨过top-k边界。当前实验缓存完全相同候选有序列表的Matcher结果，避免重复计算；未用分数作为Matcher连续权重。',
        '## 输入排序与保护校验',
        '新分数只决定候选集合，送入Matcher时始终按原分数的固定顺序排列：先保留原torch.topk返回次序，再按原分数排列未入选点。这样原候选集合不变时，输入次序也严格不变，且原始baseline与历史32帧结果一致。所有条件每帧Matcher使用随机种子2089。',
        '原型合成自检已通过；真实评估逐帧断言XYZ逐元素一致、核心分数不变、缺图分数不变、核心候选保留、未换点时候选顺序一致。内部原候选重复求解结果一致。',
        '## 最终重训规则及32帧结果',
        '固定种子2089/2090/2091，正确图像与逐帧有效描述子置换对照共享初始化、E和优化规则。置换不跨帧，不改变mask或描述子集合。',
        ('因为选中E=0，六个最终头都保持零初始化，最终64帧重训的优化更新数为0。这不是“六个训练后模型效果一样”，也不能用它分析训练后的视觉增益；它表示内部准入没有通过，按协议回退原模型。32帧的六个结果均为原始LEADER：均值0.129776m/1.148154°，平移P95 0.233199m，32/32成功，无候选替换，rescue=damage=0。' if selected==0 else '详细三种子结果见result.json。'),
        '按4个连续8帧块做2000次配对重采样。E=0时两种差值和区间自然都是0，不能解释成独立统计证据。本轮不挑选非零epoch用于开发集展示，也不根据开发结果改变alpha、保护比例或损失系数。',
        '## 适用范围与下一步',
        '这个固定原型尚未提供正向证据。保护坐标和核心候选的结构约束有效，但不保证剩余点的替换有益；当前局部数据不足以宣称该方向普遍无效。按照预设停止条件保留失败结果，暂不叠加坐标修正或其他模块。所有数据已接触且同一天参与LEADER预训练，不是盲测或完整NCLT结论。',
        '复现：egonn118环境运行experiment.py prepare、pilot、refit、evaluate，再运行report.py；proposal.md是研究方案，prototype.py是用户附件，protocol.json记录训练前固定的完整规则。',
    ])
    (ex.OUTPUT/'REPORT.md').write_text(text)
    destination=ex.HERE/'results'
    destination.mkdir(parents=True,exist_ok=True)
    for path in ex.OUTPUT.iterdir():
        if path.is_file():shutil.copy2(path,destination/path.name)
    print(json.dumps(mechanism,indent=2))


if __name__=='__main__':main()
