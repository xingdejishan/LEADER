import json
import shutil
from pathlib import Path
import run


def read(path):
    return json.loads(path.read_text())


def table(rows):
    lines = ['| 方法 | 平移均值m | 旋转均值° | 平移中位m | 旋转中位° | 平移P95m | 旋转P95° | 成功1m/5° |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, values in rows:
        numbers = values['mean'] + values['median'] + values['p95']
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in numbers)+f" | {values['successes']}/{values['count']} |")
    return '\n'.join(lines)


def main():
    root2 = Path('/home/zhang/leader-image-gate-lpgrf')
    root4 = Path('/home/zhang/leader-image-gate-geometry')
    first = read(Path('/home/zhang/leader-image-gate-raw/result.json'))
    third = read(Path('/home/zhang/leader-image-gate-context/result.json'))
    second, fourth = read(root2/'result.json'), read(root4/'result.json')
    diagnostic = read(root4/'geometry_diagnostic.json')
    training = read(root2/'aligned_training_stats.json')
    inference = read(root2/'aligned_inference_stats.json')
    pca = read(root2/'pca.json')
    shared = table([('原始LEADER', first['metrics']['baseline']), ('① 简单门控', first['metrics']['aligned']),
        ('② LP-GRF两阶段', second['metrics']['aligned']), ('③ 全局/坐标上下文', third['metrics']['aligned']), ('④ 三维监督上下文', fourth['metrics']['aligned'])])
    common = '所有定位指标来自同一2012-02-18本地32帧开发集，训练使用64帧。该日期也参与过LEADER预训练；数据已接触，不能视作盲测、跨日期泛化或完整NCLT结论。所有方法32/32成功，rescue=damage=0。复用完全一致的LiDAR编码缓存、定位坐标、原始代表点投影与有效mask；本次没有修改投影阈值。先前发现的重复编码特征差异仍未解决，但本次各方法共用同一缓存，没有混用重编码特征。'
    report2 = '\n\n'.join([
        '# ② LP-GRF 文档版实现与本地实验',
        '实现和评估已完成，本次固定配置未通过预设开发验证。平均位置误差0.149513m，高于原始LEADER的0.129776m；不能宣布多模态方法有效。', shared, common,
        '## 与设计文档的对应',
        '实现LoFTR outdoor的layer3原始256D、1/8分辨率特征，PCA降至128D；灰度632×480图像。PCA只使用64帧训练图像的32768个有效特征，不读取验证特征拟合，保留方差'+f"{pca['retained_variance']*100:.2f}%"+'。LoFTR局部前向与本地Kornia参考实现逐元素一致。',
        '融合按文档实现：LN512/LN128→各64D门控投影；拼接64+64+range/100+valid，经130→64→1 sigmoid；图像128→512无偏置线性残差，alpha初始0.1；最终相加后不做LN。推理融合参数116354，两个训练蒸馏投影器参数41088。',
        '训练损失为原始LEADER TRR + 0.05×余弦蒸馏，蒸馏mask取原始冻结LEADER可靠度top50%与图像有效mask交集，LiDAR特征stop-gradient。10%整帧模态dropout。第一阶段5个完整epoch（320步），仅训练门控和两个投影器；第二阶段600步，门控/投影器1e-3、原MMRegressor 1e-4、LoFTR layer3 1e-5。RPGE全程冻结。',
        '冻结的LoFTR前两层输出以float16缓存；训练时在线计算可微layer3、固定PCA和双线性采样，因此蒸馏确实进入图像网络，并非只训练两个独立投影器。单帧小批量下固定BatchNorm运行统计；layer3卷积及BN仿射参数参与训练。',
        '沿用已修复的raw Cartesian代表点和4px raw-scan深度筛选，替代文档中已被否定的coarse voxel中心投影和8px输出voxel zbuffer。主干、TRR、可靠度筛选和原Matcher求姿态结构保留。',
        '## 消融',
        table([(name, second['metrics'][key]) for name,key in [('完整②','aligned'), ('去蒸馏','no_distill'), ('打乱训练与输入对应','shuffled'), ('只微调回归头','lidar_only'), ('完整②、打乱输入','aligned_wrong'), ('完整②、缺图','aligned_missing')]]),
        '②比只微调回归头的0.165394m降低约9.60%，且比同模型打乱输入的0.155294m更低，说明本配置有局部视觉作用的迹象；但同模型缺图仍为0.151433m，加入图像仅小幅降低平移，旋转反而更差。去蒸馏的平移均值略好、旋转和尾部更差，因此蒸馏也不是所有指标一致改善。',
        '## 缺图行为与检查',
        '缺图时融合特征严格等于原LiDAR特征，即便图像含NaN也可回退。但是第二阶段更新了MMRegressor，缺图姿态来自微调后的回归头，不能声称与原始LEADER checkpoint完全相同。实际缺图结果已单列；不会偷偷换回原回归头来美化结果。',
        f"已验证图像layer3收到TRR梯度范数{training['image_gradients']['trr_image']:.6f}、蒸馏梯度范数{training['image_gradients']['distillation_image']:.6f}；蒸馏不向LiDAR输入反传。原始baseline的32帧姿态误差与①缓存逐帧一致。",
        f"第二阶段可训练：融合116354、投影器41088、回归头{training['decoder_parameters']}、图像layer3 {training['image_trainable_parameters']}参数。aligned训练耗时{training['seconds']:.2f}s，峰值显存{training['peak_memory_mb']:.1f}MiB。缓存stem读取→image layer3/PCA/采样→融合→decoder平均{inference['mean_seconds']*1000:.3f}ms；不含LiDAR编码、图像前两层、Matcher，不是完整端到端延迟。",
        '通过条件在训练前固定：相对原始LEADER和decoder-only平均平移降低至少5%，平均旋转和P95平移不恶化超过5%，零新增失败，优于shuffled。结果未通过；保留失败结果，不用验证集调参。',
        '## 复现',
        'lpgrf_experiment.py依次执行prepare、train、evaluate（egonn118环境）；check_lpgrf.py在rscore-l环境校验参考LoFTR。prepare会在缺少权重时下载公开LoFTR outdoor checkpoint。所有原始依赖保留在本机WSL，protocol记录权重哈希与数据口径。保存阶段1和最终权重、PCA、逐帧评估、梯度检查和资源统计。',
    ])
    report4 = '\n\n'.join([
        '# ④ 三维监督场景上下文 + LEADER门控',
        '实现和评估已完成，本次固定配置未通过开发验证：平均位置误差0.134332m，未优于③的0.132186m或原始LEADER。', shared, common,
        '## 实现',
        '与③保持相同pose Node2Vec、NetVLAD top1检索、DeDoDe/PCA128、coarse/refinement结构和184141参数的适配器/门控。仅将原有重投影监督scrfacto checkpoint换为本地geometry checkpoint；两者均完成10000步场景预训练，geometry模型对coarse和final输出增加持续LiDAR三维几何监督。没有换成lidar Node2Vec，以避免同时改变图编码。',
        '复用已训练的三维监督场景模型，不是从头重复预训练。PersistentGeometryLoss在有几何标签处用沿相机射线和平行/垂直方向的鲁棒几何损失，权重从1降至0.25，coarse权重0.5、final权重1；未知处保留重投影回退。具体实现为现有rscore_l/losses.py。当前融合训练不再次更新场景网络，仅训练适配器和门控600步、Adam1e-4、seed2089，和③一致。',
        '## 最终定位消融',
        table([(name, fourth['metrics'][key]) for name,key in [('④ 正确对应','aligned'), ('④ 打乱训练','shuffled'), ('④ 正常模型、打乱输入','aligned_wrong'), ('④ 缺图','aligned_missing')]]),
        '## 三维监督是否实际起作用',
        f"在同一32帧验证集的代表点上，final预测深度相对误差（先逐帧取中位数，再跨帧平均）由③的{diagnostic['val']['line3']['mean_frame_median_relative_depth'][1]:.4f}降至④的{diagnostic['val']['line4']['mean_frame_median_relative_depth'][1]:.4f}；对应世界坐标误差由{diagnostic['val']['line3']['mean_frame_median_world_error'][1]:.2f}m降至{diagnostic['val']['line4']['mean_frame_median_world_error'][1]:.2f}m。",
        '这表明三维监督明显减轻了视觉场景坐标/深度错误，但残差仍较大，不能说完全消除了深度膨胀；更不能把这一中间指标当成LEADER最终定位提升。诊断才使用GT和raw代表点世界坐标，检索与推理不读取查询GT。两种方法的投影mask完全相同，深度竞争的保留率没有因该监督改变。',
        '已验证缺图严格回退至原始LEADER预测、适配器梯度非零、原始baseline逐帧一致。当前固定协议未通过；该结论只适用于已冻结的场景模型、top1上下文、当前适配器和小样本预算。',
        '## 复现',
        'context_experiment.py prepare --scene-variant geometry --output /home/zhang/leader-image-gate-geometry，在rscore-l环境执行；同一output执行train、evaluate、report，在egonn118环境执行。geometry_diagnostic.py输出中间几何诊断；report_two_four.py生成本报告。输入模型与embedding哈希、训练记录、gate权重及逐帧结果全部保留。',
    ])
    for root, name, text in [(root2, 'lpgrf_retrain', report2), (root4, 'geometry_retrain', report4)]:
        (root/'REPORT.md').write_text(text)
        destination = run.HERE/'results'/name
        destination.mkdir(parents=True, exist_ok=True)
        for path in root.iterdir():
            if path.is_file():
                shutil.copy2(path, destination/path.name)
        artifacts = {path.name:run.digest(path) for path in destination.iterdir() if path.is_file() and path.name!='artifact_hashes.json'}
        run.save_json(destination/'artifact_hashes.json', artifacts)
    (run.HERE/'results/COMPARISON_1_2_3_4.md').write_text('# 本地四条实验线对比\n\n'+shared+'\n\n'+common+'\n\n②和④均未通过各自预设开发验证。②与①/③/④的训练参数和预算不同，不能把差异单独归因于某一个模块。详细消融和实现边界见各自REPORT.md。')
    print(shared)


if __name__ == '__main__':
    main()
