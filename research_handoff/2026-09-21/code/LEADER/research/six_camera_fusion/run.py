import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from model import MultiViewFusion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module('frozen_baseline', HERE.parent / 'dinov2_fusion/run.py')
geometry = load_module('projection', WORKSPACE / 'work/leader-image-gate/research/image_gate/fusion.py')
sys.path.insert(0, str(WORKSPACE / 'glace-local/code/tools'))
from full_pool_robust_v1 import full_pool_refine

SOURCE = Path('/home/zhang/leader-image-gate-multicamera')
PCA = Path('/home/zhang/rscore-l-local/data/proc/pcad3LB_128.pth')
SEEDS = [2089, 2090, 2091]
ARMS = ['lidar', 'cam5', 'mean', 'query']


def prepare(args):
    rows = json.loads((SOURCE / 'all_views.json').read_text())
    previous = json.loads((base.SOURCE / 'manifest.json').read_text())
    assert [(r['frame_id'], r['split']) for r in rows] == [(r['frame_id'], r['split']) for r in previous]
    assert len(rows) == 96 and sum(r['split'] == 'train' for r in rows) == 64
    hashes = {}
    for row in rows:
        stem = row['frame_id']
        assert sorted(v['camera'] for v in row['views']) == list(range(6))
        hashes[stem] = dict(lidar=base.sha(base.SOURCE / 'lidar' / (stem+'.npz')),
            mapping=base.sha(base.SOURCE / 'projection_audit/mapping' / (stem+'.npz')),
            aggregate=base.sha(SOURCE / 'all_features' / (stem+'.npz')),
            views={v['camera']: dict(image=base.sha(Path(v['image'])), calibration=base.sha(Path(v['calibration'])),
                mask=base.sha(Path(v['mask']))) for v in row['views']})
    protocol = dict(seeds=SEEDS, arms=ARMS, steps_per_arm_seed=600, points_per_step=1024,
        learning_rate=.0001, optimizer='Adam', modality_dropout=.1, train_frames=64, val_frames=32,
        frozen=['DeDoDe-B', 'same PCA256->128', 'same cached LEADER encoder', 'decoder except pred_out'],
        trained='decoder.pred_out for all arms; same residual gate for all visual arms; extra query/key/direction parameters for query aggregation only',
        parameter_control='Cam5 vs six-view mean has identical effective trainable architecture. Mean vs query changes aggregation and its parameter capacity; not a parameter-matched attention-only attribution.',
        sampling='Each seed has one shared frame schedule; independent seed+step point permutation and dropout draw; no visibility-based point filtering',
        solver='Same SC2 top50% and full_pool_refine(1.2m,0.6m), seed2089+frame index for every arm and training seed',
        spatial_input='Existing local LiDAR representatives and camera-to-body extrinsics only; no GT or world-pose input',
        inference_diagnostics='Missing image and shuffled feature-to-voxel association for query; no new trained models',
        checkpoint_selection='Final step600 only; no validation tuning',
        success='1m/2deg and 1m/5deg; report mean across training seeds, not best seed',
        pass_rule='Visual mean MPE and MOE must both beat matched lidar fine-tuning; report each seed and failures; no claimed gain from coverage alone',
        scope='Existing64/32 previously touched development split; not905/301/148, not independent test',
        projection_source_sha256=base.sha(WORKSPACE / 'work/leader-image-gate/research/image_gate/fusion.py'),
        pca_sha256=base.sha(PCA), descriptor_sha256=base.sha(WORKSPACE / 'rscore-assets/dedode_descriptor_B.pth'),
        leader_sha256=base.sha(WORKSPACE / 'research/image_gate_checkpoint/model.safetensors'),
        refinement_sha256=base.sha(WORKSPACE / 'glace-local/code/tools/full_pool_robust_v1.py'),
        matcher_sha256=base.sha(REPO / 'models/sc2pcr.py'), inputs=hashes)
    protocol = json.loads(json.dumps(protocol))
    path = args.root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Frozen protocol changed; use another output directory')
    base.save(path, protocol)
    base.save(args.root / 'manifest.json', rows)


def extract(args, rows):
    from kornia.feature.dedode.dedode_models import get_descriptor
    descriptor = get_descriptor('B').cuda().eval().requires_grad_(False)
    descriptor.load_state_dict(torch.load(WORKSPACE / 'rscore-assets/dedode_descriptor_B.pth', map_location='cpu', weights_only=True))
    pca = torch.load(PCA, map_location='cuda', weights_only=True)
    destination = args.root / 'features'
    destination.mkdir(exist_ok=True)
    checks = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            stem = row['frame_id']
            target = destination / (stem+'.npz')
            if target.exists():
                with np.load(target) as cached:
                    checks.append(json.loads(str(cached['check'])))
                continue
            raw = np.fromfile(row['scan'], dtype=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')]))
            raw = np.column_stack([raw[k] for k in ['x','y','z']]).astype(np.float32)*.005-100
            distance = np.linalg.norm(raw, axis=-1)
            raw = torch.tensor(raw[(distance>1)&(distance<100)], device='cuda')
            with np.load(base.SOURCE / 'projection_audit/mapping' / (stem+'.npz')) as mapping:
                points = torch.tensor(mapping['projection_xyz'], device='cuda')
                supported = torch.tensor(mapping['projection_supported'], device='cuda')
                with np.load(base.SOURCE / 'lidar' / (stem+'.npz')) as lidar:
                    assert np.array_equal(mapping['localization_xyz'], lidar['source'])
            features, masks, directions = [], [], []
            for view in sorted(row['views'], key=lambda v:v['camera']):
                image = Image.open(view['image']).convert('RGB')
                w,h = image.size
                nh,nw = int(np.ceil(h*480/min(h,w)/8))*8, int(np.ceil(w*480/min(h,w)/8))*8
                k = torch.tensor(np.loadtxt(view['calibration']), device='cuda', dtype=torch.float32)
                k[0] *= nw/w
                k[1] *= nh/h
                extrinsic = torch.tensor(view['camera_to_body'], device='cuda', dtype=torch.float32)
                mask = torch.tensor(np.load(view['mask']), device='cuda', dtype=torch.float32)
                assert tuple(mask.shape) == (h,w)
                mask = F.interpolate(mask[None,None], size=(nh,nw), mode='nearest')[0,0]
                if 'sampled_cache' in view:
                    with np.load(view['sampled_cache']) as cache:
                        values = torch.tensor(cache['image'], device='cuda').float()
                        valid = torch.tensor(cache['valid'], device='cuda')
                    _, checked = geometry.sample_visible(torch.zeros(1,1,1,1,device='cuda'), points, extrinsic,k,mask,raw)
                    assert torch.equal(valid, checked&supported)
                else:
                    pixels = torch.tensor(np.array(image.resize((nw,nh), Image.Resampling.BILINEAR)), device='cuda').permute(2,0,1)[None].float()/255
                    mean = pixels.new_tensor([.485,.456,.406])[None,:,None,None]
                    std = pixels.new_tensor([.229,.224,.225])[None,:,None,None]
                    with torch.autocast('cuda'):
                        dense = descriptor((pixels-mean)/std)
                    dense = F.conv2d(dense.float(),pca['weight'].float(),pca['bias'].float())
                    values,valid = geometry.sample_visible(dense,points,extrinsic,k,mask,raw)
                    valid &= supported
                features.append(torch.where(valid[:,None], values, torch.zeros_like(values)))
                masks.append(valid)
                directions.append(F.normalize((points-extrinsic[:3,3])@extrinsic[:3,:3],dim=-1))
            values = torch.stack(features,1)
            valid = torch.stack(masks,1)
            mean = values.sum(1)/valid.sum(1).clamp_min(1)[:,None]
            with np.load(SOURCE / 'all_features' / (stem+'.npz')) as old:
                order = [list(old['camera_ids']).index(i) for i in range(6)]
                assert np.array_equal(valid.cpu().numpy(), old['per_view_valid'][order].T)
                assert np.array_equal(valid.any(1).cpu().numpy(), old['valid'])
                delta = float(np.max(np.abs(mean.cpu().numpy()-old['image'])))
                if delta > .002:
                    raise ValueError(f'Historical six-view mean differs: {stem}: {delta}')
            check = dict(frame_id=stem, total=len(points), cam5=int(valid[:,5].sum()), union=int(valid.any(1).sum()),
                         overlap=int((valid.sum(1)>1).sum()), old_mean_max_difference=delta)
            np.savez(target,image=values.cpu().numpy(),mask=valid.cpu().numpy(),
                     direction=torch.stack(directions,1).cpu().numpy(),check=json.dumps(check))
            checks.append(check)
            print(f'views {index+1}/96 union={check["union"]}/{len(points)} parity={delta:.6g}',flush=True)
    totals = {k:sum(r[k] for r in checks) for k in ['total','cam5','union','overlap']}
    if totals != dict(total=67710,cam5=14728,union=45133,overlap=10125):
        raise ValueError(f'Coverage differs from user report: {totals}')
    base.save(args.root/'extraction.json',dict(totals=totals,records=checks))


def frame(args,row):
    with np.load(base.SOURCE/'lidar'/(row['frame_id']+'.npz')) as data:
        item = {k:torch.tensor(data[k],device='cuda',dtype=torch.float32) for k in ['features','source','prediction','target','GT','center']}
    with np.load(args.root/'features'/(row['frame_id']+'.npz')) as data:
        item.update(image=torch.tensor(data['image'],device='cuda'),mask=torch.tensor(data['mask'],device='cuda'),
                    direction=torch.tensor(data['direction'],device='cuda'))
    return item


def train(args,rows):
    training = [r for r in rows if r['split']=='train']
    loss_fn = base.source_class(REPO/'run_mink.py','TRR')(scale=10.)
    for seed in SEEDS:
        schedule = np.random.default_rng(seed).integers(len(training),size=600)
        for arm in ARMS:
            path = args.root/f'{arm}_{seed}.pt'
            if path.exists():
                continue
            torch.manual_seed(seed)
            fusion = MultiViewFusion(arm).cuda()
            decoder = base.decoder()
            decoder.pred_out.requires_grad_(True)
            parameters = list(decoder.pred_out.parameters())
            if arm!='lidar':
                parameters += list(fusion.fusion.parameters())
                if arm=='query':
                    parameters += [p for name,p in fusion.named_parameters() if not name.startswith('fusion.')]
            optimizer = torch.optim.Adam(parameters,lr=.0001)
            logs=[]
            started=time.perf_counter()
            for step,index in enumerate(schedule):
                item=frame(args,training[index])
                rng=np.random.default_rng(seed+step)
                indices=rng.permutation(len(item['features']))[:1024]
                lidar,image,mask,direction,target=[item[k][indices] for k in ['features','image','mask','direction','target']]
                if rng.random()<.1:
                    mask=torch.zeros_like(mask)
                optimizer.zero_grad(set_to_none=True)
                prediction=decoder(lidar if arm=='lidar' else fusion(lidar,image,mask,direction))
                loss=loss_fn(target,prediction[:,:3],prediction[:,3],torch.zeros(len(prediction),device='cuda',dtype=torch.long))[0].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{arm} seed{seed} step{step}')
                loss.backward()
                optimizer.step()
                if step%100==0 or step==599:
                    logs.append(dict(step=step+1,loss=float(loss.detach()),seconds=time.perf_counter()-started))
                    base.save(args.root/f'{arm}_{seed}_training.json',logs)
                    print(f'{arm} seed{seed} {step+1}/600 loss={loss.item():.5f}',flush=True)
            torch.save(dict(decoder=decoder.cpu().state_dict(),fusion=fusion.cpu().state_dict(),
                            optimized_parameters=sum(p.numel() for p in parameters)),path)


def pose_error(pose,gt):
    cosine=((pose[:3,:3].T@gt[:3,:3]).trace()-1)/2
    return [float((pose[:3,3]-gt[:3,3]).norm()),float(torch.rad2deg(cosine.clamp(-1,1).acos()))]


def evaluate(args,rows):
    from models.sc2pcr import Matcher
    matcher=Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
    records=[]
    for seed in SEEDS:
        models={}
        for arm in ARMS:
            state=torch.load(args.root/f'{arm}_{seed}.pt',map_location='cuda')
            decoder=base.decoder()
            decoder.load_state_dict(state['decoder'])
            fusion=MultiViewFusion(arm).cuda().eval()
            fusion.load_state_dict(state['fusion'])
            models[arm]=(decoder,fusion)
        for index,row in enumerate(r for r in rows if r['split']=='val'):
            item=frame(args,row)
            with torch.inference_mode():
                predictions=dict(pretrained=item['prediction'])
                for arm,(decoder,fusion) in models.items():
                    predictions[arm]=decoder(item['features'] if arm=='lidar' else fusion(item['features'],item['image'],item['mask'],item['direction']))
                    if arm=='query':
                        predictions['query_missing']=decoder(fusion(item['features'],item['image'],torch.zeros_like(item['mask']),item['direction']))
                        if not torch.equal(predictions['query_missing'],decoder(item['features'])):
                            raise ValueError('Missing-view fallback is not exact')
                        shuffled=item['image'].clone()
                        for camera in range(6):
                            valid=torch.where(item['mask'][:,camera])[0]
                            shuffled[valid,camera]=shuffled[valid.roll(1),camera]
                        predictions['query_shuffled']=decoder(fusion(item['features'],shuffled,item['mask'],item['direction']))
                errors={}
                for arm,pred in predictions.items():
                    torch.manual_seed(2089+index)
                    keep=pred[:,3].topk(max(min(50,len(pred)),int(.5*len(pred)))).indices
                    initial=matcher.estimator(item['source'][keep][None],pred[keep,:3][None])[0]
                    pose,_=full_pool_refine(initial,item['source'],pred[:,:3])
                    pose[:3,3]+=item['center']
                    errors[arm]=pose_error(pose,item['GT'])
            records.append(dict(seed=seed,frame_id=row['frame_id'],errors=errors))
            base.save(args.root/'records.json',records)
            if index%8==0 or index==31:
                print(f'validation seed{seed} {index+1}/32 mean={errors["mean"]} query={errors["query"]}',flush=True)
    previous=json.loads((HERE.parent/'query_fusion/results/records.json').read_text())
    baseline={r['frame_id']:r['errors']['baseline'] for r in previous}
    parity=max(max(abs(a-b) for a,b in zip(r['errors']['pretrained'],baseline[r['frame_id']])) for r in records)
    if parity>1e-5:
        raise ValueError(f'Historical same two-stage baseline differs: {parity}')
    summary={}
    for arm in records[0]['errors']:
        per_seed={str(s):base.metrics([r['errors'][arm] for r in records if r['seed']==s]) for s in SEEDS}
        values=np.array([[m['MPE'],m['MOE']] for m in per_seed.values()])
        summary[arm]=dict(per_seed=per_seed,mean=values.mean(0).tolist(),std=values.std(0,ddof=1).tolist())
    passed={arm:bool(np.all(np.array(summary[arm]['mean'])<np.array(summary['lidar']['mean']))) for arm in ['cam5','mean','query']}
    base.save(args.root/'result.json',dict(summary=summary,passed=passed,historical_baseline_max_difference=parity))


def report(args):
    import shutil
    result=json.loads((args.root/'result.json').read_text())
    parameter_counts={arm:int(torch.load(args.root/f'{arm}_{SEEDS[0]}.pt',map_location='cpu')['optimized_parameters']) for arm in ARMS}
    base.save(args.root/'parameter_counts.json',parameter_counts)
    records=json.loads((args.root/'records.json').read_text())
    paired={}
    for a,b in [('cam5','lidar'),('mean','cam5'),('query','mean'),('query','lidar'),('query','query_missing'),('query','query_shuffled')]:
        per_seed={str(s):np.mean([np.array(r['errors'][a])-np.array(r['errors'][b]) for r in records if r['seed']==s],axis=0).tolist() for s in SEEDS}
        values=np.array(list(per_seed.values()))
        paired[a+'_minus_'+b]=dict(per_seed_delta_MPE_MOE=per_seed,mean_delta=values.mean(0).tolist(),
                                   seeds_improving_both=int((values<0).all(1).sum()))
    base.save(args.root/'paired.json',paired)
    labels=dict(pretrained='原始 LEADER',lidar='同预算纯 LiDAR 微调',cam5='单 Cam5',mean='六相机等权平均',
                query='六相机 query 聚合',query_missing='query 模型：缺图像',query_shuffled='query 模型：打乱对应')
    lines=['# 六相机融合控制实验','',
        '同一64帧训练/32帧开发验证、相同voxel、DeDoDe/PCA、600步、三个配对种子2089/2090/2091、相同two-stage后端。',
        '以下为三个训练种子的均值±样本标准差，不挑选最好种子；32帧没有变成96个独立测试样本。','',
        '| 方法 | MPE m | MOE ° | 每个种子成功数 <1m/2° |','|---|---:|---:|---|']
    for arm,s in result['summary'].items():
        successes='/'.join(str(s['per_seed'][str(seed)]['success_1m_2deg']) for seed in SEEDS)
        lines.append(f'| {labels[arm]} | {s["mean"][0]:.6f} ± {s["std"][0]:.6f} | {s["mean"][1]:.6f} ± {s["std"][1]:.6f} | {successes}（各32帧） |')
    lines += ['',f'预设条件（MPE、MOE均值均优于同预算纯LiDAR微调）的通过状态：{result["passed"]}',
        f'实际加入优化器的参数量：{parameter_counts}',
        f'历史two-stage基线最大误差差异：{result["historical_baseline_max_difference"]}', '',
        'Cam5与六相机平均保持相同有效可训练架构，主要变化是可用视图；平均与query的比较同时包含聚合机制及额外注意力参数，不能单独归因于参数无关的视角选择。',
        'query_missing使用已微调回归头，只保证融合特征回退，不保证回到原始预训练位姿；query_shuffled保留每相机可见mask及特征分布，只打乱其与voxel的对应。',
        '本轮未加入共视损失、Gaussian、ASQB或新增correspondence，以避免同时改变多个因素。',
        '96帧是64训练+32开发验证，日期属于LEADER预训练范围且已参与开发；不是905/301/148大规模实验或独立盲测。']
    lines += ['', '## 本轮解读', '',
        '三种视觉配置都满足相对同预算LiDAR微调的均值条件，但这不是统计显著性标准，也不表示优于原始LEADER；原始模型的MPE和MOE均低于所有训练配置。',
        '六相机平均相对Cam5没有降低平均平移误差，平均旋转误差几乎相同；覆盖率提高到66.66%尚未转化成稳定的定位收益。',
        '六相机query相对等权平均略好，但变化很小；相对Cam5，平移稍差、旋转稍好，没有全面占优。',
        '去掉图像时query模型的平均平移误差反而略低，打乱对应的平均平移误差与正常输入接近，因此不能把微小差别解释为有效利用了正确视觉对应。',
        '各随机种子与逐帧配对差值见result.json、paired.json、records.json。预训练基线完全复现、五项实现检查通过、四项覆盖计数精确匹配用户报告。']
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    destination=HERE/'results'
    destination.mkdir(exist_ok=True)
    for name in ['protocol.json','manifest.json','extraction.json','records.json','result.json','parameter_counts.json','paired.json','report.md']:
        shutil.copy2(args.root/name,destination/name)
    for path in args.root.glob('*_training.json'):
        shutil.copy2(path,destination/path.name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['check','prepare','extract','train','evaluate','report'])
    parser.add_argument('--root',type=Path,default=Path('/home/zhang/leader-six-camera-controlled'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage=='check':
        import unittest
        from test_model import Tests
        outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not outcome.wasSuccessful():
            raise RuntimeError('Control checks failed')
    elif args.stage in ['prepare','report']:
        globals()[args.stage](args)
    else:
        globals()[args.stage](args,json.loads((args.root/'manifest.json').read_text()))


if __name__=='__main__':
    main()
