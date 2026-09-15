import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from model import FactorialFusion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('baseline', HERE.parent/'dinov2_fusion/run.py')
geometry = load('geometry', WORKSPACE/'work/leader-image-gate/research/image_gate/fusion.py')
sys.path.insert(0, str(WORKSPACE/'glace-local/code/tools'))
from full_pool_robust_v1 import full_pool_refine

SEEDS = [2089, 2090, 2091]
VISUAL = ['dedode_center', 'dedode_patch', 'dino_center', 'dino_patch']
ARMS = ['lidar', *VISUAL]
SOURCE = Path('/home/zhang/leader-image-gate-multicamera')
DINO = Path('/home/zhang/dinov2-leader')
HW = (448, 616)


def prepare(args):
    rows = json.loads((SOURCE/'all_views.json').read_text())
    old = json.loads((base.SOURCE/'manifest.json').read_text())
    assert [(r['frame_id'],r['split']) for r in rows] == [(r['frame_id'],r['split']) for r in old]
    assert len(rows)==96 and sum(r['split']=='train' for r in rows)==64
    protocol = dict(arms=ARMS,seeds=SEEDS,steps=600,points=1024,lr=.0001,dropout=.1,
        train=64,development=32,image_hw=HW,grid=5,spacing_pixels=14,
        backbone='Frozen official DeDoDe B and DINOv2 ViT-B/14; same RGB resize and ImageNet normalization',
        pca='Separate128D PCA fitted to identical visible center indices in all six TRAIN cameras; equal max64 samples per image, seed811; no validation fitting',
        attention='Identical shared query/key parameters for within-camera and across-camera attention; center arm selects only center token, patch arm all25',
        visibility='Identical calibrated raw-point occlusion center mask; neighborhood bounds and undistortion mask; neighbors are image context, not new 3D correspondences',
        geometry='Original local voxel points and camera extrinsics only; no GT pose input',
        trainable='Same decoder.pred_out; same391685 optimized parameters in each visual arm; final600 checkpoint only',
        solver='SC2 top50% then unmodified full_pool_refine1.2m/0.6m; seed2089+frame index for all arms',
        evaluation='3 paired seeds; MPE/MOE per seed and factorial paired differences; missing and shuffled image for all visual arms',
        interpretation='Previously used64/32 development split. Backbone contrast includes pretrained architecture and its fitted PCA, not semantic knowledge alone; patch increases tokens and computation, not parameter count.',
        inputs={r['frame_id']:dict(lidar=base.sha(base.SOURCE/'lidar'/(r['frame_id']+'.npz')),
            mapping=base.sha(base.SOURCE/'projection_audit/mapping'/(r['frame_id']+'.npz')),
            views=[dict(camera=v['camera'],image=base.sha(Path(v['image'])),calibration=base.sha(Path(v['calibration'])),mask=base.sha(Path(v['mask']))) for v in r['views']]) for r in rows},
        weights=dict(dedode=base.sha(WORKSPACE/'rscore-assets/dedode_descriptor_B.pth'),dino=base.sha(DINO/'weights/dinov2_vitb14_pretrain.pth'),leader=base.sha(WORKSPACE/'research/image_gate_checkpoint/model.safetensors')),
        code={str(p.relative_to(WORKSPACE)):base.sha(p) for p in [Path(__file__),HERE/'model.py',REPO/'models/sc2pcr.py',WORKSPACE/'glace-local/code/tools/full_pool_robust_v1.py',WORKSPACE/'work/leader-image-gate/research/image_gate/fusion.py']})
    protocol=json.loads(json.dumps(protocol))
    path=args.root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Frozen protocol changed')
    base.save(path,protocol)
    base.save(args.root/'manifest.json',rows)


def extract(args,rows):
    from kornia.feature.dedode.dedode_models import get_descriptor
    dedode=get_descriptor('B').cuda().eval().requires_grad_(False)
    dedode.load_state_dict(torch.load(WORKSPACE/'rscore-assets/dedode_descriptor_B.pth',map_location='cpu',weights_only=True))
    dino=torch.hub.load(str(DINO/'dinov2-main'),'dinov2_vitb14',source='local',pretrained=False)
    dino.load_state_dict(torch.load(DINO/'weights/dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True))
    dino=dino.cuda().eval().requires_grad_(False)
    h,w=HW
    y,x=torch.meshgrid(torch.arange(-2,3,device='cuda'),torch.arange(-2,3,device='cuda'),indexing='ij')
    offsets=torch.stack([x,y],-1).reshape(25,2)*14
    cache=args.root/'raw'
    cache.mkdir(exist_ok=True)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            target=cache/(row['frame_id']+'.npz')
            if target.exists():
                continue
            raw=np.fromfile(row['scan'],dtype=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')]))
            raw=np.column_stack([raw[k] for k in ['x','y','z']]).astype(np.float32)*.005-100
            distance=np.linalg.norm(raw,axis=-1)
            raw=torch.tensor(raw[(distance>1)&(distance<100)],device='cuda')
            with np.load(base.SOURCE/'projection_audit/mapping'/(row['frame_id']+'.npz')) as mapping:
                points=torch.tensor(mapping['projection_xyz'],device='cuda')
                supported=torch.tensor(mapping['projection_supported'],device='cuda')
                with np.load(base.SOURCE/'lidar'/(row['frame_id']+'.npz')) as lidar:
                    assert np.array_equal(mapping['localization_xyz'],lidar['source'])
            values={b:[] for b in ['dedode','dino']}
            masks,directions=[],[]
            for view in sorted(row['views'],key=lambda v:v['camera']):
                image=Image.open(view['image']).convert('RGB')
                ow,oh=image.size
                k=torch.tensor(np.loadtxt(view['calibration']),device='cuda',dtype=torch.float32)
                k[0]*=w/ow
                k[1]*=h/oh
                extrinsic=torch.tensor(view['camera_to_body'],device='cuda',dtype=torch.float32)
                raster=torch.tensor(np.load(view['mask']),device='cuda',dtype=torch.float32)
                raster=F.interpolate(raster[None,None],size=HW,mode='nearest')[0,0]
                uv,_,_=geometry.project(points,extrinsic,k,HW)
                _,visible=geometry.sample_visible(torch.zeros(1,1,1,1,device='cuda'),points,extrinsic,k,raster,raw)
                visible &= supported
                locations=uv[:,None]+offsets[None]
                flat=locations.reshape(-1,2)
                valid=visible[:,None] & (locations[...,0]>=0)&(locations[...,0]<=w-1)&(locations[...,1]>=0)&(locations[...,1]<=h-1)
                valid &= geometry.sample_map(raster[None,None],flat,HW).reshape(-1,25)>.999
                assert torch.equal(valid[:,12],visible)
                pixels=torch.tensor(np.array(image.resize((w,h),Image.Resampling.BILINEAR)),device='cuda').permute(2,0,1)[None].float()/255
                pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
                with torch.autocast('cuda'):
                    maps=dict(dedode=dedode(pixels),dino=dino.forward_features(pixels)['x_norm_patchtokens'].reshape(1,h//14,w//14,768).permute(0,3,1,2))
                for backbone,dense in maps.items():
                    sampled=geometry.sample_map(dense.float(),flat,HW).reshape(len(points),25,-1)
                    sampled=torch.where(valid[...,None],sampled,0.)
                    assert torch.isfinite(sampled).all()
                    values[backbone].append(sampled.half().cpu().numpy())
                masks.append(valid.cpu().numpy())
                directions.append(F.normalize((points-extrinsic[:3,3])@extrinsic[:3,:3],dim=-1).cpu().numpy())
            temporary=target.with_suffix('.pending.npz')
            np.savez_compressed(temporary,**{b:np.stack(v,1) for b,v in values.items()},mask=np.stack(masks,1),direction=np.stack(directions,1))
            temporary.replace(target)
            print(f'extract {index+1}/96',flush=True)


def compress(args,rows):
    training=[r for r in rows if r['split']=='train']
    fit={b:[] for b in ['dedode','dino']}
    indices=[]
    rng=np.random.default_rng(811)
    for row in training:
        with np.load(args.root/'raw'/(row['frame_id']+'.npz')) as data:
            mask=data['mask'][:,:,12]
            centers={b:data[b][:,:,12].astype(np.float32) for b in fit}
            chosen=[]
            for camera in range(6):
                selected=rng.permutation(np.where(mask[:,camera])[0])[:64]
                chosen.append(selected.tolist())
                for b in fit:
                    fit[b].append(centers[b][selected,camera])
            indices.append(dict(frame_id=row['frame_id'],indices=chosen))
    pcas={}
    for b,parts in fit.items():
        samples=torch.tensor(np.concatenate(parts),dtype=torch.float64)
        mean=samples.mean(0)
        centered=samples-mean
        eigenvalues,eigenvectors=torch.linalg.eigh(centered.T@centered/(len(samples)-1))
        weight=eigenvectors[:,-128:].flip(1).T.float()
        pcas[b]=dict(weight=weight,bias=-weight@mean.float(),samples=len(samples),variance=float(eigenvalues[-128:].sum()/eigenvalues.sum()))
        torch.save(pcas[b],args.root/(b+'_pca.pt'))
        print(f'PCA {b}: {len(samples)} training samples variance={pcas[b]["variance"]:.4f}',flush=True)
    base.save(args.root/'pca_indices.json',indices)
    totals=dict(voxels=0,visible=0,cam5=0)
    for row in rows:
        with np.load(args.root/'raw'/(row['frame_id']+'.npz')) as data:
            mask=data['mask']
            totals['voxels']+=len(mask)
            totals['visible']+=int(mask.any((1,2)).sum())
            totals['cam5']+=int(mask[:,5].any(1).sum())
            for b,pca in pcas.items():
                destination=args.root/b
                destination.mkdir(exist_ok=True)
                target=destination/(row['frame_id']+'.npz')
                if target.exists():
                    continue
                raw=torch.tensor(data[b].astype(np.float32))
                output=(raw@pca['weight'].T+pca['bias']).numpy()
                output[~mask]=0
                temporary=target.with_suffix('.pending.npz')
                np.savez_compressed(temporary,image=output.astype(np.float16),mask=mask,direction=data['direction'])
                temporary.replace(target)
    base.save(args.root/'extraction.json',dict(totals=totals,pca={b:{k:v for k,v in p.items() if k in ['samples','variance']} for b,p in pcas.items()},pca_hashes={b:base.sha(args.root/(b+'_pca.pt')) for b in pcas}))


def frame(args,row,backbone='dedode'):
    with np.load(base.SOURCE/'lidar'/(row['frame_id']+'.npz')) as data:
        item={k:torch.tensor(data[k],device='cuda',dtype=torch.float32) for k in ['features','source','prediction','target','GT','center']}
    with np.load(args.root/backbone/(row['frame_id']+'.npz')) as data:
        item.update(image=torch.tensor(data['image'],device='cuda',dtype=torch.float32),mask=torch.tensor(data['mask'],device='cuda'),direction=torch.tensor(data['direction'],device='cuda'))
    return item


def train(args,rows):
    training=[r for r in rows if r['split']=='train']
    loss_fn=base.source_class(REPO/'run_mink.py','TRR')(scale=10.)
    for seed in SEEDS:
        schedule=np.random.default_rng(seed).integers(len(training),size=600)
        for arm in ARMS:
            path=args.root/f'{arm}_{seed}.pt'
            if path.exists():
                continue
            torch.manual_seed(seed)
            fusion=FactorialFusion(arm).cuda()
            decoder=base.decoder()
            decoder.pred_out.requires_grad_(True)
            parameters=list(decoder.pred_out.parameters())+([] if arm=='lidar' else list(fusion.parameters()))
            optimizer=torch.optim.Adam(parameters,lr=.0001)
            logs=[]
            started=time.perf_counter()
            for step,index in enumerate(schedule):
                item=frame(args,training[index],'dino' if arm.startswith('dino') else 'dedode')
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
                    print(f'{arm} seed{seed} {step+1}/600 loss={loss.item():.5f} elapsed={logs[-1]["seconds"]:.1f}s',flush=True)
            torch.save(dict(decoder=decoder.cpu().state_dict(),fusion=fusion.cpu().state_dict(),optimized_parameters=sum(p.numel() for p in parameters)),path)


def error(pose,gt):
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
            fusion=FactorialFusion(arm).cuda().eval()
            fusion.load_state_dict(state['fusion'])
            models[arm]=(decoder,fusion)
        for index,row in enumerate(r for r in rows if r['split']=='val'):
            items={b:frame(args,row,b) for b in ['dedode','dino']}
            assert torch.equal(items['dedode']['mask'],items['dino']['mask'])
            item=items['dedode']
            with torch.inference_mode():
                predictions=dict(pretrained=item['prediction'])
                for arm,(decoder,fusion) in models.items():
                    item=items['dino' if arm.startswith('dino') else 'dedode']
                    if arm=='lidar':
                        predictions[arm]=decoder(item['features'])
                        continue
                    predictions[arm]=decoder(fusion(item['features'],item['image'],item['mask'],item['direction']))
                    predictions[arm+'_missing']=decoder(fusion(item['features'],item['image'],torch.zeros_like(item['mask']),item['direction']))
                    assert torch.equal(predictions[arm+'_missing'],decoder(item['features']))
                    shuffled=item['image'].clone()
                    for camera in range(6):
                        for patch in range(25):
                            valid=torch.where(item['mask'][:,camera,patch])[0]
                            shuffled[valid,camera,patch]=shuffled[valid.roll(1),camera,patch]
                    predictions[arm+'_shuffled']=decoder(fusion(item['features'],shuffled,item['mask'],item['direction']))
                errors={}
                for arm,pred in predictions.items():
                    torch.manual_seed(2089+index)
                    keep=pred[:,3].topk(max(min(50,len(pred)),int(.5*len(pred)))).indices
                    initial=matcher.estimator(item['source'][keep][None],pred[keep,:3][None])[0]
                    pose,_=full_pool_refine(initial,item['source'],pred[:,:3])
                    pose[:3,3]+=item['center']
                    errors[arm]=error(pose,item['GT'])
            records.append(dict(seed=seed,frame_id=row['frame_id'],errors=errors))
            base.save(args.root/'records.json',records)
            if index%8==0 or index==31:
                print(f'evaluate seed{seed} {index+1}/32',flush=True)
    old=json.loads((HERE.parent/'query_fusion/results/records.json').read_text())
    baseline={r['frame_id']:r['errors']['baseline'] for r in old}
    parity=max(max(abs(a-b) for a,b in zip(r['errors']['pretrained'],baseline[r['frame_id']])) for r in records)
    assert parity<1e-5,parity
    summary={}
    for arm in records[0]['errors']:
        per_seed={str(s):base.metrics([r['errors'][arm] for r in records if r['seed']==s]) for s in SEEDS}
        values=np.array([[m['MPE'],m['MOE']] for m in per_seed.values()])
        summary[arm]=dict(per_seed=per_seed,mean=values.mean(0).tolist(),std=values.std(0,ddof=1).tolist())
    base.save(args.root/'result.json',dict(summary=summary,historical_baseline_max_difference=parity))


def report(args):
    result=json.loads((args.root/'result.json').read_text())
    counts={arm:torch.load(args.root/f'{arm}_{SEEDS[0]}.pt',map_location='cpu')['optimized_parameters'] for arm in ARMS}
    assert len({counts[a] for a in VISUAL})==1
    paired={}
    contrasts=dict(patch_on_dedode={'dedode_patch':1,'dedode_center':-1},patch_on_dino={'dino_patch':1,'dino_center':-1},
        backbone_at_center={'dino_center':1,'dedode_center':-1},backbone_at_patch={'dino_patch':1,'dedode_patch':-1},
        interaction={'dino_patch':1,'dino_center':-1,'dedode_patch':-1,'dedode_center':1})
    for arm in VISUAL:
        for other in ['lidar',arm+'_missing',arm+'_shuffled']:
            contrasts[arm+'_minus_'+other]={arm:1,other:-1}
    for name,terms in contrasts.items():
        per_seed={}
        for seed in SEEDS:
            per_seed[str(seed)]=sum(coefficient*np.array([result['summary'][arm]['per_seed'][str(seed)][m] for m in ['MPE','MOE']]) for arm,coefficient in terms.items()).tolist()
        values=np.array(list(per_seed.values()))
        paired[name]=dict(per_seed=per_seed,mean=values.mean(0).tolist(),std=values.std(0,ddof=1).tolist())
    base.save(args.root/'paired.json',paired)
    base.save(args.root/'parameter_counts.json',counts)
    lines=['# Query × 图像特征交叉消融','',
        '64训练/32开发帧，三配对种子2089/2090/2091，600步，六相机，完全相同two-stage后端。不是独立测试集；三个种子不构成96个独立帧。',
        '两骨干输入均448×616；相同像素位置采样，5×5间隔14像素；中心组使用相同缓存的第12号中心token。独立128D PCA使用完全相同训练可见中心索引拟合。',
        '层次注意力在相机内选择token，再在相机间选择；两级共享query/key。四视觉组参数量相同，但邻域组计算量较大。',
        '', '| 方法 | MPE m ±种子标准差 | MOE ° ±种子标准差 |', '|---|---:|---:|']
    for arm,s in result['summary'].items():
        lines.append(f'| {arm} | {s["mean"][0]:.6f} ± {s["std"][0]:.6f} | {s["mean"][1]:.6f} ± {s["std"][1]:.6f} |')
    lines+=['','## 配对差值','', '| 对比（前减后，负数为误差降低） | ΔMPE m | ΔMOE ° |','|---|---:|---:|']
    for name,p in paired.items():
        lines.append(f'| {name} | {p["mean"][0]:.6f} | {p["mean"][1]:.6f} |')
    lines+=['',f'优化参数量：{counts}',f'历史原始LEADER two-stage基线最大差异：{result["historical_baseline_max_difference"]}',
        'missing使用该融合模型自身微调的回归头，并不恢复原始预训练模型；shuffled在每个相机和邻域位置的有效voxel间打乱图像对应，保持mask。',
        '邻域仅为视觉上下文，未新增3D点或对应，也不假设25个像素属于同一物理表面。PCA、训练及checkpoint选择不使用开发集。',
        '骨干对比包括预训练模型架构与各自PCA；并非只分离语义知识。5×5是相同像素采样范围，不代表两骨干相同感受野。']
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    destination=HERE/'results'
    destination.mkdir(exist_ok=True)
    for name in ['protocol.json','manifest.json','pca_indices.json','extraction.json','records.json','result.json','paired.json','parameter_counts.json','report.md']:
        shutil.copy2(args.root/name,destination/name)
    for path in args.root.glob('*_training.json'):
        shutil.copy2(path,destination/path.name)
    for name in ['storage_recovery.json','protocol_before_storage_fix.json']:
        if (args.root/name).exists():
            shutil.copy2(args.root/name,destination/name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['check','prepare','extract','compress','train','evaluate','report'])
    parser.add_argument('--root',type=Path,default=Path('/home/zhang/leader-query-backbone-factorial'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage=='check':
        import unittest
        from test_model import Tests
        assert unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()
    elif args.stage in ['prepare','report']:
        globals()[args.stage](args)
    else:
        globals()[args.stage](args,json.loads((args.root/'manifest.json').read_text()))


if __name__=='__main__':
    main()
