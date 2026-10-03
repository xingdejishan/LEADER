import json
import os
from pathlib import Path
import resource
import shutil
import sys

import MinkowskiEngine as ME
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from data.leader_nclt_manifest import collate_leader_frames,DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION
from models.magic_fusion import polar_voxel_centers
from models.sigma_pcv import SigmaPCV,decode,enable
from models.sigma_far import SigmaFAR
from tools.far_runtime import make_dataset,observation,verify_sources
from tools.far_covolume_runtime import visible_pool,project
from tools.magic_nclt_runtime import load_pretrained_model,sha256_file,state_dict_sha256,safe_json_dump,make_matcher


def budget(store):
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>10*1024**3:
        raise RuntimeError('own RSS exceeded 10 GiB')
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<32*1024**3 or shutil.disk_usage(store).free<40*1024**3:
        raise RuntimeError('resource floor reached')


def initialize(config):
    assert sys.prefix=='/home/zhangcy/.conda/envs/leader'
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(config['gpu'])
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device=torch.device('cuda:0')
    gt_root=str(Path(config['parent_config']['gt_root']).resolve())
    def deny_truth(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and gt_root in os.path.abspath(os.fsdecode(args[0])):
            raise PermissionError('evaluation ground truth prohibited in GPU worker')
    sys.addaudithook(deny_truth)
    verify_sources(config['parent_config'])
    root=Path(__file__).resolve().parents[1]
    assert all(sha256_file(root/name)==digest for name,digest in config['new_source_sha256'].items())
    assert sha256_file(config['far_checkpoint'])==config['far_checkpoint_sha256']
    leader,saved=load_pretrained_model(config['parent_config']['base_checkpoint'],False,device)
    center=torch.as_tensor(saved['center_t'],device=device,dtype=torch.float32).reshape(3)
    del saved
    far=SigmaFAR(**config['parent_config']['far_covolume']).to(device)
    saved=torch.load(config['far_checkpoint'],map_location='cpu')
    assert saved['config']==config['parent_config'] and saved['epoch']==config['far_epoch']
    far.load_state_dict(saved['model'],strict=True)
    del saved
    enable(far)
    pcv=SigmaPCV(**config['pcv']).to(device)
    leader.eval()
    far.eval()
    pcv.eval()
    return leader,far,pcv,center,device


def train_mode(leader,far,pcv,stage):
    leader.requires_grad_(stage==1)
    leader.train(stage==1)
    for module in leader.modules():
        if isinstance(module,(torch.nn.modules.batchnorm._BatchNorm,ME.MinkowskiBatchNorm)):
            module.eval()
    far.requires_grad_(True).train()
    pcv.requires_grad_(True).train()
    if stage==1:
        for name,p in far.named_parameters():
            if not name.startswith(('field.','detail.','descriptor.')):
                p.requires_grad_(False)
    else:
        far.raw_structure.requires_grad_(False)
        pcv.raw_temperature.requires_grad_(False)


def network_row(sample,leader,device,center,truth=False):
    online={k:v for k,v in sample.items() if k!='T_WB'}
    batch=collate_leader_frames([online])
    assert 'T' not in batch
    encoded=leader.encoder(ME.SparseTensor(features=batch['feats'].to(device),coordinates=batch['coords'].to(device)))
    source=polar_voxel_centers(encoded.C,torch.tensor(encoded.tensor_stride,device=device,dtype=torch.float32),DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION)
    predicted,hidden=decode(leader.decoder,encoded.F)
    result=dict(source=source,mu=predicted[:,:3],reliability=predicted[:,3],hidden=hidden,frame_key=sample['frame_key'])
    if truth:
        pose=torch.as_tensor(sample['T_WB'],device=device,dtype=torch.float32).clone()
        pose[:3,3]-=center
        result.update(truth=source@pose[:3,:3].T+pose[:3,3],pose_truth=pose)
    return result


def support_set(sample,source,radius=1.5,k=8):
    raw=np.asarray(sample['points'],np.float32)
    points=source.detach().cpu().numpy().astype(np.float32)
    transform=sample['camera_from_lidar'].numpy()
    _,depth,cells,valid=project(raw,transform)
    zbuffer=np.full(808*616,np.inf,np.float32)
    np.minimum.at(zbuffer,cells[valid],depth[valid])
    uv,z,cells,valid=project(points,transform)
    ids=np.flatnonzero(valid)
    ids=ids[z[ids]<=zbuffer[cells[ids]]+.4]
    if len(ids):
        order=np.lexsort((ids,z[ids],cells[ids]))
        ids=ids[order]
        ids=ids[np.r_[True,cells[ids][1:]!=cells[ids][:-1]]]
        distance,local=cKDTree(points[ids]).query(points,k=list(range(1,k+1)),distance_upper_bound=radius,workers=1)
        mask=np.isfinite(distance)
        neighbours=ids[np.minimum(local,len(ids)-1)]
    else:
        neighbours=np.zeros((len(points),k),np.int64)
        mask=np.zeros((len(points),k),bool)
    device=source.device
    return dict(support_ids=torch.as_tensor(ids,device=device),support_pixels=torch.as_tensor(uv[ids],device=device,dtype=torch.float32),
        neighbours=torch.as_tensor(neighbours,device=device),support_valid=torch.as_tensor(mask,device=device))


def image_observations(sample,record,row,far,device,backend=False):
    with Image.open(record.image_path) as image:
        rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float().to(device)/255
    pixels=row['support_pixels']
    result={}
    if backend:
        count=max(min(50,len(row['source'])),int(.5*len(row['source'])))
        selected=row['reliability'].topk(count).indices
        points,pool_pixels,visible,structure=[v.to(device) for v in visible_pool(sample,row['source'][selected],2048)]
        pixels=torch.cat((pixels,pool_pixels))
        result.update(selected=selected,points=points,visible=visible,structure=structure)
    if len(pixels):
        observed=observation(far,sample,pixels,rgb,device)
    else:
        observed=row['mu'].new_empty((0,32))
    count=len(row['support_ids'])
    descriptors=row['mu'].new_zeros((len(row['mu']),32)).index_copy(0,row['support_ids'],observed[:count])
    result['descriptors']=descriptors
    if backend:
        result['observed']=observed[count:]
    return result


def posterior(row,far,pcv):
    L=pcv.cholesky(row['hidden'])
    mean,cov,supported=pcv.posterior(far.field,row['mu'],L,row['descriptors'],row['neighbours'],row['support_valid'])
    return L,mean,cov,supported


def backend_target(row,L,mean,cov):
    ids=row['selected']
    return dict(prior=row['mu'][ids],covariance=(L@L.transpose(-1,-2))[ids],mean=mean[ids],posterior_covariance=cov[ids])


@torch.no_grad()
def sc2(config,row,mean,index):
    torch.manual_seed(config['seed']+int(index))
    torch.cuda.manual_seed_all(config['seed']+int(index))
    selected=row['selected']
    pose=make_matcher(mean.device).estimator(row['source'][selected][None],mean[selected][None])[0]
    assert torch.isfinite(pose).all()
    return pose


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu(v) for v in value)
    return value


def move(row,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in row.items()}


def save(path,leader,far,pcv,config,**extra):
    temp=path.with_suffix('.tmp')
    torch.save(dict(leader=cpu(leader.state_dict()),far=cpu(far.state_dict()),pcv=cpu(pcv.state_dict()),config_sha256=config['self_sha256'],**extra),temp)
    temp.replace(path)
    return sha256_file(path)


def load(path,leader,far,pcv,config):
    saved=torch.load(path,map_location='cpu')
    assert saved['config_sha256']==config['self_sha256']
    for name,model in [('leader',leader),('far',far),('pcv',pcv)]:
        model.load_state_dict(saved[name],strict=True)
    return saved


def progress(store,**values):
    safe_json_dump(store/'progress.json',values)
    print(json.dumps(values,allow_nan=False),flush=True)
