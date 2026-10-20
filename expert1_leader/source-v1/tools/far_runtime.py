import json
import os
from pathlib import Path
import resource
import shutil

import numpy as np
from PIL import Image
import torch

from data.leader_nclt_manifest import ManifestLEADERDataset,TRAIN_DATES,DEV_DATES,CAM5_INTRINSICS
from tools.magic_nclt_runtime import safe_json_dump,sha256_file


def budget(output):
    if shutil.disk_usage(output).free<3*2**30:
        raise RuntimeError('disk floor reached')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>12*2**30:
        raise RuntimeError('process memory budget reached')
    info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(info['MemAvailable'].split()[0])*1024<32*2**30:
        raise RuntimeError('system memory floor reached')
    if sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())>2*2**30:
        raise RuntimeError('experiment output storage budget reached')


def setup(output):
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.50)
    budget(output)


def make_dataset(config,split):
    dates=TRAIN_DATES if split=='train' else DEV_DATES
    cache=config['train_sam'] if split=='train' else config['dev_sam']
    dataset=ManifestLEADERDataset([Path(config['data_root'])/date/'manifest.json' for date in dates],
        Path(config['data_root']),split,sam_cache_manifest=Path(cache),
        expected_cache_sha256=config['train_sam_sha256'] if split=='train' else config['dev_sam_sha256'],
        cache_memory_frames=1)
    assert len(dataset)==1200
    return dataset


def visible_points(sample,max_points=2048):
    points=sample['points']
    voxel=np.floor(points/.2).astype(np.int64)
    _,indices=np.unique(voxel,axis=0,return_index=True)
    points=points[np.sort(indices)]
    transform=sample['camera_from_lidar'].numpy()
    camera=points@transform[:3,:3].T+transform[:3,3]
    z=camera[:,2]
    uv=camera[:,:2]/np.maximum(z[:,None],1e-8)
    uv=uv*np.array([CAM5_INTRINSICS[0,0],CAM5_INTRINSICS[1,1]])+np.array([CAM5_INTRINSICS[0,2],CAM5_INTRINSICS[1,2]])
    inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<808)&(uv[:,1]>=0)&(uv[:,1]<616)
    indices=np.flatnonzero(inside)
    pixels=np.floor(uv[indices]).astype(np.int64)
    cell=pixels[:,1]*808+pixels[:,0]
    order=np.lexsort((indices,z[indices],cell))
    indices=indices[order]
    cell=cell[order]
    indices=indices[np.r_[True,cell[1:]!=cell[:-1]]] if len(indices) else indices
    if len(indices)>max_points:
        indices=indices[np.linspace(0,len(indices)-1,max_points,dtype=np.int64)]
    return torch.from_numpy(points[indices].copy()),torch.from_numpy(uv[indices].astype(np.float32))


def load_frame(dataset,index,device,max_points=2048):
    sample=dataset[index]
    points,pixels=visible_points(sample,max_points)
    record=dataset.records[index]
    with Image.open(record.image_path) as image:
        rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
    return sample,points.to(device),pixels.to(device),rgb.to(device)


def observation(model,sample,pixels,rgb,device):
    return model.observation(sample['sam_features'].to(device=device,dtype=torch.float32),rgb,pixels,
        sample['sam_metadata']['resized_size_hw'])


def save_checkpoint(path,state):
    temp=path.with_suffix('.tmp')
    torch.save(state,temp)
    os.replace(temp,path)
    return sha256_file(path)


def verify_sources(config):
    root=Path(__file__).resolve().parents[1]
    for name,digest in config['source_sha256'].items():
        if sha256_file(root/name)!=digest:
            raise ValueError('frozen runtime source differs: '+name)
    if sha256_file(config['base_checkpoint'])!=config['base_checkpoint_sha256']:
        raise ValueError('base checkpoint changed')


def report(output,**data):
    safe_json_dump(Path(output)/'progress.json',data)
    print(json.dumps(data,allow_nan=False),flush=True)
