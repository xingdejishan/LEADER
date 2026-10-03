import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data.leader_nclt_manifest import ManifestLEADERDataset,DEV_DATES,CAM5_INTRINSICS
from tools.far_runtime import make_dataset as original_dataset
from tools.magic_nclt_runtime import sha256_file


class TrainingB:
    def __init__(self,config,sam=True,truth=True):
        self.config=config
        root=Path(config['train_b_root'])
        self.base=ManifestLEADERDataset([root/date/'manifest.json' for date in DEV_DATES],root,'dev',strict_frozen=False,
            sam_cache_manifest=Path(config['train_sam']) if sam else None,expected_cache_sha256=config.get('train_sam_sha256') if sam else None,cache_memory_frames=1)
        self.records=self.base.records
        self.manifest_hashes=self.base.manifest_hashes
        self.sam_store=self.base.sam_store
        self.keys=self.base.keys
        self.truth=truth
        assert len(self)==1200
        for date,digest in config['train_b_manifest_sha256'].items():assert self.manifest_hashes[date]==digest
        self.targets={}
        if truth:
            path=Path(config['train_b_truth'])
            assert sha256_file(path)==config['train_b_truth_sha256']
            self.targets=json.loads(path.read_text())['frames']
            assert set(self.targets)==set(self.keys)

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        sample=self.base[index]
        if self.truth:
            pose=np.asarray(self.targets[sample['frame_key']]['T_WB'],np.float32)
            assert pose.shape==(4,4) and np.isfinite(pose).all()
            sample['T_WB']=pose
        return sample

    def verify_all_input_hashes(self):self.base.verify_all_input_hashes()

    def validate_all_sam_embeddings(self):self.base.validate_all_sam_embeddings()


def make_dataset(config,split):
    return TrainingB(config) if split=='train' else original_dataset(config,'dev')


def spherical_cells(points):
    ranges=np.linalg.norm(points,axis=1)
    rows=np.floor((np.arctan2(points[:,2],np.linalg.norm(points[:,:2],axis=1))+np.pi/2)/np.deg2rad(.5)).astype(np.int64).clip(0,359)
    columns=np.floor((np.arctan2(points[:,1],points[:,0])+np.pi)*1024/(2*np.pi)).astype(np.int64)%1024
    return rows,columns,ranges


def local_structure(raw,queries):
    rows,columns,ranges=spherical_cells(raw)
    cells=rows*1024+columns
    order=np.lexsort((ranges,cells))
    first=order[np.r_[True,cells[order][1:]!=cells[order][:-1]]]
    raster=np.full((360,1024),-1,np.int64)
    raster[rows[first],columns[first]]=first
    qr,qc,query_ranges=spherical_cells(queries)
    row_offsets,col_offsets=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij')
    result=[]
    for start in range(0,len(queries),256):
        stop=min(start+256,len(queries))
        rr=qr[start:stop,None]+row_offsets.ravel()
        cc=(qc[start:stop,None]+col_offsets.ravel())%1024
        ids=raster[rr.clip(0,359),cc]
        neighbours=raw[ids.clip(0)]
        valid=(ids>=0)&(rr>=0)&(rr<360)&(np.linalg.norm(neighbours-queries[start:stop,None],axis=-1)<=np.maximum(.5,.1*query_ranges[start:stop,None]))
        count=valid.sum(-1)
        mean=(neighbours*valid[...,None]).sum(1)/np.maximum(count[:,None],1)
        difference=neighbours-mean[:,None]
        scatter=np.einsum('nk,nki,nkj->nij',valid,difference,difference)/np.maximum(count[:,None,None],1)
        trace=np.trace(scatter,axis1=1,axis2=2)
        supported=(count>=6)&(trace>1e-8)
        scatter=scatter/np.maximum(trace[:,None,None],1e-8)
        scatter[~supported]=0
        result.append(scatter.astype(np.float32))
    return np.concatenate(result) if result else np.empty((0,3,3),np.float32)


def project(points,transform):
    camera=points@transform[:3,:3].T+transform[:3,3]
    z=camera[:,2]
    uv=camera[:,:2]/np.maximum(z[:,None],1e-8)
    uv=uv*np.array([CAM5_INTRINSICS[0,0],CAM5_INTRINSICS[1,1]])+np.array([CAM5_INTRINSICS[0,2],CAM5_INTRINSICS[1,2]])
    valid=(z>0)&(uv[:,0]>=0)&(uv[:,0]<808)&(uv[:,1]>=0)&(uv[:,1]<616)
    pixels=np.floor(uv.clip(-1e6,1e6)).astype(np.int64)
    return uv,z,pixels[:,1]*808+pixels[:,0],valid


def visible_pool(sample,source,max_points=2048):
    raw=np.asarray(sample['points'],np.float32)
    source=source.detach().cpu().numpy().astype(np.float32)
    transform=sample['camera_from_lidar'].numpy()
    uv,z,cells,valid=project(raw,transform)
    ids=np.flatnonzero(valid)
    order=np.lexsort((ids,z[ids],cells[ids]))
    ids=ids[order]
    cell=cells[ids]
    ids=ids[np.r_[True,cell[1:]!=cell[:-1]]] if len(ids) else ids
    depth=np.full(808*616,np.inf,np.float32)
    depth[cells[ids]]=z[ids]
    guv,gz,gcells,gvalid=project(source,transform)
    gids=np.flatnonzero(gvalid)
    gids=gids[gz[gids]<=depth[gcells[gids]]+.4]
    if len(gids):
        order=np.lexsort((gids,gz[gids],gcells[gids]))
        gids=gids[order]
        cells_order=gcells[gids]
        gids=gids[np.r_[True,cells_order[1:]!=cells_order[:-1]]]
    if len(gids)>max_points:gids=gids[np.linspace(0,len(gids)-1,max_points,dtype=np.int64)]
    occupied=np.zeros(808*616,bool)
    occupied[gcells[gids]]=True
    ids=ids[~occupied[cells[ids]]]
    _,voxel_ids=np.unique(np.floor(raw[ids]/.2).astype(np.int64),axis=0,return_index=True)
    ids=ids[np.sort(voxel_ids)]
    remaining=max_points-len(gids)
    if len(ids)>remaining:ids=ids[np.linspace(0,len(ids)-1,remaining,dtype=np.int64)] if remaining else ids[:0]
    points=np.concatenate((source[gids],raw[ids]))
    pixels=np.concatenate((guv[gids],uv[ids])).astype(np.float32)
    visible_ids=np.concatenate((gids,np.full(len(ids),-1,np.int64)))
    assert len(points)==len(pixels)==len(visible_ids) and len(points)<=max_points
    return torch.from_numpy(points.copy()),torch.from_numpy(pixels),torch.from_numpy(visible_ids),torch.from_numpy(local_structure(raw,source))


def load_frame(dataset,index,device,source,max_points=2048):
    sample=dataset[index]
    points,pixels,visible_ids,structure=visible_pool(sample,source,max_points)
    with Image.open(dataset.records[index].image_path) as image:
        rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
    return sample,points.to(device),pixels.to(device),rgb.to(device),visible_ids.to(device),structure.to(device)


class InitialCache:
    def __init__(self,config,dataset,center):
        path=Path(config['train_initial_cache'])
        assert sha256_file(path)==config['train_initial_cache_sha256']
        self.root=path.parent
        self.cache=json.loads(path.read_text())
        assert self.cache['ground_truth_used_for_prediction'] is False
        assert self.cache['base_checkpoint_sha256']==config['base_checkpoint_sha256']
        assert self.cache['manifest_sha256']==dataset.manifest_hashes
        assert self.cache['seed']==config['seed'] and set(self.cache['frames'])==set(dataset.keys)
        assert all(config['source_sha256'][name]==digest for name,digest in self.cache['producer_source_sha256'].items())
        np.testing.assert_array_equal(np.asarray(self.cache['center_t'],np.float32),center.detach().cpu().numpy())
        for record in dataset.records:
            row=self.cache['frames'][record.frame_key]
            assert row['scan_sha256']==record.scan_sha256 and row['image_sha256']==record.image_sha256
            assert sha256_file(self.root/row['file'])==row['sha256']

    def frame(self,key,device):
        row=self.cache['frames'][key]
        path=self.root/row['file']
        assert sha256_file(path)==row['sha256']
        with np.load(path,allow_pickle=False) as archive:
            values=[torch.from_numpy(archive[name].copy()).to(device) for name in ('initial','source','target')]
        assert values[0].shape==(4,4) and values[1].shape==values[2].shape and values[1].shape[1]==3
        assert all(torch.isfinite(value).all() for value in values)
        return values
