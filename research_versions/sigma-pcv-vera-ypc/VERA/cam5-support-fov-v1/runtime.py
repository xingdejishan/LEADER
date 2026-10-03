import json
from pathlib import Path

from common import PLAN, ROOT, STORE, guard, sha
import MinkowskiEngine as ME
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from data.leader_nclt_manifest import ManifestLEADERDataset, collate_leader_frames
from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER
from tools.magic_nclt_runtime import trr_loss, make_matcher, state_dict_sha256
from utils.full_pool_robust_v1 import full_pool_refine
from vera import VERA,geometry,coverage
from fov_visibility_v2 import prepare

class Dataset:
    def __init__(self,split):
        self.split=split
        dates=PLAN['train_dates'] if split=='train' else PLAN['dev_dates']
        root=Path(PLAN['data_root'])
        self.base=ManifestLEADERDataset([root/d/'manifest.json' for d in dates],root,split,strict_frozen=False,cache_memory_frames=0)
        self.inputs=json.loads((STORE/'inputs.json').read_text())[split]

    def __len__(self):return len(self.inputs)

    def load(self,i,multimodal,device):
        record=self.inputs[i];sample=self.base[record['dataset_index']]
        assert sample['frame_key']==record['key']
        batch=collate_leader_frames([sample]);extra=None
        if multimodal:
            images=[];sam=[]
            for c in record['cameras']:
                assert sha(c['image'])==c['image_sha256'] and sha(c['cache'])==c['cache_file_sha256']
                image=np.asarray(Image.open(c['image']).convert('RGB'),dtype=np.float32).copy()/255
                images.append(torch.from_numpy(image).permute(2,0,1))
                sam.append(torch.from_numpy(np.load(c['cache'],allow_pickle=False)).permute(2,0,1))
            images=torch.stack(images).to(device);mean=images.new_tensor([.485,.456,.406])[None,:,None,None];std=images.new_tensor([.229,.224,.225])[None,:,None,None]
            extra={'images':(images-mean)/std,'sam':torch.stack(sam).to(device,dtype=torch.float32),'cameras':record['cameras'],'raw':sample['points']}
            extra['visibility']=prepare(extra['raw'],extra['cameras'],coverage(),PLAN['fov_full_coverage_threshold'])
        return batch,extra,record

class System(torch.nn.Module):
    def __init__(self,arm,device):
        super().__init__();self.arm=arm
        self.leader=LEADER(in_channels=3,out_channels=4,feat_channels=512,width=1024)
        checkpoint=torch.load(PLAN['checkpoint'],map_location='cpu')
        self.leader.load_state_dict(checkpoint['model'],strict=True)
        self.register_buffer('center',torch.as_tensor(checkpoint['center_t'],dtype=torch.float32))
        self.vera=VERA() if arm=='vera' else None
        self.to(device)

    def mode(self,epoch):
        self.train()
        for p in self.leader.encoder.parameters():p.requires_grad_(False)
        if epoch>PLAN['stage1_epochs']:
            for p in self.leader.encoder.decoders.parameters():p.requires_grad_(True)
        for m in self.modules():
            if isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,ME.MinkowskiBatchNorm)):m.eval()
        return self

    def forward(self,batch,extra,record,device):
        features,points=self.encode(batch,device)
        return self.from_encoded(features,points,extra,record,device)

    def encode(self,batch,device):
        sparse=ME.SparseTensor(features=batch['feats'].to(device),coordinates=batch['coords'].to(device,dtype=torch.int32))
        encoded=self.leader.encoder(sparse)
        stride=torch.tensor(encoded.tensor_stride,device=device,dtype=torch.float32)
        points=polar_voxel_centers(encoded.C,stride,.2,1024)
        return encoded.F,points

    def from_encoded(self,features,points,extra,record,device):
        self.last_source_points=points
        eta=None;active=None
        if self.vera is not None:
            geo_dir=STORE/'geometry';geo_dir.mkdir(exist_ok=True)
            path=geo_dir/(record['key'].replace('/','_')+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:geo={k:saved[k] for k in saved.files}
                current=points.detach().cpu().numpy()
                if not np.array_equal(geo['points'],current):
                    distances,indices=cKDTree(geo['points']).query(current,workers=1)
                    assert distances.max()<1e-8 and len(set(indices.tolist()))==len(indices)
                    geo={k:v[indices] if k!='eta' else v for k,v in geo.items()}
                assert np.array_equal(geo['points'],current)
            else:
                geo=geometry(points,extra['raw'],extra['cameras'])
                np.savez_compressed(path,**geo)
            features,points,eta,active=self.vera(features,points,geo,extra['images'],extra['sam'],extra['cameras'],extra['visibility'])
        return self.leader.decoder(features),points,eta,active

def loss(system,output,points,eta,batch,device):
    pose=batch['T'].to(device)[0]@torch.linalg.inv(batch['T_corr'].to(device)[0])
    target=points@pose[:3,:3].T+pose[:3,3]-system.center
    trr,raw=trr_loss(target,output[:,:3],output[:,3],torch.zeros(len(points),device=device,dtype=torch.long))
    penalty=output.new_tensor(0.) if eta is None else PLAN['beta']*eta.square().sum(-1).mean()
    return trr+penalty,{'trr':float(trr.detach()),'raw':float(raw.detach()),'penalty':float(penalty.detach())}

def pose_from_correspondences(system,output,points,batch,seed,device):
    count=max(min(50,len(output)),int(.5*len(output)))
    assert count>=3
    idx=output[:,3].topk(count).indices;source=points[idx].float();target=output[idx,:3].float()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    initial=make_matcher(device).estimator(source[None],target[None])[0]
    result=full_pool_refine(initial,source,target)
    result[:3,3]+=system.center;result=result@batch['T_corr'].to(device)[0]
    assert torch.isfinite(result).all()
    return result.detach().cpu().double().numpy(),count
