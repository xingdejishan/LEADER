"""Materialise the LiDAR-reprojection supervision as a training set (step 4 -> 5).

For every selected cross-date pair this repeats the checked construction and, for
each surviving label, resolves the query and reference keypoint that the two
projected pixels fall on, emitting an index pair (query keypoint, reference
keypoint). That index pair is exactly what a LightGlue fine-tune consumes: the
frozen SuperPoint keypoints are reproduced deterministically from the frozen
configuration, so indices are stable.

Pairing is by co-visibility but the score is the number of labels that survive
ALL checks, not the raw frustum count, because the earlier measurement showed the
frustum count selects large baselines and then loses 73 percent of the points at
the occlusion check.

One label per query keypoint is kept, the one with the smallest combined
keypoint distance, so each query keypoint has a single ground-truth target.
Only the library's own mapping poses on the four training dates are used.
"""
from common import *
import argparse
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(VENDOR_PY))
import torch
from PIL import Image
torch.hub.set_dir(str(HUB))
torch.set_num_threads(1)
torch.manual_seed(20)
from lightglue import SuperPoint

parser=argparse.ArgumentParser()
parser.add_argument('--shard',type=int,default=0)
parser.add_argument('--of',type=int,default=1)
parser.add_argument('--limit',type=int,default=0)
argument=parser.parse_args()

L=PLAN['label_yield']
D=PLAN['label_dataset']
OUT=STORE/D['output_dir']
OUT.mkdir(parents=True,exist_ok=True)
FEATURES=STORE/'label-dataset-features'
FEATURES.mkdir(exist_ok=True)

library=load(STORE/'library.json')
references=library['references']
T_BC=np.asarray(library['T_BC']); T_CB=np.linalg.inv(T_BC)
MATRIX=model_matrix()
extractor=SuperPoint(max_num_keypoints=int(L['max_num_keypoints'])).eval()
for parameter in extractor.parameters():
    parameter.requires_grad_(False)

SCAN='/data2/dataset/NCLT/%s/velodyne_sync/%s.bin'
positions=np.asarray([r['body_position_world'] for r in references],dtype=np.float64)
dates=np.asarray([r['key'].split('/')[0] for r in references])
index_of_date={}
for i,date in enumerate(dates):
    index_of_date.setdefault(date,[]).append(i)
train_dates=sorted(index_of_date)


def model_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        image=image.transpose(Image.Transpose.ROTATE_270)
        w,h=image.size
        scale=min(1.,L['image_edge']/max(w,h))
        rw,rh=round(w*scale),round(h*scale)
        image=image.resize((rw,rh),Image.Resampling.BICUBIC)
        cw,ch=rw//14*14,rh//14*14
        left,top=(rw-cw)//2,(rh-ch)//2
        image=image.crop((left,top,left+cw,top+ch))
        return np.asarray(image).copy()


def keypoints(index):
    path=FEATURES/('%06d.pt'%index)
    if path.is_file():
        return torch.load(path,map_location='cpu').numpy()
    array=model_image(references[index]['image_path'])
    x=torch.from_numpy(array).permute(2,0,1).float()/255
    with torch.inference_mode():
        value=extractor.extract(x,resize=None)['keypoints'][0].numpy().astype(np.float64)
    handle,temp=tempfile.mkstemp(dir=str(path.parent),suffix='.pt')
    os.close(handle)
    torch.save(torch.from_numpy(value),temp)
    os.replace(temp,path)
    return value


def camera_geometry(index):
    row=references[index]
    body=decode(SCAN%tuple(row['key'].split('/')))
    body=body[np.isfinite(body).all(1)]
    camera=transform(body,T_CB)
    keep=(camera[:,2]>L['min_depth_m'])&(camera[:,2]<L['max_depth_m'])
    camera=camera[keep]
    projected=camera@K.T
    pixel=projected[:,:2]/projected[:,2:3]
    model=(np.column_stack((pixel,np.ones(len(pixel))))@MATRIX.T)[:,:2]
    inside=(model[:,0]>=0)&(model[:,0]<L['model_width'])&(model[:,1]>=0)&(model[:,1]<L['model_height'])
    return model[inside],camera[inside]


GEOMETRY=STORE/'label-dataset-geometry'
GEOMETRY.mkdir(exist_ok=True)


def reference_geometry(index):
    """disk-cached own-scan projection of a reference image: model pixels and
    camera-frame points, both restricted to the model image"""
    path=GEOMETRY/('%06d.npz'%index)
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            return z['model'].astype(np.float64),z['camera'].astype(np.float64)
    model,camera=camera_geometry(index)
    handle,temp=tempfile.mkstemp(dir=str(path.parent),suffix='.npz')
    os.close(handle)
    np.savez_compressed(temp,model=model.astype(np.float32),camera=camera.astype(np.float32))
    os.replace(temp,path)
    return model,camera


def try_pair(query_model,query_world,query_tree,reference_index):
    """returns the list of (query keypoint index, reference keypoint index) labels

    The query-side pixel for each surviving point is carried through by index, so
    the query keypoint lookup happens in query image space and the reference
    keypoint lookup in reference image space, as it must.
    """
    inverse=np.linalg.inv(np.asarray(references[reference_index]['T_WC'],dtype=np.float64))
    camera=transform(query_world,inverse)
    front=(camera[:,2]>L['min_depth_m'])&(camera[:,2]<L['max_depth_m'])
    if not front.any():
        return []
    local_camera=camera[front]
    projected=local_camera@K.T
    pixel=projected[:,:2]/projected[:,2:3]
    model=(np.column_stack((pixel,np.ones(len(pixel))))@MATRIX.T)[:,:2]
    inside=(model[:,0]>=0)&(model[:,0]<L['model_width'])&(model[:,1]>=0)&(model[:,1]<L['model_height'])
    if not inside.any():
        return []
    index=np.flatnonzero(front)[inside]
    model_r=model[inside]
    depth_r=local_camera[inside][:,2]
    model_q=query_model[index]
    reference_scan,reference_camera_scan=reference_geometry(reference_index)
    if len(reference_scan)==0:
        return []
    distance,nearest=cKDTree(reference_scan).query(model_r)
    has_return=distance<=L['binding_radius_px']
    consistent=np.zeros(len(model_r),bool)
    if has_return.any():
        measured=reference_camera_scan[nearest[has_return]][:,2]
        consistent[has_return]=np.abs(measured-depth_r[has_return])<=L['depth_tolerance_m']
    if not consistent.any():
        return []
    distance_q,index_q=query_tree.query(model_q[consistent])
    has_q=distance_q<=L['keypoint_radius_px']
    if not has_q.any():
        return []
    reference_keypoints=keypoints(reference_index)
    if len(reference_keypoints)==0:
        return []
    distance_r,index_r=cKDTree(reference_keypoints).query(model_r[consistent][has_q])
    has_r=distance_r<=L['keypoint_radius_px']
    if not has_r.any():
        return []
    best={}
    for qi,rj,cost in zip(index_q[has_q][has_r],index_r[has_r],distance_q[has_q][has_r]+distance_r[has_r]):
        current=best.get(int(qi))
        if current is None or cost<current[1]:
            best[int(qi)]=(int(rj),float(cost))
    return [(qi,rj) for qi,(rj,cost) in best.items()]


started=time.monotonic()
processed=0
totals=dict(queries=0,pairs=0,labels=0)
for query_index in sorted(sum(index_of_date.values(),[])):
    if query_index%argument.of!=argument.shard:
        continue
    if argument.limit and processed>=argument.limit:
        break
    guard()
    path=OUT/('%06d.json'%query_index)
    if path.is_file():
        processed+=1
        continue
    date=dates[query_index]
    distance=np.linalg.norm(positions-positions[query_index],axis=1)
    other=np.flatnonzero((dates!=date)&(distance<=L['pool_radius_m']))
    if len(other)==0:
        save(path,dict(index=query_index,key=references[query_index]['key'],pairs=[]))
        processed+=1
        continue
    query_model,query_camera=camera_geometry(query_index)
    query_world=transform(query_camera,np.asarray(references[query_index]['T_WC'],dtype=np.float64))
    query_keypoints=keypoints(query_index)
    if len(query_keypoints)==0:
        save(path,dict(index=query_index,key=references[query_index]['key'],pairs=[]))
        processed+=1
        continue
    query_tree=cKDTree(query_keypoints)
    shortlist=other[np.argsort(distance[other])][:int(D['shortlist'])]
    scored=[]
    for reference_index in shortlist:
        labels=try_pair(query_model,query_world,query_tree,int(reference_index))
        scored.append((len(labels),int(reference_index),labels))
    scored.sort(key=lambda x:-x[0])
    chosen=[s for s in scored[:int(D['references_per_query'])] if s[0]>=int(D['min_labels_per_pair'])]
    entry=dict(index=query_index,key=references[query_index]['key'],
               pairs=[dict(reference=r,labels=l) for _,r,l in chosen])
    save(path,entry)
    totals['queries']+=1
    totals['pairs']+=len(chosen)
    totals['labels']+=sum(len(l) for _,_,l in chosen)
    processed+=1
    if processed%10==0:
        print(json.dumps(dict(shard=argument.shard,processed=processed,seconds=round(time.monotonic()-started,1),**totals)),flush=True)
print(json.dumps(dict(shard=argument.shard,complete=True,processed=processed,seconds=round(time.monotonic()-started,1),**totals)))
