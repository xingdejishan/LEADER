"""Labels for the 1502 hard frames, split into a training and a test half.

The split is by DATE, not by frame index: frames adjacent in time on the same
date see the same road, so an index split would put the same scene on both sides.
Two dates train and two dates test.

The query side is a hard frame: its own scan and its ground-truth pose give every
LiDAR point a true world position. The reference side is exactly the deployed
provider list for that frame (CLS top5 union the 5 references nearest the frame's
own SC2 position), so the supervision distribution matches deployment.

The pair is kept only if it survives the same checks as the library labelling:
reprojection in front of and inside the reference camera, a reference scan return
near that pixel, a consistent depth, and a SuperPoint keypoint near both pixels.
Labels are emitted as (query keypoint index, reference keypoint index).
"""
from common import *
import argparse
import importlib.util
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
H=PLAN['label_dataset_hard']
OUT=STORE/H['output_dir']
OUT.mkdir(parents=True,exist_ok=True)
# This stage builds TRAINING labels, so ground-truth poses are legitimate here.
# deny_GT is deliberately not installed; instead the split itself is enforced:
# every frame whose ground truth is actually touched must belong to a training
# date, and the receipt below records exactly which frames those were.
ground_truth_used=set()
TRAIN_DATES=set(H['dates'])
TEST_DATES=set(H['test_dates'])
assert not (TRAIN_DATES & TEST_DATES)

library=load(STORE/'library.json')
references=library['references']
T_BC=np.asarray(library['T_BC']); T_CB=np.linalg.inv(T_BC)
MATRIX=model_matrix()
provider=np.load('/data2/zhangcy/sp-provider-cls5-sc2pos5-20261015-v1/provider.npz',allow_pickle=False)
provider_keys=list(provider['keys'].astype(str))
provider_offsets=provider['offsets'].astype(np.int64)
provider_indices=provider['indices'].astype(np.int64)
provider_of={k:i for i,k in enumerate(provider_keys)}
queries=load(STORE/'queries.json')['queries']

POLICY=Path(H['policy_project'])
spec=importlib.util.spec_from_file_location('policy',POLICY/'nclt_reference_policy.py')
helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
scored=np.load(Path(H['scored']),allow_pickle=False)
truth,_ ,_=helper.apply_reference_policy(scored['dates'],scored['timestamps_us'],scored['raw'],POLICY/'nclt-reference-policy.json')
truth_of={k:i for i,k in enumerate(scored['keys'].astype(str))}

extractor=SuperPoint(max_num_keypoints=int(L['max_num_keypoints'])).eval()
for parameter in extractor.parameters():
    parameter.requires_grad_(False)
FEATURES=STORE/'label-dataset-hard-features'
GEOMETRY=STORE/'label-dataset-geometry'
FEATURES.mkdir(exist_ok=True); GEOMETRY.mkdir(exist_ok=True)


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


def keypoints(image_path,tag):
    path=FEATURES/('%s.pt'%tag)
    if path.is_file():
        return torch.load(path,map_location='cpu').numpy()
    array=model_image(image_path)
    x=torch.from_numpy(array).permute(2,0,1).float()/255
    with torch.inference_mode():
        value=extractor.extract(x,resize=None)['keypoints'][0].numpy().astype(np.float64)
    handle,temp=tempfile.mkstemp(dir=str(path.parent),suffix='.pt')
    os.close(handle)
    torch.save(torch.from_numpy(value),temp)
    os.replace(temp,path)
    return value


def scan_geometry(scan_path,index,tag):
    path=GEOMETRY/('%s.npz'%tag)
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            return z['model'].astype(np.float64),z['camera'].astype(np.float64)
    body=decode(scan_path)
    body=body[np.isfinite(body).all(1)]
    camera=transform(body,T_CB)
    keep=(camera[:,2]>L['min_depth_m'])&(camera[:,2]<L['max_depth_m'])
    camera=camera[keep]
    projected=camera@K.T
    pixel=projected[:,:2]/projected[:,2:3]
    model=(np.column_stack((pixel,np.ones(len(pixel))))@MATRIX.T)[:,:2]
    inside=(model[:,0]>=0)&(model[:,0]<L['model_width'])&(model[:,1]>=0)&(model[:,1]<L['model_height'])
    handle,temp=tempfile.mkstemp(dir=str(path.parent),suffix='.npz')
    os.close(handle)
    np.savez_compressed(temp,model=model[inside].astype(np.float32),camera=camera[inside].astype(np.float32))
    os.replace(temp,path)
    return model[inside],camera[inside]


started=time.monotonic()
processed=0
totals=dict(queries=0,pairs=0,labels=0)
for query in queries:
    date=query['key'].split('/')[0]
    if date not in H['dates']:
        continue
    if query['index']%argument.of!=argument.shard:
        continue
    if argument.limit and processed>=argument.limit:
        break
    guard()
    path=OUT/('%06d.json'%query['index'])
    if path.is_file():
        processed+=1
        continue
    row=provider_of[query['key']]
    reference_list=provider_indices[provider_offsets[row]:provider_offsets[row+1]]
    key=query['key']
    truth_pose=truth[truth_of[key]]
    ground_truth_used.add(key)
    query_model,query_camera=scan_geometry(query['scan_path'],query['index'],'q-%06d'%query['index'])
    # the policy supplies the BODY pose, so the camera pose is T_WB @ T_BC; the
    # scan geometry above is already in the camera frame
    query_world=transform(query_camera,truth_pose@T_BC)
    query_keypoints=keypoints(query['image_path'],'q-%06d'%query['index'])
    if len(query_keypoints)==0 or len(query_world)==0:
        save(path,dict(index=query['index'],key=key,date=date,pairs=[]))
        processed+=1
        continue
    query_tree=cKDTree(query_keypoints)
    pairs=[]
    for reference_index in reference_list:
        reference_index=int(reference_index)
        reference_key='%06d'%reference_index
        inverse=np.linalg.inv(np.asarray(references[reference_index]['T_WC'],dtype=np.float64))
        camera=transform(query_world,inverse)
        front=(camera[:,2]>L['min_depth_m'])&(camera[:,2]<L['max_depth_m'])
        if not front.any():
            continue
        local=camera[front]
        projected=local@K.T
        pixel=projected[:,:2]/projected[:,2:3]
        model=(np.column_stack((pixel,np.ones(len(pixel))))@MATRIX.T)[:,:2]
        inside=(model[:,0]>=0)&(model[:,0]<L['model_width'])&(model[:,1]>=0)&(model[:,1]<L['model_height'])
        if not inside.any():
            continue
        index=np.flatnonzero(front)[inside]
        model_r=model[inside]; depth_r=local[inside][:,2]
        reference_scan,reference_camera=scan_geometry(
            '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT/%s/velodyne_sync/%s.bin'%tuple(references[reference_index]['key'].split('/')),
            reference_index,'r-'+reference_key)
        if len(reference_scan)==0:
            continue
        distance,nearest=cKDTree(reference_scan).query(model_r)
        has_return=distance<=L['binding_radius_px']
        consistent=np.zeros(len(model_r),bool)
        if has_return.any():
            measured=reference_camera[nearest[has_return]][:,2]
            consistent[has_return]=np.abs(measured-depth_r[has_return])<=L['depth_tolerance_m']
        if not consistent.any():
            continue
        distance_q,index_q=query_tree.query(query_model[index][consistent])
        has_q=distance_q<=L['keypoint_radius_px']
        if not has_q.any():
            continue
        reference_keypoints=keypoints(references[reference_index]['image_path'],'r-'+reference_key)
        if len(reference_keypoints)==0:
            continue
        distance_r,index_r=cKDTree(reference_keypoints).query(model_r[consistent][has_q])
        has_r=distance_r<=L['keypoint_radius_px']
        if not has_r.any():
            continue
        best={}
        for qi,rj,cost in zip(index_q[has_q][has_r],index_r[has_r],distance_q[has_q][has_r]+distance_r[has_r]):
            current=best.get(int(qi))
            if current is None or cost<current[1]:
                best[int(qi)]=(int(rj),float(cost))
        labels=[[qi,rj] for qi,(rj,cost) in best.items()]
        if len(labels)>=int(H['min_labels_per_pair']):
            pairs.append(dict(reference=reference_index,labels=labels))
    save(path,dict(index=query['index'],key=key,date=date,pairs=pairs))
    totals['queries']+=1
    totals['pairs']+=len(pairs)
    totals['labels']+=sum(len(p['labels']) for p in pairs)
    processed+=1
    if processed%25==0:
        print(json.dumps(dict(shard=argument.shard,processed=processed,seconds=round(time.monotonic()-started,1),**totals)),flush=True)
for k in ground_truth_used:
    assert k.split('/')[0] in TRAIN_DATES,(k,'ground truth touched for a non-training date')
receipt=dict(shard=argument.shard,ground_truth_frames=len(ground_truth_used),
             dates_touched=sorted(set(k.split('/')[0] for k in ground_truth_used)),
             test_dates_untouched=sorted(TEST_DATES))
save(OUT/('receipt-%02d.json'%argument.shard),receipt)
print(json.dumps(dict(shard=argument.shard,complete=True,processed=processed,seconds=round(time.monotonic()-started,1),**totals)))
print(json.dumps(receipt))
