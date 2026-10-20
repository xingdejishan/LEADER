"""Stage 3: SuperPoint+LightGlue candidates for the full hard set.

Run as one shard:  worker.py match --shard k --of N
Each shard writes STORE/match/<index>.json for the frames it owns.
No GT is read here; the PnP, gates and consensus rule are the frozen ones.
"""
from common import *
import argparse
import sys
import time
from collections import OrderedDict

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--shard',type=int,default=0)
parser.add_argument('--of',type=int,default=1)
parser.add_argument('--limit',type=int,default=0)
argument=parser.parse_args()

denied=deny_GT()
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(VENDOR_PY))
import torch
import cv2
from PIL import Image
from scipy.spatial import cKDTree
torch.hub.set_dir(str(HUB))
# identical determinism settings to the frozen four-frame matcher: single-threaded
# OpenCV RANSAC and the same torch seed, otherwise solve_pnp returns different
# poses for the same correspondences.
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE=='cuda':
    torch.cuda.set_per_process_memory_fraction(.1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
torch.set_num_threads(1)
torch.manual_seed(20)
cv2.setNumThreads(1)
from lightglue import LightGlue,SuperPoint

EXTRA=PLAN.get('lightglue_weights',{})
for name,digest in {'superpoint_v1.pth':'52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e','superpoint_lightglue_v0-1_arxiv.pth':'6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a'}.items():
    assert sha(HUB_CHECKPOINTS/name)==digest,(name,sha(HUB_CHECKPOINTS/name))
assert sha(VENDOR/'lightglue'/'lightglue.py')=='dcb75b9cad1985c5e7a537c512d4318d47fbcc99df75b39a513de60e027c039f'
assert sha(VENDOR_PY/'kornia'/'__init__.py')=='8718bb5a91e58019bd51c34741bda74bb399b23d3d63883a3b0c8384470e77c4'

MATCHER=PLAN['matcher']
PNP=PLAN['PnP']
ASSOCIATION_RADIUS=float(MATCHER['association_radius_px'])
IMAGE_EDGE=560
AXIS=140
MATRIX=model_matrix()

library=load(STORE/'library.json')
positions=np.asarray([r['body_position_world'] for r in library['references']])
queries=load(STORE/'queries.json')['queries']
T_BC=np.asarray(library['T_BC'])
(STORE/'match').mkdir(parents=True,exist_ok=True)

extractor=SuperPoint(max_num_keypoints=int(MATCHER['max_num_keypoints']) if 'max_num_keypoints' in MATCHER else 1024).eval().to(DEVICE)
matcher=LightGlue(features='superpoint',flash=False,mp=False).eval().to(DEVICE)
for module in (extractor,matcher):
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def model_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        image=image.transpose(Image.Transpose.ROTATE_270)
        w,h=image.size
        scale=min(1.,IMAGE_EDGE/max(w,h))
        rw,rh=round(w*scale),round(h*scale)
        image=image.resize((rw,rh),Image.Resampling.BICUBIC)
        cw,ch=rw//14*14,rh//14*14
        left,top=(rw-cw)//2,(rh-ch)//2
        image=image.crop((left,top,left+cw,top+ch))
        return np.asarray(image).copy()


def extract(path):
    array=model_image(path)
    x=torch.from_numpy(array).permute(2,0,1).float().to(DEVICE)/255
    with torch.inference_mode():
        return extractor.extract(x,resize=None)


features=OrderedDict()


def cached_features(path,key):
    if key in features:
        value=features.pop(key);features[key]=value;return value
    value=extract(path)
    features[key]=value
    if len(features)>192:
        features.popitem(last=False)
    return value


projections=OrderedDict()


def cached_projection(reference):
    index=reference['index']
    if index in projections:
        value=projections.pop(index);projections[index]=value;return value
    path=STORE/'ref-projection'/('%06d-%s.npz'%(index,reference['scan_sha256'][:16]))
    with np.load(path,allow_pickle=False) as z:
        value=dict(model=z['model'].astype(np.float64),world=z['world'].astype(np.float64))
    value['tree']=cKDTree(value['model'])
    projections[index]=value
    if len(projections)>192:
        projections.popitem(last=False)
    return value


started=time.monotonic()
processed=0
statistics=dict(frames=0,candidates=0,accepted=0,qualified=0,lightglue_matches=0,bound=0,selected=0,pnp_calls=0)
for query in queries:
    if query['index']%argument.of!=argument.shard:
        continue
    if argument.limit and processed>=argument.limit:
        break
    guard()
    out_path=STORE/'match'/('%06d.json'%query['index'])
    if out_path.is_file():
        processed+=1
        continue
    q_raw_matrix=MATRIX
    qfeatures=cached_features(query['image_path'],'q%d'%query['index'])
    q_model=qfeatures['keypoints'][0].cpu().numpy().astype(np.float64)
    q_raw=model_to_raw(q_raw_matrix,q_model)
    _,q_patch=patch_of(q_model)
    pair_references=[]
    per_pair={}
    matches_by_reference={}
    solutions=[]
    for ri in query['pair_references']:
        reference=library['references'][ri]
        table=cached_projection(reference)
        rfeatures=cached_features(reference['image_path'],'r%d'%ri)
        r_model=rfeatures['keypoints'][0].cpu().numpy().astype(np.float64)
        with torch.inference_mode():
            result=matcher({'image0':qfeatures,'image1':rfeatures})
        matches=result['matches'][0].cpu().numpy().astype(np.int64)
        scores=result['scores'][0].cpu().numpy().astype(np.float64)
        statistics['lightglue_matches']+=int(len(matches))
        if not len(matches):
            per_pair[ri]=dict(matches=0,bound=0,selected=0)
            continue
        reference_uv=r_model[matches[:,1]]
        distance,nearest=table['tree'].query(reference_uv)
        keep=distance<=ASSOCIATION_RADIUS
        bound=int(keep.sum())
        statistics['bound']+=bound
        entry=dict(matches=int(len(matches)),bound=bound,selected=0)
        if not bound:
            per_pair[ri]=entry
            continue
        ids=matches[keep,0]
        aid=nearest[keep]
        score=scores[keep]
        order=np.argsort(-score,kind='stable')
        _,first=np.unique(ids[order],return_index=True)
        chosen=order[first]
        ids,aid,score=ids[chosen],aid[chosen],score[chosen]
        entry['selected']=int(len(ids))
        statistics['selected']+=int(len(ids))
        per_pair[ri]=entry
        pair_references.append(ri)
        matches_by_reference[ri]=(ids,aid,score)
        statistics['pnp_calls']+=1
        solution=solve_pnp(table['world'][aid],q_raw[ids],ids,T_BC)
        if solution is not None:
            solutions.append(dict(channel='pair',reference_ids=[int(ri)],pose=solution['pose'].tolist(),unique_query_inliers=solution['unique_query_inliers'],inlier_ratio=solution['inlier_ratio'],median_reprojection_px=solution['median_reprojection_px'],coverage_cells=solution['coverage_cells'],qualified=bool(solution['qualified'])))
    clusters=[]
    for ri in sorted(matches_by_reference,key=lambda i:-float(matches_by_reference[i][2].max())):
        distances=[float(np.linalg.norm(positions[ri]-positions[group[0]])) for group in clusters]
        near=int(np.argmin(distances)) if distances else None
        if near is None or distances[near]>MATCHER['local_region_radius_m']:
            clusters.append([ri])
        else:
            clusters[near].append(ri)
    for cluster in clusters:
        records=[]
        for ri in cluster:
            ids,aid,score=matches_by_reference[ri]
            table=cached_projection(library['references'][ri])
            records.extend((int(q),float(s),int(ri),int(a)) for q,s,a in zip(ids,score,aid))
        if len(records)<8:
            continue
        records.sort(key=lambda r:(-r[1],r[0],r[2]))
        seen=set()
        chosen=[]
        for record in records:
            if record[0] not in seen:
                chosen.append(record);seen.add(record[0])
        ids=np.array([r[0] for r in chosen])
        world=np.array([cached_projection(library['references'][r[2]])['world'][r[3]] for r in chosen])
        statistics['pnp_calls']+=1
        solution=solve_pnp(world,q_raw[ids],ids,T_BC)
        if solution is None:
            continue
        observed=[chosen[j][2] for j in solution['inlier_indices']]
        inlier_references=sorted({ri for ri in observed if observed.count(ri)>=3})
        solutions.append(dict(channel='pair_plus_local_region',reference_ids=inlier_references,pose=solution['pose'].tolist(),unique_query_inliers=solution['unique_query_inliers'],inlier_ratio=solution['inlier_ratio'],median_reprojection_px=solution['median_reprojection_px'],coverage_cells=solution['coverage_cells'],qualified=bool(solution['qualified'])))
    qualified=[s for s in solutions if s['qualified']]
    for solution in qualified:
        p=np.asarray(solution['pose'])
        support=set()
        for other in qualified:
            op=np.asarray(other['pose'])
            translation=float(np.linalg.norm(p[:3,3]-op[:3,3]))
            rotation=float(np.rad2deg(np.arccos(np.clip((np.trace(p[:3,:3].T@op[:3,:3])-1)/2,-1,1))))
            if translation<=PNP['consensus_translation_m'] and rotation<=PNP['consensus_rotation_deg']:
                support.update(other['reference_ids'])
        solution['consensus_reference_count']=len(support)
    accepted=[s for s in qualified if s.get('consensus_reference_count',0)>=PNP['min_supporting_reference_images']]
    accepted.sort(key=lambda s:(-s['unique_query_inliers'],-s['coverage_cells'],s['median_reprojection_px']))
    selected=accepted[0] if accepted else None
    predicted=np.asarray(selected['pose']) if selected else np.asarray(query['original_SC2'])
    statistics['frames']+=1
    statistics['candidates']+=len(solutions)
    statistics['qualified']+=len(qualified)
    statistics['accepted']+=1 if selected is not None else 0
    save(out_path,dict(index=query['index'],key=query['key'],pair_references=query['pair_references'],per_pair={str(k):v for k,v in per_pair.items()},hypotheses=solutions,accepted=selected is not None,selected=selected,predicted_pose=predicted.tolist()))
    processed+=1
    if processed%25==0:
        print(json.dumps(dict(shard=argument.shard,processed=processed,seconds=time.monotonic()-started,**statistics)),flush=True)
assert not denied
print(json.dumps(dict(shard=argument.shard,complete=True,processed=processed,seconds=time.monotonic()-started,**statistics)))
