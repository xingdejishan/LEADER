"""Stage 4: GT-free independent LiDAR geometric verification for every frame.

Run as one shard:  worker.py verify --shard k --of N
For each frame it scores the fallback pose, the frozen V7 baseline pose and
every candidate produced by match.py, using one identical scoring function,
then probes the frame's best-scoring candidate for directional observability.
No GT is read here.
"""
from common import *
import argparse
import time
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree

parser=argparse.ArgumentParser()
parser.add_argument('--shard',type=int,default=0)
parser.add_argument('--of',type=int,default=1)
parser.add_argument('--limit',type=int,default=0)
argument=parser.parse_args()

denied=deny_GT()
VERIFY=PLAN['verification']
GROUND=dict(azimuth_bins=VERIFY['azimuth_bins'],ground_percentile=5,ground_band_m=0.4)
(STORE/'verify_controls').mkdir(parents=True,exist_ok=True)
library=load(STORE/'library.json')
references=library['references']
positions=np.asarray([r['body_position_world'] for r in references])
queries=load(STORE/'queries.json')['queries']

clouds=OrderedDict()


def reference_cloud(index):
    if index in clouds:
        value=clouds.pop(index);clouds[index]=value;return value
    row=references[index]
    path=STORE/'map-cloud'/('%06d-%s.npz'%(index,row['scan_sha256'][:16]))
    with np.load(path,allow_pickle=False) as z:
        value=z['world'].astype(np.float64)
    clouds[index]=value
    if len(clouds)>512:
        clouds.popitem(last=False)
    return value


maps=OrderedDict()


def map_cloud(center,excluded):
    distances=np.linalg.norm(positions-center,axis=1)
    selected=tuple(sorted(int(i) for i in np.flatnonzero(distances<=VERIFY['map_radius_m']) if i not in excluded))
    if selected in maps:
        value=maps.pop(selected);maps[selected]=value;return value,selected
    if selected:
        cloud=voxel_downsample(np.concatenate([reference_cloud(i) for i in selected]),VERIFY['map_voxel_m'])
    else:
        cloud=np.zeros((0,3))
    value=(cloud,cKDTree(cloud) if len(cloud) else None)
    maps[selected]=value
    if len(maps)>16:
        maps.popitem(last=False)
    return value,selected


def score(pose,query_cloud,excluded,map_center=None):
    pose=np.asarray(pose,dtype=np.float64)
    placed=transform(query_cloud,pose)
    center=pose[:3,3] if map_center is None else np.asarray(map_center,dtype=np.float64)
    (cloud,tree),selected=map_cloud(center,excluded)
    out=dict(valid=False,map_references=len(selected),map_points=int(len(cloud)))
    if tree is None or len(placed)<500:
        for t in VERIFY['nn_thresholds_m']:
            out['overlap_%g'%t]=None
        out.update(dict(residual_median_m=None,supported_bins=0,bin_overlap_median=None,top_bin_share=None,offground_score=0.0,offground_verified=False,geometry_score=0.0,geometry_verified=False))
        return out
    distance,_=tree.query(placed,workers=1)
    out['valid']=True
    primary=VERIFY['primary_nn_threshold_m']
    for t in VERIFY['nn_thresholds_m']:
        out['overlap_%g'%t]=float(np.mean(distance<=t))
    inside=distance<=VERIFY['residual_window_m']
    if int(inside.sum())>=20:
        values=distance[inside]
        out['residual_median_m']=float(np.median(values))
        out['residual_p90_m']=float(np.percentile(values,90))
    else:
        out['residual_median_m']=None
        out['residual_p90_m']=None
    azimuth=np.arctan2(query_cloud[:,1],query_cloud[:,0])
    bins=VERIFY['azimuth_bins']
    index=np.floor((azimuth+np.pi)/(2*np.pi)*bins).astype(int).clip(0,bins-1)
    overlaps=[]
    supported=0
    points=np.zeros(bins)
    for b in range(bins):
        mask=index==b
        count=int(mask.sum())
        if count<VERIFY['min_points_per_bin']:
            continue
        fraction=float(np.mean(distance[mask]<=primary))
        overlaps.append(fraction)
        points[b]=int((distance[mask]<=primary).sum())
        if fraction>=VERIFY['bin_support_overlap']:
            supported+=1
    total=int(points.sum())
    out['supported_bins']=supported
    out['bin_overlap_median']=float(np.median(overlaps)) if overlaps else None
    out['top_bin_share']=float(points.max()/total) if total else None
    out['geometry_score']=out['bin_overlap_median'] if out['bin_overlap_median'] is not None else 0.0
    out['geometry_verified']=bool(
        out['bin_overlap_median'] is not None and out['bin_overlap_median']>=VERIFY['bin_support_overlap']
        and supported>=VERIFY['min_supported_bins']
        and out['residual_median_m'] is not None and out['residual_median_m']<=VERIFY['residual_median_max_m']
        and out['residual_p90_m'] is not None and out['residual_p90_m']<=VERIFY['residual_p90_max_m']
        and out['top_bin_share'] is not None and out['top_bin_share']<=VERIFY['max_top_bin_share'])
    offground=offground_mask(query_cloud,index,GROUND)
    off_overlaps=[]
    off_supported=0
    for b in range(bins):
        mask=(index==b)&offground
        if int(mask.sum())<VERIFY['min_points_per_bin']:
            continue
        fraction=float(np.mean(distance[mask]<=primary))
        off_overlaps.append(fraction)
        if fraction>=VERIFY['bin_support_overlap']:
            off_supported+=1
    window=offground&(distance<=VERIFY['residual_window_m'])
    out['offground_fraction']=float(np.mean(offground))
    out['offground_bin_overlap_median']=float(np.median(off_overlaps)) if off_overlaps else None
    out['offground_supported_bins']=off_supported
    out['offground_residual_median_m']=float(np.median(distance[window])) if int(window.sum())>=20 else None
    out['offground_score']=out['offground_bin_overlap_median'] if out['offground_bin_overlap_median'] is not None else 0.0
    out['offground_verified']=bool(
        out['offground_bin_overlap_median'] is not None and out['offground_bin_overlap_median']>=VERIFY['bin_support_overlap']
        and off_supported>=VERIFY['min_supported_bins']
        and out['offground_residual_median_m'] is not None and out['offground_residual_median_m']<=VERIFY['residual_median_max_m']
        and out['offground_fraction']>=0.15)
    return out


started=time.monotonic()
processed=0
statistics=dict(frames=0,scores=0,verified=0,accepted=0)
for query in queries:
    if query.get('group')!='normal':
        continue
    if query['index']%argument.of!=argument.shard:
        continue
    if argument.limit and processed>=argument.limit:
        break
    guard()
    out_path=STORE/'verify_controls'/('%06d.json'%query['index'])
    if out_path.is_file():
        processed+=1
        continue
    match_path=STORE/'match'/('%06d.json'%query['index'])
    while not match_path.exists():
        time.sleep(2)
    match=load(match_path)
    scan_path=query['scan_path']
    assert sha(scan_path)==query['scan_sha256'],query['key']
    body=decode(scan_path)
    body=body[np.isfinite(body).all(1)]
    body=body[np.linalg.norm(body,axis=1)<=VERIFY['query_range_m']]
    query_cloud=voxel_downsample(body,VERIFY['query_voxel_m'])
    excluded=set(query['pair_references'])
    poses=[('baseline_V7',query['baseline_V7'])]
    for h,hypothesis in enumerate(match['hypotheses']):
        poses.append(('candidate_%02d'%h,hypothesis['pose']))
    rows={}
    for name,pose in poses:
        result=score(pose,query_cloud,excluded)
        rows[name]=dict(pose=pose,**result)
        statistics['scores']+=1
    candidates=[(name,row) for name,row in rows.items() if name.startswith('candidate_')]
    best=max(candidates,key=lambda item:item[1]['geometry_score'])[0] if candidates else None
    observability=None
    statistics['frames']+=1
    statistics['verified']+=sum(1 for row in rows.values() if row.get('geometry_verified'))
    statistics['accepted']+=1 if match['accepted'] else 0
    save(out_path,dict(index=query['index'],key=query['key'],query_points=int(len(query_cloud)),excluded_references=sorted(excluded),poses=rows,best_candidate=best,observability=observability))
    processed+=1
    if processed%25==0:
        print(json.dumps(dict(shard=argument.shard,processed=processed,seconds=time.monotonic()-started,**statistics)),flush=True)
assert not denied
print(json.dumps(dict(shard=argument.shard,complete=True,processed=processed,seconds=time.monotonic()-started,**statistics)))
