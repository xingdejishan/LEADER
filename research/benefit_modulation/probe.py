import json
import shutil
import numpy as np
import torch
from scipy.spatial import cKDTree
from probe_data import OUT, HERE, run


def normalize(x):
    return torch.nn.functional.normalize(torch.as_tensor(x, device='cuda', dtype=torch.float32), dim=-1)


def load(row):
    with np.load(OUT/'lidar'/(row['frame_id']+'.npz')) as l, np.load(OUT/'visual_raw'/(row['frame_id']+'.npz')) as v:
        mask = v['valid']
        return dict(lidar=l['features'][mask], image=v['image'][mask], xyz=l['representative_world'][mask], pose=l['camera_pose'], voxel=np.flatnonzero(mask), total=len(mask))


def shuffle(image, frame, seed):
    rng = np.random.default_rng(seed+int(frame)%1000000007)
    return image[rng.permutation(len(image))]


def aggregate(records):
    keys = ['lidar','aligned','shuffle2089','shuffle2090','shuffle2091']
    total = sum(r['queries'] for r in records)
    ambiguous = sum(r['ambiguous'] for r in records)
    return dict(frames=len(records), query_points=total, geometric_positive_any=sum(r['positive_any'] for r in records),
                candidate_positive=sum(r['positive16'] for r in records), ambiguous=ambiguous,
                ambiguous_fraction=ambiguous/max(total,1), contributing_frames=sum(r['ambiguous']>0 for r in records),
                accuracy={k:sum(r['hits'][k] for r in records)/max(ambiguous,1) for k in keys},
                mrr={k:sum(r['mrr'][k] for r in records)/max(ambiguous,1) for k in keys},
                rescue={k:sum(r['rescue'][k] for r in records) for k in keys[1:]},
                damage={k:sum(r['damage'][k] for r in records) for k in keys[1:]})


def interval(records, opponent):
    counts = np.array([r['ambiguous'] for r in records])
    delta = np.array([r['hits']['aligned'] - (r['hits']['lidar'] if opponent=='lidar' else np.mean([r['hits'][f'shuffle{s}'] for s in [2089,2090,2091]])) for r in records])
    rng = np.random.default_rng(271828)
    indices = rng.integers(len(records), size=(10000,len(records)))
    denominator = counts[indices].sum(1)
    values = delta[indices].sum(1)[denominator>0]/denominator[denominator>0]
    return np.percentile(values,[2.5,97.5]).tolist() if len(values) else None


def main():
    torch.set_num_threads(4)
    rows = json.loads((OUT/'manifest.json').read_text())
    refs = [r for r in rows if r['split']=='reference']
    queries = [r for r in rows if r['split']=='query']
    reference = [load(r) for r in refs]
    lf = normalize(np.concatenate([r['lidar'] for r in reference]))
    vf = normalize(np.concatenate([r['image'] for r in reference]))
    xyz = np.concatenate([r['xyz'] for r in reference])
    frame_indices = np.concatenate([np.full(len(r['xyz']),i) for i,r in enumerate(reference)])
    fi = torch.tensor(frame_indices, device='cuda')
    poses = np.stack([r['pose'] for r in reference])
    shuffled = {s:normalize(np.concatenate([shuffle(d['image'],r['frame_id'],s) for r,d in zip(refs,reference)])) for s in [2089,2090,2091]}
    destination = OUT/'probe'
    destination.mkdir(exist_ok=True)
    records = []
    for index,row in enumerate(queries):
        q = load(row)
        distances = np.linalg.norm(poses[:,:3,3]-q['pose'][:3,3],axis=1)
        angles = np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',poses[:,:3,:3],q['pose'][:3,:3])-1)/2,-1,1)))
        recent = np.array([r['session_id']==row['session_id'] and abs(int(r['frame_id'])-int(row['frame_id']))<10000000 for r in refs])
        eligible = ~(recent | ((distances<5)&(angles<15)))
        point_eligible = eligible[frame_indices]
        assert point_eligible.sum()>=16
        positive_any = cKDTree(xyz[point_eligible]).query(q['xyz'])[0]<=.5
        ql,qv = normalize(q['lidar']),normalize(q['image'])
        qs = {s:normalize(shuffle(q['image'],row['frame_id'],s)) for s in shuffled}
        masks, correct, reciprocal, candidates_all, choices_all, spatial_all = [],{}, {}, [], {}, []
        keys = ['lidar','aligned']+[f'shuffle{s}' for s in shuffled]
        for k in keys:
            correct[k],reciprocal[k],choices_all[k] = [],[],[]
        positive_count=0
        for start in range(0,len(ql),64):
            end = min(start+64,len(ql))
            similarity = ql[start:end] @ lf.T
            similarity[:,~torch.tensor(eligible,device='cuda')[fi]] = -float('inf')
            scores, candidates = similarity.topk(16,dim=-1,sorted=True)
            candidate_numpy = candidates.cpu().numpy()
            spatial = np.linalg.norm(xyz[candidate_numpy]-q['xyz'][start:end,None],axis=-1)
            positives = spatial<=.5
            assert not (positives.any(1) & ~positive_any[start:end]).any()
            positive_count += int(positives.any(1).sum())
            ambiguous = positives.any(1)&(spatial>=2).any(1)&((scores[:,0]-scores[:,1]).cpu().numpy()<=.02)
            rankings = dict(lidar=torch.arange(16,device='cuda').expand(end-start,-1),
                            aligned=(qv[start:end,None]*vf[candidates]).sum(-1).argsort(dim=-1,descending=True,stable=True))
            for s in shuffled:
                rankings[f'shuffle{s}']=(qs[s][start:end,None]*shuffled[s][candidates]).sum(-1).argsort(dim=-1,descending=True,stable=True)
            for k,ranking in rankings.items():
                order = ranking.cpu().numpy()
                ranked = np.take_along_axis(positives,order,axis=1)
                hit = ranked[:,0]
                rr = np.where(ranked.any(1),1/(ranked.argmax(1)+1),0)
                correct[k].extend(hit.tolist())
                reciprocal[k].extend(rr.tolist())
                choices_all[k].extend(order[:,0].tolist())
            masks.extend(ambiguous.tolist())
            candidates_all.extend(candidate_numpy.tolist())
            spatial_all.extend(spatial.tolist())
        masks = np.array(masks,dtype=bool)
        correct = {k:np.array(v,dtype=bool) for k,v in correct.items()}
        r = dict(frame_id=row['frame_id'],date=row['session_id'],queries=len(ql),total_voxels=q['total'],
                 eligible_reference_frames=int(eligible.sum()),positive_any=int(positive_any.sum()),positive16=positive_count,
                 ambiguous=int(masks.sum()),hits={k:int(v[masks].sum()) for k,v in correct.items()},
                 mrr={k:float(np.array(v)[masks].sum()) for k,v in reciprocal.items()},
                 rescue={k:int((v&~correct['lidar']&masks).sum()) for k,v in correct.items() if k!='lidar'},
                 damage={k:int((~v&correct['lidar']&masks).sum()) for k,v in correct.items() if k!='lidar'})
        np.savez_compressed(destination/(row['frame_id']+'.npz'),query_voxel=q['voxel'],candidate_global_indices=np.asarray(candidates_all),spatial_distance=np.asarray(spatial_all),ambiguous=masks,**{k:np.asarray(v) for k,v in choices_all.items()})
        records.append(r)
        if index%10==0:
            print('probe',index+1,len(queries),r['ambiguous'],flush=True)
    overall = aggregate(records)
    dates = {date:aggregate([r for r in records if r['date']==date]) for date in sorted({r['date'] for r in records})}
    ci = {k:interval(records,k) for k in ['lidar','shuffled']}
    viable = overall['ambiguous']>=200 and overall['contributing_frames']>=20 and sum(r['ambiguous']>0 for r in dates.values())>=2
    def beats(r):
        return all(r['accuracy']['aligned']>v for k,v in r['accuracy'].items() if k!='aligned')
    passed = viable and beats(overall) and all(beats(r) for r in dates.values() if r['ambiguous']>=50) and all(v is not None and v[0]>0 for v in ci.values())
    run.save_json(destination/'frames.json',records)
    run.save_json(destination/'summary.json',dict(overall=overall,dates=dates,paired_frame_bootstrap95=ci,viable=viable,passed=passed))
    run.save_json(destination/'reference_index.json',[dict(frame_id=r['frame_id'],count=len(d['xyz']),voxel=d['voxel'].tolist()) for r,d in zip(refs,reference)])
    shutil.copytree(destination,HERE/'results/crossframe_probe',dirs_exist_ok=True)
    for name in ['manifest.json','protocol.json']:
        shutil.copy2(OUT/name,HERE/'results/crossframe_probe'/name)
    print(json.dumps(dict(overall=overall,dates=dates,ci=ci,viable=viable,passed=passed),indent=2))


if __name__=='__main__':
    main()
