import argparse
import json

import numpy as np
import torch

import visual_relation as v


def prepare():
    p=json.loads((v.OUT/'protocol.json').read_text())
    rows=[r for r in p['rows'] if r['role']=='fit']
    assert len(rows)==578
    query_xyz=[]; anchor_xyz=[]; query_images=[]; anchor_images=[]
    allowed=[]; offset=0
    for r in rows:
        fid=r['frame_id']
        with np.load(v.e.CACHE/'lidar'/(fid+'.npz')) as f, np.load(v.e.CACHE/'visual_raw'/(fid+'.npz')) as a, np.load(v.e.OUT/'candidates'/(fid+'.npz')) as b:
            xyz=f['representative_world']; valid=a['valid']; image=a['image']
            query_xyz.append(xyz); anchor_xyz.append(xyz[valid]); query_images.append(image); anchor_images.append(image[valid])
            for i in np.flatnonzero((b['gap']<=.02)&b['positive'].any(1)&b['negative'].any(1)):
                for pos in np.flatnonzero(b['positive'][i]):
                    for neg in np.flatnonzero(b['negative'][i]):
                        allowed.append((offset+int(b['indices'][i]),int(b['candidates'][i,pos]),int(b['candidates'][i,neg])))
            offset+=len(xyz)
    qxyz=np.concatenate(query_xyz); axyz=np.concatenate(anchor_xyz)
    qi=np.concatenate(query_images).astype(float); ai=np.concatenate(anchor_images).astype(float)
    qi/=np.maximum(np.linalg.norm(qi,axis=1,keepdims=True),1e-12); ai/=np.maximum(np.linalg.norm(ai,axis=1,keepdims=True),1e-12)
    with np.load(v.OUT/'relations.npz') as a:
        t=a['triples']; selected=a['V']; shuffled=a['S']
        assert np.array_equal(np.asarray(allowed),t)
        max_pos=0.; min_neg=float('inf')
        for start in range(0,len(t),4096):
            q,pp,nn=t[start:start+4096].T
            dp=np.linalg.norm(qxyz[q]-axyz[pp],axis=1); dn=np.linalg.norm(qxyz[q]-axyz[nn],axis=1)
            max_pos=max(max_pos,float(dp.max())); min_neg=min(min_neg,float(dn.min()))
            assert (dp<=.5).all() and (dn>=2).all()
            flag=(qi[q]*ai[pp]).sum(1)>(qi[q]*ai[nn]).sum(1)
            assert np.array_equal(flag,selected[start:start+4096])
        for s in np.unique(a['strata']):
            mask=a['strata']==s
            assert selected[mask].sum()==shuffled[mask].sum()
        count=int(selected.sum())
    e=dict(pool=len(allowed),selected=count,max_positive_m=max_pos,min_negative_m=min_neg,relations_sha256=v.e.run.digest(v.OUT/'relations.npz'),candidate_membership=True,training_only=True,visual_flags_recomputed=True,stratum_counts_matched=True)
    v.e.run.save_json(v.OUT/'relation_verification.json',e)
    print(e,flush=True)


def final():
    p=json.loads((v.OUT/'protocol.json').read_text())
    baseline=json.loads((v.OUT/'baseline_internal.json').read_text())
    base=float(np.mean([r['standard'][0] for r in baseline]))
    states=[]; records={}
    for arm in ['B1','V','S']:
        folder=v.OUT/arm
        logs=json.loads((folder/'training.json').read_text())
        assert [x['epoch'] for x in logs]==list(range(1,101))
        selection=json.loads((folder/'selection.json').read_text())
        scores=[(0,base)]+[(x['epoch'],x['internal']['mean'][0]) for x in logs if 'internal' in x]
        assert selection['epoch']==min(scores,key=lambda x:x[1])[0]
        if arm!='B1': assert all(x['relation_gradient']>0 for x in logs)
        state=torch.load(folder/'resume.pt',map_location='cpu'); states.append(state)
        best=torch.load(folder/'best.pt',map_location='cpu')
        assert all(torch.isfinite(t).all() for t in best.values())
        records[arm]=dict(best_sha256=v.e.run.digest(folder/'best.pt'),last_sha256=v.e.run.digest(folder/'last.pt'),selection=selection,epochs=len(logs),updates=json.loads((folder/'complete.json').read_text())['updates'],parameter_count=sum(t.numel() for t in best.values()))
    for s in states[1:]:
        assert s['rng']==states[0]['rng'] and s['rrng']==states[0]['rrng']
        assert s['cursor']==states[0]['cursor'] and np.array_equal(s['schedule'],states[0]['schedule'])
    v.e.run.save_json(v.OUT/'training_verification.json',dict(runs=records,matched_frame_and_relation_schedules=True,protocol_sha256=v.e.run.digest(v.OUT/'protocol.json')))
    print(records,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','final']); args=parser.parse_args()
    globals()[args.stage]()
