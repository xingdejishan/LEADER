import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import torch
import experiment as e
from certificate_fusion import OUT as TRAIN

OUT=e.HERE/'results/teacher_realization'
TAU=1e-7


def main():
    OUT.mkdir(exist_ok=True)
    protocol=dict(source='befb9c1',population='Exact retained1500 correction and2255 zero fit targets; all six selected best.pt; no selection or training',
        tau=TAU,normalization='Applied float32 fused-minus-original converted to float64, divided by float64 original feature norm',
        ranking='Cosine similarity; earliest fixed candidate index within tau of maximum. Strict correction requires max positive minus max nonpositive >tau; gray and tie separate.',
        margins='Also save unnormalized-query margins for selected teacher positive against N and N+G; common original-norm scaling as teacher certificate.',
        fit_bins=[0,.1,.25,.5,1,2],near_zero='Student norm <=0.1 teacher norm, only correction targets',
        boundary_check='Teacher-positive per-competitor margin divided by difference-vector norm gives sufficient perturbation radius; compare residual error and signed margin changes, not assume small gamma caused failure',
        controls='Same targets, reference library, protection, shuffle seeds, checkpoints and masks; no developer/query GT training',
        teacher_manifest_sha256=e.run.digest(TRAIN/'teacher_verified.json'))
    e.run.save_json(OUT/'protocol.json',protocol)
    p=json.loads((TRAIN/'protocol.json').read_text()); fit=[r for r in p['rows'] if r['role']=='fit']
    anchors=[]
    for r in fit:
        with np.load(e.CACHE/'lidar'/(r['frame_id']+'.npz')) as l,np.load(e.CACHE/'visual_raw'/(r['frame_id']+'.npz')) as v:
            anchors.append(l['features'][v['valid']])
    anchors=np.concatenate(anchors).astype(float); anchors/=np.linalg.norm(anchors,axis=1,keepdims=True)
    audit=json.loads((TRAIN/'audit.json').read_text())
    hashes=json.loads((TRAIN/'teacher_verified.json').read_text())['teacher_sha256']
    heads={}
    for seed in e.SEEDS:
        for arm in ['aligned','shuffled']:
            name=f'{arm}_{seed}'; file=TRAIN/name/'best.pt'
            assert e.run.digest(file)==audit['runs'][name]['checkpoint_sha256']
            h=e.AnchoredFusion().cuda().eval(); h.load_state_dict(torch.load(file))
            heads[name]=(h,arm,seed)
    torch.set_num_threads(4)
    metadata=[]; teacher=[]; teacher_scores=[]; original_scores=[]; target_scores=[]; labels=[]; positive_choice=[]; robustness=[]; queries=[]; teacher_lengths=[]
    deltas={n:[] for n in heads}; scores={n:[] for n in heads}
    for frame,r in enumerate(fit):
        file=TRAIN/'teacher'/(r['frame_id']+'.npz')
        assert e.run.digest(file)==hashes[r['frame_id']]
        with np.load(file) as t,np.load(e.CACHE/'lidar'/(r['frame_id']+'.npz')) as l,np.load(e.CACHE/'visual_raw'/(r['frame_id']+'.npz')) as v:
            kind=t['kind']; ids=np.flatnonzero(kind>0)
            if not len(ids):
                continue
            f=torch.tensor(l['features'],device='cuda'); image=torch.tensor(v['image'],device='cuda')
            editable=torch.tensor(t['editable'],device='cuda'); valid=torch.tensor(v['valid'],device='cuda')
            assert not ((kind>0)&~t['editable']).any()
            target=t['targets'][ids].astype(float)
            query_rows=np.searchsorted(t['indices'],ids)
            assert np.array_equal(t['indices'][query_rows],ids)
            pos=t['positive'][query_rows]; neg=t['negative'][query_rows]
            ks=anchors[t['candidates'][query_rows]]
            ff=f[torch.tensor(ids,device='cuda')]; raw=ff.cpu().numpy().astype(float)
            lengths=np.linalg.norm(raw,axis=1,keepdims=True); x=raw/lengths
            with torch.no_grad():
                teach=(ff+torch.tensor(target,device='cuda',dtype=torch.float32)*ff.norm(dim=-1,keepdim=True)).cpu().numpy().astype(float)/lengths
                for name,(head,arm,seed) in heads.items():
                    current=image
                    if arm=='shuffled':
                        vi=torch.where(valid)[0]; order=torch.arange(len(f),device='cuda')
                        rng=np.random.default_rng(seed+int(r['frame_id'])%1000000007)
                        order[vi]=vi[torch.tensor(rng.permutation(len(vi)),device='cuda')]
                        current=image[order]
                    fused=head(f,current,editable)
                    assert torch.equal(fused[~editable],f[~editable])
                    actual=fused[torch.tensor(ids,device='cuda')].cpu().numpy().astype(float)/lengths
                    deltas[name].append(actual-x)
                    scores[name].append(np.einsum('nd,nkd->nk',actual,ks))
            meta=json.loads(file.with_suffix('.json').read_text())
            checks={}
            for c in meta['checks']:
                if c['status']=='certified':
                    checks.setdefault(c['voxel'],[]).append(c)
            ts=np.einsum('nd,nkd->nk',teach,ks)
            for i,voxel in enumerate(ids):
                if kind[voxel]==2:
                    choices=checks[int(voxel)]
                    selected=min(choices,key=lambda c:(round(float(t['d'+c['key']]@t['d'+c['key']]),12),c['positive']))['positive']
                    a=ks[i,selected]-ks[i,~pos[i]]
                    margins=(x[i]+target[i])@a.T
                    radii=np.divide(margins,np.linalg.norm(a,axis=1),out=np.full(len(a),np.inf),where=np.linalg.norm(a,axis=1)>0)
                    radius=float(radii.min())
                    assert ts[i,selected]-ts[i,~pos[i]].max()>TAU
                else:
                    selected=0; radius=0.
                    assert pos[i,0] and not target[i].any()
                metadata.append(dict(frame_id=r['frame_id'],date=r['session_id'],voxel=int(voxel),kind=int(kind[voxel])))
                positive_choice.append(int(selected)); robustness.append(radius)
            teacher.append(target); teacher_scores.append(ts); original_scores.append(np.einsum('nd,nkd->nk',x,ks))
            queries.append(x); teacher_lengths.append(np.linalg.norm(teach,axis=1))
            target_scores.append(np.einsum('nd,nkd->nk',x+target,ks)); labels.append(np.where(pos,1,np.where(neg,-1,0)))
        if frame%100==0:
            print('frames',frame+1,flush=True)
    kind=np.array([r['kind'] for r in metadata]); assert (kind==2).sum()==1500 and (kind==1).sum()==2255
    np.savez_compressed(OUT/'reference.npz',target=np.concatenate(teacher),teacher_scores=np.concatenate(teacher_scores),
        original_scores=np.concatenate(original_scores),target_scores=np.concatenate(target_scores),labels=np.concatenate(labels),
        selected_positive=np.array(positive_choice),robustness_radius=np.array(robustness),kind=kind,x=np.concatenate(queries),teacher_norm=np.concatenate(teacher_lengths))
    for name in heads:
        np.savez_compressed(OUT/(name+'.npz'),delta=np.concatenate(deltas[name]),scores=np.concatenate(scores[name]))
    e.run.save_json(OUT/'queries.json',metadata)
    e.run.save_json(OUT/'inputs.json',dict(weights={n:audit['runs'][n]['checkpoint_sha256'] for n in heads},teacher_sha256=hashes))
    print('INFERENCE COMPLETE',flush=True)


if __name__=='__main__':
    main()
